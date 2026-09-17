#include <Arduino.h>
#include <ArduinoJson.h>
#include <Audio.h>
#include <HTTPClient.h>
#include <WiFi.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

// MAX98357A I2S wiring. Change these if your board is wired differently.
constexpr int I2S_BCLK_PIN = 26; // MAX98357A BCLK
constexpr int I2S_LRC_PIN = 25;  // MAX98357A LRC / WS
constexpr int I2S_DIN_PIN = 22;  // MAX98357A DIN

// Fill these in here, or define them in platformio.ini build_flags.
#ifndef WIFI_SSID
#define WIFI_SSID "Zone Six 10NTP"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "10nguyentriphuong"
#endif

// Optional: set to your Python web server, for example:
// http://192.168.1.10:8000/status
#ifndef FALL_STATUS_URL
#define FALL_STATUS_URL "https://web.khanhtoan.click/status"
#endif

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint8_t ALERT_REPEAT_COUNT = 5;
constexpr uint8_t AUDIO_VOLUME = 18; // ESP32-audioI2S uses 0..21.
constexpr uint32_t TTS_REPEAT_GAP_MS = 350;
constexpr uint32_t WIFI_RETRY_INTERVAL_MS = 15000;
constexpr uint32_t STATUS_POLL_INTERVAL_MS = 1000;
constexpr uint16_t HTTP_TIMEOUT_MS = 1200;
constexpr size_t SERIAL_LINE_LIMIT = 256;
constexpr uint8_t MAX_TRACKED_CAMERAS = 8;
constexpr size_t SPEECH_TEXT_LIMIT = 160;

enum class AudioEventType : uint8_t
{
  Speech,
  FallAlert,
  FaintAlert
};

enum class AlertStatus : uint8_t
{
  None,
  Fallen,
  Faint
};

struct AudioEvent
{
  AudioEventType type;
  uint8_t cameraNumber;
  uint8_t repeatCount;
  char speech[SPEECH_TEXT_LIMIT];
};

Audio audio;
QueueHandle_t audioQueue = nullptr;
volatile bool speechEnded = false;

String serialLine;
uint32_t lastWifiAttemptMs = 0;
bool wifiWasConnected = false;

uint8_t validCameraNumber(int cameraNumber)
{
  if (cameraNumber < 1)
  {
    return 1;
  }
  if (cameraNumber > 255)
  {
    return 255;
  }
  return static_cast<uint8_t>(cameraNumber);
}

size_t cameraIndex(uint8_t cameraNumber, size_t activeCount)
{
  if (activeCount == 0)
  {
    return 0;
  }

  const size_t index = cameraNumber > 0 ? static_cast<size_t>(cameraNumber - 1) : 0;
  return index < activeCount ? index : activeCount - 1;
}

bool wifiConfigured()
{
  return strlen(WIFI_SSID) > 0;
}

bool statusUrlConfigured()
{
  return strlen(FALL_STATUS_URL) > 0;
}

int firstNumberInText(const String &text)
{
  int value = -1;
  bool readingNumber = false;

  for (size_t i = 0; i < text.length(); ++i)
  {
    const char c = text.charAt(i);
    if (isDigit(c))
    {
      if (!readingNumber)
      {
        value = 0;
        readingNumber = true;
      }
      value = (value * 10) + (c - '0');
    }
    else if (readingNumber)
    {
      break;
    }
  }

  return value;
}

String normalizedStatusText(String text)
{
  text.trim();
  text.replace("_", " ");
  text.replace("-", " ");
  text.replace("=", " ");
  text.replace(":", " ");
  text.replace(";", " ");
  text.replace(",", " ");
  text.toUpperCase();

  while (text.indexOf("  ") >= 0)
  {
    text.replace("  ", " ");
  }

  return text;
}

bool isFallStatus(const String &status)
{
  const String normalized = normalizedStatusText(status);
  return normalized == "FALLEN" ||
         normalized == "NGA";
}

bool isFaintStatus(const String &status)
{
  const String normalized = normalizedStatusText(status);
  return normalized == "FAINT" ||
         normalized == "UNCONSCIOUS" ||
         normalized == "NGAT";
}

bool textContainsFallStatus(const String &payload)
{
  const String normalized = normalizedStatusText(payload);
  return normalized.indexOf("FALLEN") >= 0 ||
         normalized == "NGA" ||
         normalized.indexOf(" NGA ") >= 0 ||
         normalized.endsWith(" NGA") ||
         normalized.startsWith("NGA ");
}

bool textContainsFaintStatus(const String &payload)
{
  const String normalized = normalizedStatusText(payload);
  return normalized.indexOf("FAINT") >= 0 ||
         normalized.indexOf("UNCONSCIOUS") >= 0 ||
         normalized == "NGAT" ||
         normalized.indexOf(" NGAT ") >= 0 ||
         normalized.endsWith(" NGAT") ||
         normalized.startsWith("NGAT ");
}

AlertStatus mergeAlertStatus(AlertStatus current, AlertStatus next)
{
  if (current == AlertStatus::Faint || next == AlertStatus::Faint)
  {
    return AlertStatus::Faint;
  }
  if (current == AlertStatus::Fallen || next == AlertStatus::Fallen)
  {
    return AlertStatus::Fallen;
  }
  return AlertStatus::None;
}

AlertStatus alertStatusFromText(const String &text)
{
  if (isFaintStatus(text))
  {
    return AlertStatus::Faint;
  }
  if (isFallStatus(text))
  {
    return AlertStatus::Fallen;
  }
  return AlertStatus::None;
}

AlertStatus alertStatusFromPayloadText(const String &payload)
{
  if (textContainsFaintStatus(payload))
  {
    return AlertStatus::Faint;
  }
  if (textContainsFallStatus(payload))
  {
    return AlertStatus::Fallen;
  }
  return AlertStatus::None;
}

String jsonString(JsonVariantConst object, const char *primaryKey, const char *secondaryKey = nullptr)
{
  JsonVariantConst value = object[primaryKey];
  if (value.isNull() && secondaryKey != nullptr)
  {
    value = object[secondaryKey];
  }

  if (value.is<const char *>())
  {
    return String(value.as<const char *>());
  }
  if (value.is<String>())
  {
    return value.as<String>();
  }

  return "";
}

int jsonCameraNumber(JsonVariantConst object, uint8_t defaultNumber = 1)
{
  JsonVariantConst cameraNumber = object["camera_number"];
  if (!cameraNumber.isNull())
  {
    return validCameraNumber(cameraNumber.as<int>());
  }

  JsonVariantConst camera = object["camera"];
  if (!camera.isNull())
  {
    if (camera.is<int>())
    {
      return validCameraNumber(camera.as<int>());
    }

    const int parsed = firstNumberInText(camera.as<String>());
    if (parsed >= 0)
    {
      return validCameraNumber(parsed);
    }
  }

  JsonVariantConst cameraId = object["camera_id"];
  if (!cameraId.isNull())
  {
    return validCameraNumber(cameraId.as<int>() + 1);
  }

  JsonVariantConst id = object["id"];
  if (!id.isNull())
  {
    return validCameraNumber(id.as<int>() + 1);
  }

  return defaultNumber;
}

bool queueAudioEvent(const AudioEvent &event)
{
  if (audioQueue == nullptr)
  {
    return false;
  }

  const BaseType_t queued = xQueueSend(audioQueue, &event, 0);
  if (queued == pdTRUE)
  {
    return true;
  }

  Serial.println("[AUDIO] Queue full, speech event was skipped");
  return false;
}

bool queueSpeech(const char *speech, uint8_t repeatCount = 1)
{
  AudioEvent event{};
  event.type = AudioEventType::Speech;
  event.cameraNumber = 0;
  event.repeatCount = repeatCount;
  snprintf(event.speech, sizeof(event.speech), "%s", speech);

  if (queueAudioEvent(event))
  {
    Serial.printf("[AUDIO] Queued speech: %s\n", event.speech);
    return true;
  }

  return false;
}

bool queueFallAlert(uint8_t cameraNumber)
{
  AudioEvent event{};
  event.type = AudioEventType::FallAlert;
  event.cameraNumber = validCameraNumber(cameraNumber);
  event.repeatCount = ALERT_REPEAT_COUNT;

  if (queueAudioEvent(event))
  {
    Serial.printf("[ALERT] Queued audio warning for camera %u\n", event.cameraNumber);
    return true;
  }

  return false;
}

bool queueFaintAlert(uint8_t cameraNumber)
{
  AudioEvent event{};
  event.type = AudioEventType::FaintAlert;
  event.cameraNumber = validCameraNumber(cameraNumber);
  event.repeatCount = ALERT_REPEAT_COUNT;

  if (queueAudioEvent(event))
  {
    Serial.printf("[ALERT] Queued faint audio warning for camera %u\n", event.cameraNumber);
    return true;
  }

  return false;
}

bool queueAlert(AlertStatus status, uint8_t cameraNumber)
{
  if (status == AlertStatus::Faint)
  {
    return queueFaintAlert(cameraNumber);
  }
  if (status == AlertStatus::Fallen)
  {
    return queueFallAlert(cameraNumber);
  }

  return false;
}

AlertStatus jsonNodeAlertStatus(JsonVariantConst node)
{
  AlertStatus status = AlertStatus::None;

  if (node["faint_alert"] | false)
  {
    status = AlertStatus::Faint;
  }
  else if (node["fall_alert"] | false)
  {
    status = AlertStatus::Fallen;
  }

  status = mergeAlertStatus(status, alertStatusFromText(jsonString(node, "status", "event")));
  status = mergeAlertStatus(status, alertStatusFromText(jsonString(node, "status_label", "label")));

  JsonArrayConst tracks = node["tracks"].as<JsonArrayConst>();
  for (JsonObjectConst track : tracks)
  {
    status = mergeAlertStatus(status, alertStatusFromText(jsonString(track, "status", "event")));
    status = mergeAlertStatus(status, alertStatusFromText(jsonString(track, "status_label", "label")));
  }

  return status;
}

void handleJsonEvent(JsonVariantConst root)
{
  if (root["cameras"].is<JsonArrayConst>())
  {
    JsonArrayConst cameras = root["cameras"].as<JsonArrayConst>();
    for (JsonObjectConst camera : cameras)
    {
      queueAlert(jsonNodeAlertStatus(camera), jsonCameraNumber(camera));
    }
    return;
  }

  queueAlert(jsonNodeAlertStatus(root), jsonCameraNumber(root));
}

void handleIncomingPayload(const String &payload)
{
  JsonDocument doc;
  const DeserializationError error = deserializeJson(doc, payload);

  if (!error)
  {
    handleJsonEvent(doc.as<JsonVariantConst>());
    return;
  }

  const AlertStatus status = alertStatusFromPayloadText(payload);
  if (status == AlertStatus::None)
  {
    return;
  }

  const int parsedCamera = firstNumberInText(payload);
  queueAlert(status, parsedCamera > 0 ? parsedCamera : 1);
}

void serviceSerialInput()
{
  while (Serial.available() > 0)
  {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r')
    {
      serialLine.trim();
      if (serialLine.length() > 0)
      {
        handleIncomingPayload(serialLine);
        serialLine = "";
      }
      continue;
    }

    if (serialLine.length() < SERIAL_LINE_LIMIT)
    {
      serialLine += c;
    }
    else
    {
      serialLine = "";
      Serial.println("[SERIAL] Input line too long, dropped");
    }
  }
}

String buildAlertSpeech(uint8_t cameraNumber)
{
  (void)cameraNumber;
  return String(u8"Phát hiện có người ngã");
  String speech = "CẢNH BÁO PHÁT HIỆN CÓ NGƯỜI NGÃ TẠI CAMERA SỐ ";
  speech += cameraNumber;
  return speech;
}

String buildFaintAlertSpeech(uint8_t cameraNumber)
{
  (void)cameraNumber;
  return String(u8"Phát hiện có người ngất");
}

String buildEventSpeech(const AudioEvent &event)
{
  if (event.type == AudioEventType::FallAlert)
  {
    return buildAlertSpeech(event.cameraNumber);
  }
  if (event.type == AudioEventType::FaintAlert)
  {
    return buildFaintAlertSpeech(event.cameraNumber);
  }

  return String(event.speech);
}

bool startSpeech(const String &speech)
{
  speechEnded = false;
  Serial.printf("[AUDIO] Speaking: %s\n", speech.c_str());
  return audio.connecttospeech(speech.c_str(), "vi");
}

void audioTask(void *parameter)
{
  AudioEvent currentEvent{};
  bool hasCurrentEvent = false;
  bool isPlaying = false;
  uint8_t remainingRepeats = 0;
  uint32_t nextStartMs = 0;

  for (;;)
  {
    audio.loop();

    if (speechEnded)
    {
      speechEnded = false;
      isPlaying = false;
      nextStartMs = millis() + TTS_REPEAT_GAP_MS;
    }

    if (!hasCurrentEvent)
    {
      if (xQueueReceive(audioQueue, &currentEvent, 0) == pdTRUE)
      {
        hasCurrentEvent = true;
        isPlaying = false;
        remainingRepeats = currentEvent.repeatCount;
        nextStartMs = millis();
      }
    }

    if (hasCurrentEvent && !isPlaying && remainingRepeats == 0)
    {
      audio.stopSong();
      hasCurrentEvent = false;
      Serial.println("[AUDIO] Completed speech event");
    }

    if (hasCurrentEvent && !isPlaying && remainingRepeats > 0 &&
        static_cast<int32_t>(millis() - nextStartMs) >= 0)
    {
      if (WiFi.status() != WL_CONNECTED)
      {
        Serial.println("[AUDIO] WiFi is not connected; cannot start online TTS");
        remainingRepeats = 0;
      }
      else if (startSpeech(buildEventSpeech(currentEvent)))
      {
        --remainingRepeats;
        isPlaying = true;
      }
      else
      {
        Serial.println("[AUDIO] Failed to start TTS stream");
        remainingRepeats = 0;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void processStatusJson(JsonVariantConst root, AlertStatus *cameraAlertStatus, size_t activeCount)
{
  if (!root["cameras"].is<JsonArrayConst>())
  {
    const AlertStatus status = jsonNodeAlertStatus(root);
    const uint8_t cameraNumber = jsonCameraNumber(root);
    const size_t index = cameraIndex(cameraNumber, activeCount);

    if (status != AlertStatus::None && status != cameraAlertStatus[index])
    {
      queueAlert(status, cameraNumber);
    }
    cameraAlertStatus[index] = status;
    return;
  }

  bool seen[MAX_TRACKED_CAMERAS] = {};
  JsonArrayConst cameras = root["cameras"].as<JsonArrayConst>();
  for (JsonObjectConst camera : cameras)
  {
    const uint8_t cameraNumber = jsonCameraNumber(camera);
    const size_t index = cameraIndex(cameraNumber, activeCount);
    const AlertStatus status = jsonNodeAlertStatus(camera);
    seen[index] = true;

    if (status != AlertStatus::None && status != cameraAlertStatus[index])
    {
      queueAlert(status, cameraNumber);
    }
    cameraAlertStatus[index] = status;
  }

  for (size_t i = 0; i < activeCount; ++i)
  {
    if (!seen[i])
    {
      cameraAlertStatus[i] = AlertStatus::None;
    }
  }
}

void statusPollTask(void *parameter)
{
  AlertStatus cameraAlertStatus[MAX_TRACKED_CAMERAS] = {};
  bool webWasConnected = false;
  bool webConnectionAnnounced = false;
  TickType_t lastWake = xTaskGetTickCount();

  for (;;)
  {
    if (WiFi.status() == WL_CONNECTED && statusUrlConfigured())
    {
      HTTPClient http;
      http.setTimeout(HTTP_TIMEOUT_MS);

      if (http.begin(FALL_STATUS_URL))
      {
        const int statusCode = http.GET();
        const bool webConnected = statusCode >= 200 && statusCode < 400;
        if (webConnected && !webWasConnected && !webConnectionAnnounced)
        {
          queueSpeech("ĐÃ KẾT NỐI ĐƯỢC VỚI WEB");
          webConnectionAnnounced = true;
        }
        webWasConnected = webConnected;

        if (statusCode == HTTP_CODE_OK)
        {
          JsonDocument doc;
          const DeserializationError error = deserializeJson(doc, http.getString());
          if (!error)
          {
            processStatusJson(doc.as<JsonVariantConst>(), cameraAlertStatus, MAX_TRACKED_CAMERAS);
          }
          else
          {
            Serial.printf("[HTTP] JSON parse failed: %s\n", error.c_str());
          }
        }
        else if (statusCode > 0)
        {
          Serial.printf("[HTTP] /status returned %d\n", statusCode);
        }
        else
        {
          Serial.printf("[HTTP] GET failed: %s\n", http.errorToString(statusCode).c_str());
        }
      }

      http.end();
    }
    else
    {
      webWasConnected = false;
    }

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(STATUS_POLL_INTERVAL_MS));
  }
}

void beginWiFi()
{
  if (!wifiConfigured())
  {
    Serial.println("[WIFI] WIFI_SSID is empty; Serial events still work but TTS needs WiFi");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWifiAttemptMs = millis();
  Serial.printf("[WIFI] Connecting to %s\n", WIFI_SSID);
}

void serviceWiFi()
{
  if (!wifiConfigured())
  {
    return;
  }

  if (WiFi.status() == WL_CONNECTED)
  {
    if (!wifiWasConnected)
    {
      wifiWasConnected = true;
      Serial.print("[WIFI] Connected, IP: ");
      Serial.println(WiFi.localIP());
      queueSpeech("ĐÃ KẾT NỐI ĐƯỢC VỚI WIFI");
    }
    return;
  }

  wifiWasConnected = false;

  if (millis() - lastWifiAttemptMs >= WIFI_RETRY_INTERVAL_MS)
  {
    lastWifiAttemptMs = millis();
    Serial.println("[WIFI] Retrying connection");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}

void audio_info(const char *info)
{
  Serial.printf("[AUDIO] %s\n", info);
}

void audio_eof_speech(const char *info)
{
  speechEnded = true;
}

void setup()
{
  Serial.begin(SERIAL_BAUD);
  Serial.println();
  Serial.println("[BOOT] ESP32 fall alert audio module starting");

  audioQueue = xQueueCreate(6, sizeof(AudioEvent));
  if (audioQueue == nullptr)
  {
    Serial.println("[BOOT] Failed to create audio queue");
    return;
  }

  audio.setPinout(I2S_BCLK_PIN, I2S_LRC_PIN, I2S_DIN_PIN);
  audio.setVolume(AUDIO_VOLUME);

  beginWiFi();

  xTaskCreatePinnedToCore(audioTask, "fall_audio", 8192, nullptr, 2, nullptr, 0);

  if (statusUrlConfigured())
  {
    xTaskCreatePinnedToCore(statusPollTask, "fall_status", 8192, nullptr, 1, nullptr, 1);
    Serial.printf("[HTTP] Polling fall status from %s\n", FALL_STATUS_URL);
  }
  else
  {
    Serial.println("[HTTP] FALL_STATUS_URL is empty; waiting for Serial fall events");
  }

  Serial.println("[SERIAL] Examples:");
  Serial.println("[SERIAL] {\"status\":\"FALLEN\",\"camera_id\":0}");
  Serial.println("[SERIAL] CAMERA 2 NGA");
}

void loop()
{
  serviceSerialInput();
  serviceWiFi();
  yield();
}
