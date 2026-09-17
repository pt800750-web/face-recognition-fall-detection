# Camera AI Fall Detection System

Hệ thống phát hiện té ngã từ camera/video, kết hợp AI thị giác máy tính, web giám sát, cảnh báo Telegram, xác thực khuôn mặt và phần cứng ESP32 phát cảnh báo bằng giọng nói.

Mục tiêu của đồ án không chỉ là nhận diện té ngã trên video, mà là xây dựng một hệ thống cảnh báo hoàn chỉnh có thể chạy demo theo luồng thực tế: camera phát hiện sự cố, web cập nhật trạng thái, Telegram gửi ảnh cảnh báo, ESP32 phát âm thanh và người giám sát xác nhận an toàn.

## Thành Phần Chính

1. **YOLOv8 Pose pretrained**
   - Dùng `yolov8n-pose.pt` để phát hiện người và trích xuất keypoint tư thế.
   - Đây là model nền có sẵn, không phải model té ngã do đồ án tự train.

2. **Mô hình GRU tự huấn luyện**
   - File train chính: `train.py`.
   - Kiến trúc model: `src/models.py`, lớp `TemporalGRUFallClassifier`.
   - Checkpoint sau huấn luyện:
     - `checkpoints/best_model.pt`
     - `checkpoints/last_model.pt`
   - Model nhận chuỗi feature theo thời gian để phân loại trạng thái như đi/đứng, ngồi, nằm chủ động, đang ngã và đã ngã.

3. **State machine giảm báo nhầm**
   - File chính: `src/fall_logic.py`.
   - Xác nhận té ngã bằng bằng chứng theo thời gian: chuyển động rơi nhanh, thay đổi chiều cao cơ thể, tư thế nằm sau ngã và trạng thái bất động.
   - Giảm báo nhầm với các tình huống như ngồi, cúi, nằm chủ động, tiến sát camera hoặc mất track ngắn.

4. **Web dashboard**
   - File chính: `web_server.py`.
   - Theo dõi nhiều camera, xem trạng thái từng người, lịch sử cảnh báo, ảnh cảnh báo và video output.
   - Có đăng nhập và bước xác thực khuôn mặt trước khi vào dashboard.

5. **Quét mặt và quản lý hồ sơ**
   - File chính: `../quetmat.py`.
   - Lưu hồ sơ người già/người thân và encoding khuôn mặt vào `../data/database.json`.
   - Web server dùng dữ liệu này để xác thực người được cấp quyền.

6. **Cảnh báo Telegram**
   - File chính: `src/alert_manager.py`.
   - Gửi tin nhắn kèm ảnh snapshot khi phát hiện trạng thái nguy hiểm.
   - Có link xác nhận an toàn; nếu chưa xác nhận, bot có thể nhắc lại theo cấu hình.

7. **Phần cứng ESP32**
   - File chính: `../PhanCung/Esp32/src/main.cpp`.
   - ESP32 kết nối WiFi, đọc trạng thái từ endpoint `/status`, nhận JSON cảnh báo và phát âm thanh qua module I2S/MAX98357A.
   - Có thể nhận cảnh báo qua Serial để test nhanh.

## Luồng Xử Lý

```text
Camera/video
  -> YOLOv8 Pose phát hiện người và keypoint
  -> Tracker gán ID từng người
  -> Trích xuất feature hình học, pose và chuyển động
  -> GRU phân loại trạng thái theo chuỗi thời gian
  -> State machine xác nhận té ngã và giảm báo nhầm
  -> Web dashboard cập nhật trạng thái
  -> Telegram gửi cảnh báo
  -> ESP32 phát cảnh báo giọng nói
```

## Cài Đặt

```bash
cd phan_mem
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng xác thực khuôn mặt, cần cài thêm `face_recognition` theo môi trường Windows phù hợp.

## Chuẩn Bị Dữ Liệu

```bash
python download_datasets.py --config configs/config.yaml
python prepare_dataset.py --config configs/config.yaml
```

Các manifest được tạo trong:

```text
data/manifests/
```

File thống kê dataset:

```text
data/manifests/dataset_stats.json
```

## Huấn Luyện Model Té Ngã

```bash
python train.py --config configs/config.yaml
```

Kết quả train được lưu ở:

```text
checkpoints/best_model.pt
checkpoints/last_model.pt
logs/train_metrics.json
```

Khi bảo vệ, nên nói rõ:

> Hệ thống sử dụng YOLOv8 Pose pretrained để trích xuất tư thế người, sau đó tự xây dựng và huấn luyện mô hình GRU theo chuỗi thời gian để nhận diện trạng thái té ngã. Ngoài model học máy, hệ thống có state machine để giảm báo nhầm trong các tình huống sinh hoạt thường ngày.

## Chạy Inference

Chạy với webcam:

```bash
python inference.py --config configs/config.yaml --source 0
```

Chạy với video:

```bash
python inference.py --config configs/config.yaml --source path\to\video.mp4
```

## Chạy Web Dashboard

```bash
python web_server.py --config configs/config.yaml
```

Sau đó mở trình duyệt vào địa chỉ server hiển thị trong terminal.

Luồng demo nên chuẩn bị:

1. Đăng nhập web.
2. Xác thực khuôn mặt.
3. Mở camera giám sát.
4. Thực hiện tình huống bình thường.
5. Thực hiện tình huống té ngã.
6. Kiểm tra dashboard đổi trạng thái.
7. Kiểm tra Telegram nhận ảnh cảnh báo.
8. Kiểm tra ESP32 phát âm thanh cảnh báo.
9. Bấm xác nhận an toàn để dừng nhắc lại.

## Test Telegram

```bash
python test_alert_telegram.py
```

Script này tạo cảnh báo thử và gửi qua Telegram nếu `.env` đã cấu hình đúng.

## Cấu Hình Quan Trọng

Các biến cấu hình nằm trong `.env`, ví dụ:

```text
PUBLIC_BASE_URL=
ALERT_ENABLE_TELEGRAM=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_IDS=
AUTH_USERNAME=
AUTH_PASSWORD=
FACE_AUTH_DATABASE=
```

Khi nộp hoặc public source code, không để lộ token Telegram, mật khẩu WiFi, mật khẩu đăng nhập và secret xác nhận cảnh báo.

## Đánh Giá Hiện Tại

Project đã có pipeline train/validation và log metric trong `logs/train_metrics.json`. Tuy nhiên tập test trong manifest hiện còn nhỏ, vì vậy khi làm báo cáo nên bổ sung thêm bảng kiểm thử thực nghiệm bằng camera thật.

Gợi ý bảng kiểm thử demo:

| Tình huống | Số lần test | Đúng | Sai | Ghi chú |
|---|---:|---:|---:|---|
| Đi/đứng bình thường | 10 |  |  | Không cảnh báo |
| Ngồi xuống | 10 |  |  | Không cảnh báo té |
| Cúi nhặt đồ | 10 |  |  | Không cảnh báo té |
| Nằm chủ động | 10 |  |  | Không cảnh báo té |
| Té ngã | 10 |  |  | Có cảnh báo |
| Nằm bất động sau ngã | 5 |  |  | Có cảnh báo ngất/bất tỉnh |

## Điểm Mạnh Khi Bảo Vệ

- Có model GRU tự train, không chỉ dùng model có sẵn.
- Có xử lý chuỗi thời gian thay vì chỉ nhìn từng frame.
- Có tracker để theo dõi từng người.
- Có logic giảm báo nhầm với nằm/ngồi/cúi.
- Có web dashboard, Telegram, xác nhận cảnh báo và nhắc lại.
- Có quét mặt để kiểm soát người truy cập.
- Có ESP32 phát cảnh báo âm thanh, tạo thành hệ thống AI + IoT hoàn chỉnh.

## Hướng Phát Triển

- Mở rộng tập test độc lập với nhiều góc camera và nhiều người.
- Bổ sung dữ liệu cho các lớp còn thiếu như cúi người và bất tỉnh.
- Thêm chống giả mạo khuôn mặt bằng liveness detection.
- Đưa cấu hình nhạy cảm ra biến môi trường hoặc file không nộp kèm source.
- Đóng gói web server thành service để chạy ổn định hơn khi triển khai.
