import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

try:
    import face_recognition
except ImportError:
    face_recognition = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
DATABASE_FILE = DATA_DIR / "database.json"

ADD_TAB_NAME = "Đăng ký nhân viên"
SEARCH_TAB_NAME = "Kiểm tra danh tính"

CAMERA_INDEX = 0
CAMERA_FALLBACK_INDICES = (0, 1, 2)
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
ADD_CAMERA_SIZE = (560, 360)
SEARCH_CAMERA_SIZE = (430, 280)
RESULT_FACE_SIZE = (260, 170)

SCAN_REQUIRED_FRAMES = 10
SCAN_PROCESS_EVERY_N_FRAMES = 3
FACE_MATCH_TOLERANCE = 0.5
CAMERA_RECONNECT_FAILS = 20
CAMERA_RECONNECT_BLACK_FRAMES = 8
BLACK_FRAME_MEAN_THRESHOLD = 3.0
BLACK_FRAME_STD_THRESHOLD = 3.0


def ensure_data_storage():
    DATA_DIR.mkdir(exist_ok=True)
    FACES_DIR.mkdir(exist_ok=True)
    if not DATABASE_FILE.exists():
        DATABASE_FILE.write_text("[]", encoding="utf-8")


def load_database():
    ensure_data_storage()
    try:
        data = json.loads(DATABASE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_database(records):
    ensure_data_storage()
    DATABASE_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def staff_records(records):
    return [record for record in records if isinstance(record, dict) and isinstance(record.get("staff_info"), dict)]


def sanitize_file_id(value):
    cleaned = str(value or "").strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    return cleaned or "unknown"


def path_to_json_value(path):
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def frame_to_ctk_image(frame_bgr, target_size):
    if frame_bgr is None:
        return None

    target_width, target_height = target_size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", target_size, (15, 17, 20))
    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2
    canvas.paste(image, (x, y))

    return ctk.CTkImage(light_image=canvas, dark_image=canvas, size=target_size)


def image_file_to_ctk_preview(image_path, target_size):
    if not image_path or not image_path.exists():
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return None

    target_width, target_height = target_size
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (15, 17, 20))
    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2
    canvas.paste(image, (x, y))
    return ctk.CTkImage(light_image=canvas, dark_image=canvas, size=target_size)


def draw_face_box(frame, location, label, color=(34, 197, 94)):
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, max(top - 30, 0)), (right, top), color, cv2.FILLED)
    cv2.putText(
        frame,
        label,
        (left + 6, max(top - 9, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_scan_effect(frame, progress):
    height, width = frame.shape[:2]
    y = int(height * (0.18 + 0.64 * min(max(progress, 0), 1)))
    cv2.line(frame, (40, y), (width - 40, y), (56, 189, 248), 3)
    cv2.rectangle(frame, (35, int(height * 0.18)), (width - 35, int(height * 0.82)), (56, 189, 248), 1)


class StaffFaceDatabaseApp(ctk.CTk):
    """Ứng dụng tạo database khuôn mặt nhân viên giám sát để đăng nhập hệ thống."""

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        ensure_data_storage()

        self.title("Database nhân viên giám sát")
        self.geometry("1180x760")
        self.minsize(1080, 680)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.database = load_database()
        self.known_faces = []
        self.refresh_known_faces()

        self.cap = None
        self.camera_backend_name = ""
        self.camera_index_in_use = CAMERA_INDEX
        self.current_frame = None
        self.last_good_frame = None
        self.camera_after_id = None
        self.camera_tick = 0
        self.camera_fail_count = 0
        self.camera_black_count = 0
        self.active_tab_name = ADD_TAB_NAME

        self.active_scan = None
        self.scan_encodings = []
        self.scan_success_count = 0
        self.scan_last_frame = None
        self.scan_last_location = None
        self.staff_face_data = None

        self.add_camera_image = None
        self.search_camera_image = None
        self.result_face_image = None

        self.entries = {}
        self.result_labels = {}

        self.create_layout()

        if face_recognition is None:
            self.show_dialog(
                "Thiếu thư viện",
                "Không tìm thấy face_recognition. Hãy cài đặt đủ thư viện trước khi quét mặt.",
                "error",
            )

        self.open_camera()
        self.update_camera_loop()

    def create_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Database khuôn mặt nhân viên giám sát",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="Dữ liệu này dùng cho bước quét mặt khi nhân viên đăng nhập hệ thống giám sát té ngã.",
            text_color=("#475569", "#94a3b8"),
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        theme_frame = ctk.CTkFrame(header, fg_color="transparent")
        theme_frame.grid(row=0, column=1, rowspan=2, sticky="e")
        ctk.CTkLabel(theme_frame, text="Theme").grid(row=0, column=0, padx=(0, 8))
        self.theme_switch = ctk.CTkSegmentedButton(theme_frame, values=["Light", "Dark", "System"], command=ctk.set_appearance_mode)
        self.theme_switch.set("System")
        self.theme_switch.grid(row=0, column=1)

        self.tabview = ctk.CTkTabview(self, command=self.on_tab_change, corner_radius=14)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=18, pady=(6, 18))
        self.add_tab = self.tabview.add(ADD_TAB_NAME)
        self.search_tab = self.tabview.add(SEARCH_TAB_NAME)

        self.create_add_tab()
        self.create_search_tab()

    def create_add_tab(self):
        self.add_tab.grid_columnconfigure(0, weight=1, minsize=500)
        self.add_tab.grid_columnconfigure(1, weight=1, minsize=560)
        self.add_tab.grid_rowconfigure(0, weight=1)

        form_panel = ctk.CTkScrollableFrame(self.add_tab, corner_radius=16)
        form_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 8), pady=10)
        form_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(form_panel, text="Thông tin nhân viên", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 4)
        )

        info_section = ctk.CTkFrame(form_panel, corner_radius=14)
        info_section.grid(row=1, column=0, sticky="ew", padx=12, pady=10)
        info_section.grid_columnconfigure((0, 1), weight=1)

        self.create_entry(info_section, "staff_id", "Mã nhân viên", 0, 0)
        self.create_entry(info_section, "staff_name", "Họ tên nhân viên", 0, 1)
        self.create_entry(info_section, "role", "Chức vụ", 1, 0)
        self.create_entry(info_section, "department", "Bộ phận/khu vực phụ trách", 1, 1)
        self.create_entry(info_section, "phone", "Số điện thoại", 2, 0)
        self.create_entry(info_section, "email", "Email", 2, 1)

        self.active_checkbox = ctk.CTkCheckBox(info_section, text="Cho phép nhân viên này đăng nhập bằng khuôn mặt")
        self.active_checkbox.select()
        self.active_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 14))

        self.save_button = ctk.CTkButton(
            form_panel,
            text="Lưu nhân viên vào database",
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.save_record,
        )
        self.save_button.grid(row=2, column=0, sticky="ew", padx=12, pady=(12, 18))

        camera_panel = ctk.CTkFrame(self.add_tab, corner_radius=16)
        camera_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 10), pady=10)
        camera_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(camera_panel, text="Quét mặt nhân viên", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(16, 8)
        )
        self.add_camera_label = ctk.CTkLabel(
            camera_panel,
            text="Đang mở camera...",
            width=ADD_CAMERA_SIZE[0],
            height=ADD_CAMERA_SIZE[1],
            corner_radius=12,
            fg_color=("#111827", "#020617"),
        )
        self.add_camera_label.grid(row=1, column=0, sticky="n", padx=16, pady=8)

        self.add_progress = ctk.CTkProgressBar(camera_panel, height=14)
        self.add_progress.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 2))
        self.add_progress.set(0)

        self.add_scan_status = ctk.CTkLabel(
            camera_panel,
            text="Chưa quét khuôn mặt nhân viên.",
            wraplength=520,
            justify="left",
            text_color=("#334155", "#cbd5e1"),
        )
        self.add_scan_status.grid(row=3, column=0, sticky="w", padx=16, pady=(4, 8))

        self.scan_staff_button = ctk.CTkButton(
            camera_panel,
            text="Quét mặt nhân viên",
            height=42,
            command=lambda: self.start_face_scan("add"),
        )
        self.scan_staff_button.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 10))

        self.staff_scan_badge = ctk.CTkLabel(
            camera_panel,
            text="Nhân viên: chưa quét",
            height=34,
            corner_radius=10,
            fg_color=("#e2e8f0", "#1e293b"),
        )
        self.staff_scan_badge.grid(row=5, column=0, sticky="ew", padx=16, pady=(4, 16))

    def create_search_tab(self):
        self.search_tab.grid_columnconfigure(0, weight=0, minsize=470)
        self.search_tab.grid_columnconfigure(1, weight=1, minsize=560)
        self.search_tab.grid_rowconfigure(0, weight=1)

        search_panel = ctk.CTkFrame(self.search_tab, corner_radius=16)
        search_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 8), pady=10)
        search_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(search_panel, text="Tìm kiếm nhân viên", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(16, 8)
        )
        self.search_entry = ctk.CTkEntry(search_panel, height=42, placeholder_text="Nhập tên, mã nhân viên, chức vụ hoặc bộ phận")
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 8))
        self.search_entry.bind("<Return>", lambda _event: self.search_by_text())

        ctk.CTkButton(search_panel, text="Tìm theo chữ", height=40, command=self.search_by_text).grid(
            row=2, column=0, sticky="ew", padx=16, pady=(0, 14)
        )

        ctk.CTkLabel(search_panel, text="Quét mặt để kiểm tra danh tính", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=3, column=0, sticky="w", padx=16, pady=(4, 8)
        )
        self.search_camera_label = ctk.CTkLabel(
            search_panel,
            text="Camera kiểm tra sẽ chạy khi tab này được mở.",
            width=SEARCH_CAMERA_SIZE[0],
            height=SEARCH_CAMERA_SIZE[1],
            corner_radius=12,
            fg_color=("#111827", "#020617"),
        )
        self.search_camera_label.grid(row=4, column=0, sticky="n", padx=16, pady=8)

        self.search_progress = ctk.CTkProgressBar(search_panel, height=14)
        self.search_progress.grid(row=5, column=0, sticky="ew", padx=16, pady=(8, 2))
        self.search_progress.set(0)

        self.search_status = ctk.CTkLabel(
            search_panel,
            text="Sẵn sàng kiểm tra danh tính nhân viên.",
            wraplength=420,
            justify="left",
            text_color=("#334155", "#cbd5e1"),
        )
        self.search_status.grid(row=6, column=0, sticky="w", padx=16, pady=(4, 10))

        self.search_scan_button = ctk.CTkButton(
            search_panel,
            text="Quét để kiểm tra",
            height=42,
            command=lambda: self.start_face_scan("search"),
        )
        self.search_scan_button.grid(row=7, column=0, sticky="ew", padx=16, pady=(4, 16))

        result_panel = ctk.CTkFrame(self.search_tab, corner_radius=16)
        result_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 10), pady=10)
        result_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(result_panel, text="Kết quả nhân viên", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(16, 8)
        )
        self.result_face_label = ctk.CTkLabel(
            result_panel,
            text="Chưa có ảnh",
            width=RESULT_FACE_SIZE[0],
            height=RESULT_FACE_SIZE[1],
            corner_radius=12,
            fg_color=("#e2e8f0", "#020617"),
            text_color=("#64748b", "#94a3b8"),
        )
        self.result_face_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(6, 12))

        fields = [
            ("staff_id", "Mã nhân viên"),
            ("staff_name", "Họ tên"),
            ("role", "Chức vụ"),
            ("department", "Bộ phận/khu vực"),
            ("phone", "Số điện thoại"),
            ("email", "Email"),
            ("active", "Trạng thái đăng nhập"),
            ("updated_at", "Cập nhật lần cuối"),
        ]
        for row, (key, label) in enumerate(fields, start=2):
            row_frame = ctk.CTkFrame(result_panel, fg_color="transparent")
            row_frame.grid(row=row, column=0, sticky="ew", padx=16, pady=6)
            row_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row_frame, text=label, font=ctk.CTkFont(size=12, weight="bold"), text_color=("#64748b", "#94a3b8")).grid(
                row=0, column=0, sticky="w"
            )
            value_label = ctk.CTkLabel(row_frame, text="", justify="left", anchor="w", wraplength=430, font=ctk.CTkFont(size=14))
            value_label.grid(row=1, column=0, sticky="ew", pady=(2, 0))
            self.result_labels[key] = value_label

        self.clear_result()

    def create_entry(self, parent, key, placeholder, row, column):
        wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        wrapper.grid(row=row, column=column, sticky="ew", padx=12, pady=8)
        wrapper.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            wrapper,
            text=placeholder,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#475569", "#94a3b8"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        entry = ctk.CTkEntry(wrapper, height=38, placeholder_text=placeholder, corner_radius=10)
        entry.grid(row=1, column=0, sticky="ew")
        self.entries[key] = entry
        return entry

    def show_dialog(self, title, message, kind="info"):
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("440x220")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        color = {
            "info": ("#2563eb", "#60a5fa"),
            "success": ("#16a34a", "#4ade80"),
            "warning": ("#ca8a04", "#facc15"),
            "error": ("#dc2626", "#f87171"),
        }.get(kind, ("#2563eb", "#60a5fa"))

        frame = ctk.CTkFrame(dialog, corner_radius=16)
        frame.pack(fill="both", expand=True, padx=14, pady=14)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=color).pack(
            anchor="w", padx=16, pady=(16, 6)
        )
        ctk.CTkLabel(frame, text=message, wraplength=370, justify="left").pack(anchor="w", padx=16, pady=6)
        ctk.CTkButton(frame, text="Đóng", command=dialog.destroy).pack(anchor="e", padx=16, pady=(12, 16))
        self.center_window(dialog)
        self.wait_window(dialog)

    def ask_confirm(self, title, message):
        result = {"ok": False}
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("460x230")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        frame = ctk.CTkFrame(dialog, corner_radius=16)
        frame.pack(fill="both", expand=True, padx=14, pady=14)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=16, pady=(16, 6))
        ctk.CTkLabel(frame, text=message, wraplength=390, justify="left").pack(anchor="w", padx=16, pady=6)

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.pack(fill="x", padx=16, pady=(16, 16))

        def accept():
            result["ok"] = True
            dialog.destroy()

        ctk.CTkButton(buttons, text="Hủy", fg_color="gray", command=dialog.destroy).pack(side="right", padx=(8, 0))
        ctk.CTkButton(buttons, text="Ghi đè", command=accept).pack(side="right")
        self.center_window(dialog)
        self.wait_window(dialog)
        return result["ok"]

    def center_window(self, window):
        window.update_idletasks()
        main_x = self.winfo_x()
        main_y = self.winfo_y()
        main_w = self.winfo_width()
        main_h = self.winfo_height()
        win_w = window.winfo_width()
        win_h = window.winfo_height()
        x = main_x + (main_w - win_w) // 2
        y = main_y + (main_h - win_h) // 2
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def open_camera(self):
        if self.cap is not None and self.cap.isOpened():
            return True

        for camera_index in self.get_camera_indices():
            for backend_name, backend in self.get_camera_backends():
                cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                self.camera_index_in_use = camera_index
                self.camera_backend_name = backend_name
                self.camera_fail_count = 0
                self.camera_black_count = 0
                return True

        self.cap = None
        self.camera_backend_name = ""
        self.set_active_camera_message("Không mở được camera. Hãy kiểm tra quyền camera hoặc đổi CAMERA_INDEX.")
        return False

    def get_camera_indices(self):
        indices = [CAMERA_INDEX]
        for index in CAMERA_FALLBACK_INDICES:
            if index not in indices:
                indices.append(index)
        return indices

    def get_camera_backends(self):
        if os.name != "nt":
            return [("Tự động", None)]
        backends = []
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("Media Foundation", cv2.CAP_MSMF))
        backends.append(("Tự động", None))
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DirectShow", cv2.CAP_DSHOW))
        return backends

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_frame = None

    def reconnect_camera(self):
        self.release_camera()
        self.camera_fail_count = 0
        self.camera_black_count = 0
        self.set_active_camera_message("Đang kết nối lại camera...")
        self.after(250, self.open_camera)

    def update_camera_loop(self):
        self.active_tab_name = self.tabview.get()

        if self.cap is None or not self.cap.isOpened():
            self.open_camera()
            self.camera_after_id = self.after(250, self.update_camera_loop)
            return

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.camera_fail_count = 0
            if self.is_black_camera_frame(frame):
                self.camera_black_count += 1
                if self.camera_black_count >= CAMERA_RECONNECT_BLACK_FRAMES:
                    self.reconnect_camera()
                else:
                    self.render_cached_frame_or_message("Camera đang khởi động lại...")
                self.camera_after_id = self.after(60, self.update_camera_loop)
                return

            self.camera_black_count = 0
            self.camera_tick += 1
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            self.last_good_frame = frame.copy()
            display_frame = frame.copy()

            if self.active_scan and self.is_scan_on_active_tab():
                display_frame = self.process_active_scan(display_frame)

            if self.active_tab_name == ADD_TAB_NAME:
                self.update_camera_label(self.add_camera_label, display_frame, ADD_CAMERA_SIZE, "add")
            elif self.active_tab_name == SEARCH_TAB_NAME:
                self.update_camera_label(self.search_camera_label, display_frame, SEARCH_CAMERA_SIZE, "search")
        else:
            self.camera_fail_count += 1
            if self.camera_fail_count >= CAMERA_RECONNECT_FAILS:
                self.reconnect_camera()
            else:
                self.render_cached_frame_or_message("Camera đang khởi động...")

        self.camera_after_id = self.after(30, self.update_camera_loop)

    def is_black_camera_frame(self, frame):
        return float(np.mean(frame)) < BLACK_FRAME_MEAN_THRESHOLD and float(np.std(frame)) < BLACK_FRAME_STD_THRESHOLD

    def render_cached_frame_or_message(self, message):
        if self.last_good_frame is not None:
            if self.active_tab_name == ADD_TAB_NAME:
                self.update_camera_label(self.add_camera_label, self.last_good_frame, ADD_CAMERA_SIZE, "add")
            elif self.active_tab_name == SEARCH_TAB_NAME:
                self.update_camera_label(self.search_camera_label, self.last_good_frame, SEARCH_CAMERA_SIZE, "search")
        else:
            self.set_active_camera_message(message)

    def update_camera_label(self, label, frame, size, target):
        image = frame_to_ctk_image(frame, size)
        if image is None:
            return
        label.configure(image=image, text="")
        label.image = image
        if target == "add":
            self.add_camera_image = image
        else:
            self.search_camera_image = image

    def set_active_camera_message(self, message):
        if self.active_tab_name == ADD_TAB_NAME:
            self.add_camera_label.configure(image=None, text=message)
            self.add_camera_label.image = None
            self.add_camera_image = None
        else:
            self.search_camera_label.configure(image=None, text=message)
            self.search_camera_label.image = None
            self.search_camera_image = None

    def on_tab_change(self):
        self.cancel_active_scan(reset_message=True)
        self.active_tab_name = self.tabview.get()
        self.camera_fail_count = 0
        self.camera_black_count = 0
        self.render_cached_frame_or_message("Camera đang khởi động...")
        if self.cap is None or not self.cap.isOpened():
            self.after(150, self.open_camera)

    def start_face_scan(self, context):
        if face_recognition is None:
            self.show_dialog("Thiếu thư viện", "Không thể quét mặt vì chưa cài face_recognition.", "error")
            return

        expected_tab = ADD_TAB_NAME if context == "add" else SEARCH_TAB_NAME
        if self.tabview.get() != expected_tab:
            self.show_dialog("Sai tab", "Hãy mở đúng tab trước khi quét mặt.", "warning")
            return

        if not self.open_camera():
            self.show_dialog("Lỗi camera", "Không mở được webcam.", "error")
            return

        if context == "search" and not self.known_faces:
            self.show_dialog("Chưa có dữ liệu", "Database chưa có khuôn mặt nhân viên nào để so khớp.", "warning")
            return

        self.active_scan = {"context": context}
        self.scan_encodings = []
        self.scan_success_count = 0
        self.scan_last_frame = None
        self.scan_last_location = None

        if context == "add":
            self.add_progress.set(0)
            self.add_scan_status.configure(text=f"Đang quét mặt nhân viên: cần đủ {SCAN_REQUIRED_FRAMES} frame có một khuôn mặt rõ.")
            self.scan_staff_button.configure(state="disabled")
        else:
            self.search_progress.set(0)
            self.search_status.configure(text="Đang quét khuôn mặt để kiểm tra danh tính...")
            self.search_scan_button.configure(state="disabled")

    def is_scan_on_active_tab(self):
        if not self.active_scan:
            return False
        if self.active_scan["context"] == "add":
            return self.active_tab_name == ADD_TAB_NAME
        return self.active_tab_name == SEARCH_TAB_NAME

    def process_active_scan(self, display_frame):
        progress = self.scan_success_count / SCAN_REQUIRED_FRAMES
        draw_scan_effect(display_frame, progress)
        if self.camera_tick % SCAN_PROCESS_EVERY_N_FRAMES != 0:
            return display_frame

        analysis_frame = self.current_frame if self.current_frame is not None else display_frame
        try:
            encoding, location = self.detect_single_face(analysis_frame)
        except ValueError as error:
            self.update_scan_status(str(error))
            return display_frame
        except Exception as error:
            self.update_scan_status(f"Lỗi nhận diện: {error}")
            return display_frame

        self.scan_encodings.append(encoding)
        self.scan_success_count += 1
        self.scan_last_frame = self.current_frame.copy()
        self.scan_last_location = location

        draw_face_box(display_frame, location, f"Đang quét {self.scan_success_count}/{SCAN_REQUIRED_FRAMES}")
        self.update_scan_status(f"Đã nhận {self.scan_success_count}/{SCAN_REQUIRED_FRAMES} frame hợp lệ.")
        if self.active_scan["context"] == "add":
            self.add_progress.set(min(self.scan_success_count / SCAN_REQUIRED_FRAMES, 1))
        else:
            self.search_progress.set(min(self.scan_success_count / SCAN_REQUIRED_FRAMES, 1))

        if self.scan_success_count >= SCAN_REQUIRED_FRAMES:
            self.finish_active_scan()
        return display_frame

    def detect_single_face(self, frame_bgr):
        if face_recognition is None:
            raise RuntimeError("Chưa cài thư viện face_recognition.")

        scale = 0.5
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small = np.ascontiguousarray(rgb_small)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        if not locations:
            raise ValueError("Chưa thấy khuôn mặt rõ trong khung hình.")
        if len(locations) > 1:
            raise ValueError("Đang thấy nhiều khuôn mặt. Vui lòng chỉ để một nhân viên trong khung hình.")

        encodings = face_recognition.face_encodings(rgb_small, locations)
        if not encodings:
            raise ValueError("Không trích xuất được mã hóa khuôn mặt.")

        top, right, bottom, left = locations[0]
        original_location = (int(top / scale), int(right / scale), int(bottom / scale), int(left / scale))
        return encodings[0], original_location

    def finish_active_scan(self):
        if not self.active_scan:
            return

        context = self.active_scan["context"]
        averaged_encoding = np.mean(np.array(self.scan_encodings), axis=0)

        if context == "add":
            self.staff_face_data = {"encoding": averaged_encoding.tolist(), "frame": self.scan_last_frame.copy()}
            self.staff_scan_badge.configure(
                text="Nhân viên: đã quét",
                fg_color=("#bbf7d0", "#14532d"),
                text_color=("#166534", "#dcfce7"),
            )
            self.add_scan_status.configure(text="Quét mặt nhân viên thành công. Bạn có thể lưu hồ sơ.")
            self.add_progress.set(1)
            self.scan_staff_button.configure(state="normal")
            self.active_scan = None
            return

        self.search_progress.set(1)
        match = self.find_best_face_match(averaged_encoding)
        self.search_scan_button.configure(state="normal")
        self.active_scan = None

        if match is None:
            self.clear_result()
            self.search_status.configure(text="Không tìm thấy khuôn mặt khớp với nhân viên được cấp quyền.")
            self.show_dialog("Không tìm thấy", "Khuôn mặt vừa quét chưa khớp với nhân viên giám sát nào.", "warning")
            return

        self.display_record(match["record"])
        self.search_status.configure(text=f"Đã nhận diện nhân viên: {match['person_name']} (độ lệch {match['distance']:.2f}).")

    def update_scan_status(self, message):
        if not self.active_scan:
            return
        if self.active_scan["context"] == "add":
            self.add_scan_status.configure(text=message)
        else:
            self.search_status.configure(text=message)

    def cancel_active_scan(self, reset_message=False):
        if not self.active_scan:
            return
        context = self.active_scan["context"]
        self.active_scan = None
        self.scan_encodings = []
        self.scan_success_count = 0
        if context == "add":
            self.add_progress.set(0)
            self.scan_staff_button.configure(state="normal")
            if reset_message:
                self.add_scan_status.configure(text="Đã hủy phiên quét do chuyển tab.")
        else:
            self.search_progress.set(0)
            self.search_scan_button.configure(state="normal")
            if reset_message:
                self.search_status.configure(text="Đã hủy phiên quét do chuyển tab.")

    def save_record(self):
        values = self.get_form_values()
        missing = self.get_missing_fields(values)
        if missing:
            self.show_dialog("Thiếu thông tin", "Vui lòng nhập các trường bắt buộc:\n- " + "\n- ".join(missing), "warning")
            return

        if not self.staff_face_data:
            self.show_dialog("Thiếu khuôn mặt", "Vui lòng quét mặt nhân viên trước khi lưu.", "warning")
            return

        staff_id = values["staff_id"]
        existing_index = self.find_record_index(staff_id)
        if existing_index is not None:
            ok = self.ask_confirm("Mã nhân viên đã tồn tại", f"Nhân viên mã '{staff_id}' đã tồn tại. Bạn có muốn ghi đè hồ sơ này không?")
            if not ok:
                return

        safe_id = sanitize_file_id(staff_id)
        image_path = FACES_DIR / f"{safe_id}_staff.jpg"
        if not cv2.imwrite(str(image_path), self.staff_face_data["frame"]):
            self.show_dialog("Lỗi lưu ảnh", "Không lưu được ảnh khuôn mặt nhân viên.", "error")
            return

        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "staff_id": staff_id,
            "staff_info": {
                "staff_id": staff_id,
                "name": values["staff_name"],
                "role": values["role"],
                "department": values["department"],
                "phone": values["phone"],
                "email": values["email"],
                "active": bool(self.active_checkbox.get()),
                "image_path": path_to_json_value(image_path),
                "face_encoding": self.staff_face_data["encoding"],
                "updated_at": now_text,
            },
        }

        if existing_index is None:
            self.database.append(record)
        else:
            self.database[existing_index] = record

        try:
            save_database(self.database)
        except OSError as error:
            self.show_dialog("Lỗi database", f"Không ghi được data/database.json:\n{error}", "error")
            return

        self.refresh_known_faces()
        self.reset_add_form()
        self.show_dialog("Thành công", "Đã lưu nhân viên giám sát vào database.json.", "success")

    def get_form_values(self):
        return {key: entry.get().strip() for key, entry in self.entries.items()}

    def get_missing_fields(self, values):
        labels = {
            "staff_id": "Mã nhân viên",
            "staff_name": "Họ tên nhân viên",
            "role": "Chức vụ",
            "department": "Bộ phận/khu vực phụ trách",
        }
        return [label for key, label in labels.items() if not values.get(key)]

    def find_record_index(self, staff_id):
        staff_id_lower = staff_id.strip().lower()
        for index, record in enumerate(self.database):
            staff_info = record.get("staff_info", {}) or {}
            current_id = str(record.get("staff_id") or staff_info.get("staff_id") or "").strip().lower()
            if current_id == staff_id_lower:
                return index
        return None

    def reset_add_form(self):
        for entry in self.entries.values():
            entry.delete(0, "end")
        self.active_checkbox.select()
        self.staff_face_data = None
        self.add_progress.set(0)
        self.add_scan_status.configure(text="Chưa quét khuôn mặt nhân viên.")
        self.staff_scan_badge.configure(
            text="Nhân viên: chưa quét",
            fg_color=("#e2e8f0", "#1e293b"),
            text_color=("#0f172a", "#f8fafc"),
        )

    def refresh_known_faces(self):
        self.known_faces = []
        for record in staff_records(self.database):
            staff_info = record.get("staff_info", {}) or {}
            if staff_info.get("active") is False:
                continue
            self.add_known_face(
                record=record,
                staff_id=str(record.get("staff_id") or staff_info.get("staff_id") or "").strip(),
                person_name=staff_info.get("name", ""),
                encoding=staff_info.get("face_encoding"),
            )

    def add_known_face(self, record, staff_id, person_name, encoding):
        if not encoding:
            return
        try:
            encoding_array = np.array(encoding, dtype=np.float64)
            if encoding_array.shape[0] != 128:
                return
        except (TypeError, ValueError):
            return
        self.known_faces.append(
            {
                "record": record,
                "staff_id": staff_id,
                "person_type": "staff",
                "person_label": "Nhân viên giám sát",
                "person_name": str(person_name or "").strip(),
                "encoding": encoding_array,
            }
        )

    def search_by_text(self):
        query = self.search_entry.get().strip().lower()
        if not query:
            self.show_dialog("Thiếu từ khóa", "Vui lòng nhập từ khóa tìm kiếm.", "warning")
            return

        matches = []
        for record in staff_records(self.database):
            staff_info = record.get("staff_info", {}) or {}
            fields = [
                record.get("staff_id", ""),
                staff_info.get("staff_id", ""),
                staff_info.get("name", ""),
                staff_info.get("role", ""),
                staff_info.get("department", ""),
                staff_info.get("phone", ""),
                staff_info.get("email", ""),
            ]
            if any(query in str(value).lower() for value in fields):
                matches.append(record)

        if not matches:
            self.clear_result()
            self.search_status.configure(text="Không tìm thấy nhân viên phù hợp.")
            self.show_dialog("Không tìm thấy", "Không có nhân viên nào khớp từ khóa.", "warning")
            return

        self.display_record(matches[0])
        self.search_status.configure(
            text="Đã tìm thấy 1 nhân viên." if len(matches) == 1 else f"Đã tìm thấy {len(matches)} nhân viên, đang hiển thị kết quả đầu tiên."
        )

    def find_best_face_match(self, encoding):
        if not self.known_faces:
            return None
        known_encodings = [item["encoding"] for item in self.known_faces]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance > FACE_MATCH_TOLERANCE:
            return None
        match = dict(self.known_faces[best_index])
        match["distance"] = best_distance
        return match

    def display_record(self, record):
        staff_info = record.get("staff_info", {}) or {}
        values = {
            "staff_id": record.get("staff_id") or staff_info.get("staff_id", ""),
            "staff_name": staff_info.get("name", ""),
            "role": staff_info.get("role", ""),
            "department": staff_info.get("department", ""),
            "phone": staff_info.get("phone", ""),
            "email": staff_info.get("email", ""),
            "active": "Được phép đăng nhập" if staff_info.get("active", True) else "Đã khóa đăng nhập",
            "updated_at": staff_info.get("updated_at", ""),
        }
        for key, label in self.result_labels.items():
            label.configure(text=values.get(key, ""))
        self.update_result_face_image(record)

    def clear_result(self):
        for label in self.result_labels.values():
            label.configure(text="")
        self.result_face_image = None
        self.result_face_label.image = None
        self.result_face_label.configure(image=None, text="Chưa có ảnh")

    def update_result_face_image(self, record):
        image_path = self.resolve_face_image_path(record)
        image = image_file_to_ctk_preview(image_path, RESULT_FACE_SIZE) if image_path else None
        if image is None:
            self.result_face_image = None
            self.result_face_label.image = None
            self.result_face_label.configure(image=None, text="Chưa có ảnh")
            return
        self.result_face_image = image
        self.result_face_label.image = image
        self.result_face_label.configure(image=image, text="")

    def resolve_face_image_path(self, record):
        staff_info = record.get("staff_info", {}) or {}
        stored_path = str(staff_info.get("image_path", "")).strip()
        if stored_path:
            candidate = Path(stored_path)
            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate
            if candidate.exists():
                return candidate
        safe_id = sanitize_file_id(str(record.get("staff_id") or staff_info.get("staff_id") or ""))
        for suffix in ("jpg", "jpeg", "png"):
            candidate = FACES_DIR / f"{safe_id}_staff.{suffix}"
            if candidate.exists():
                return candidate
        return None

    def on_close(self):
        if self.camera_after_id is not None:
            self.after_cancel(self.camera_after_id)
            self.camera_after_id = None
        self.cancel_active_scan(reset_message=False)
        self.release_camera()
        cv2.destroyAllWindows()
        self.destroy()


def main():
    app = StaffFaceDatabaseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
