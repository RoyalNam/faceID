from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import QFileDialog, QWidget
import os
import cv2
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap

from src.ui.register import Ui_Register
from src import FaceSystem


class RegisterLogic(QWidget):
    def __init__(self, face_system: FaceSystem, parent=None):
        super().__init__(parent)
        self.ui_register = Ui_Register()
        self.ui_register.setupUi(self)

        self.face_system = face_system
        self.cap = None
        self.timer = QTimer()
        self.tm = cv2.TickMeter()
        self.last_faces = []  # Store the last recognized faces
        self.last_fps = 0  # Store the last calculated FPS
        self.video_path = None  # Store the selected video path
        self.source = None  # 'camera' or 'video'

        # Connect UI buttons to their respective functions
        self.ui_register.btn_select_video.clicked.connect(self.select_video)
        self.ui_register.btn_use_camera.clicked.connect(self.select_camera)
        self.ui_register.btn_start.clicked.connect(
            lambda: self.start_camera(self.default_callback)
        )
        self.ui_register.btn_stop.clicked.connect(self.stop_camera)
        self.ui_register.btn_register.clicked.connect(self.handle_register)

    def select_video(self):
        """Open a file dialog to select a video."""
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(
            None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.source = "video"
            self.ui_register.feed_view.setText(f"Selected: {path}")
        else:
            self.ui_register.feed_view.setText("No video selected!")

    def select_camera(self):
        """Set the source to the default camera."""
        self.source = "camera"
        self.ui_register.feed_view.setText("Using Camera as source")

    def start_camera(self, callback=None):
        """Start the camera or video feed."""
        if self.source == "video":
            if not self.video_path:
                self.ui_register.feed_view.setText("No video selected!")
                return
            self.cap = cv2.VideoCapture(self.video_path)
        elif self.source == "camera":
            self.cap = cv2.VideoCapture(0)  # Default camera
        else:
            self.ui_register.feed_view.setText("No source selected!")
            return

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera or video.")

        try:
            self.timer.timeout.disconnect()
        except TypeError:
            pass

        if callback is None:
            callback = self.default_callback

        self.timer.timeout.connect(lambda: self.update_frame(callback))
        self.timer.start(30)

    def update_frame(self, callback):
        """Capture and send the current frame."""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame.")
            self.stop_camera()
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape

        qimage = QImage(rgb_frame.data, width, height, QImage.Format.Format_RGB888)
        callback(qimage)

    def stop_camera(self):
        """Stop the camera feed."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()

    def default_callback(self, qimage):
        """Default callback to update the UI feed."""
        self.ui_register.feed_view.setPixmap(QPixmap.fromImage(qimage))

    def display_captured_image(self, image):
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QtGui.QImage(
            image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888
        )

        # Tạo một QLabel để hiển thị hình ảnh trong ScrollArea
        image_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)

        # Thêm QLabel vào ScrollArea
        layout = self.ui_register.scrollAreaWidgetContents.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout()
            self.ui_register.scrollAreaWidgetContents.setLayout(layout)
        layout.addWidget(image_label)

    def handle_register(self):
        """Đăng ký người dùng với các ảnh đã chụp."""
        username = self.ui_register.lbl_username.text().strip()
        try:
            self.register_user(username)
            QtWidgets.QMessageBox.information(
                None,
                "Thành công",
                f"Người dùng '{username}' đã được đăng ký thành công!",
            )
        except ValueError as e:
            QtWidgets.QMessageBox.warning(None, "Lỗi", str(e))

    def register_user(self, name):
        """Đăng ký người dùng với các ảnh đã chụp."""
        if not name:
            raise ValueError("Tên người dùng không được để trống.")
        if self.source == "video":
            if not self.video_path:
                self.ui_register.feed_view.setText("No video selected!")
                return
            self.face_system.register_from_video(self.video_path, name)

    def register_from_video(self):
        """Đăng ký người dùng từ video."""
        username = self.ui_register.lbl_username.text().strip()
        if not username:
            QtWidgets.QMessageBox.warning(
                None, "Lỗi", "Tên người dùng không được để trống."
            )
            return

        video_path, _ = QFileDialog.getOpenFileName(
            None, "Chọn video để đăng ký", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not video_path or not os.path.exists(video_path):
            QtWidgets.QMessageBox.warning(None, "Lỗi", "Không tìm thấy file video.")
            return

        try:
            self.face_system.register_from_video(video_path, username)
            QtWidgets.QMessageBox.information(
                None,
                "Thành công",
                f"Người dùng '{username}' đã được đăng ký từ video thành công!",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Lỗi", f"Đăng ký thất bại: {str(e)}")
