import cv2
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QWidget

# from src import FaceSystem
from src.ui import Ui_Recognition


class LiveViewLogic(QWidget):
    def __init__(self, face_system, parent=None):
        super().__init__(parent)
        self.ui_recognition = Ui_Recognition()
        self.ui_recognition.setupUi(self)

        self.face_system = face_system
        self.cap = None
        self.timer = QTimer()
        self.captured_images = []
        self.tm = cv2.TickMeter()
        self.frame_count = 0  # Counter to track the number of frames
        self.last_faces = []  # Store the last recognized faces
        self.last_fps = 0  # Store the last calculated FPS
        self.video_path = None  # Store the selected video path
        self.source = None  # 'camera' or 'video'

        # Connect UI buttons to their respective functions
        self.ui_recognition.btn_select_video.clicked.connect(self.select_video)
        self.ui_recognition.btn_use_camera.clicked.connect(self.select_camera)
        self.ui_recognition.btn_start.clicked.connect(
            lambda: self.start_camera(self.default_callback)
        )
        self.ui_recognition.btn_stop.clicked.connect(self.stop_camera)

    def select_video(self):
        """Open a file dialog to select a video."""
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(
            None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.source = "video"
            self.ui_recognition.feed_view.setText(f"Selected: {path}")
        else:
            self.ui_recognition.feed_view.setText("No video selected!")

    def select_camera(self):
        """Set the source to the default camera."""
        self.source = "camera"
        self.ui_recognition.feed_view.setText("Using Camera as source")

    def start_camera(self, callback=None):
        """Start the camera or video feed."""
        if self.source == "video":
            if not self.video_path:
                self.ui_recognition.feed_view.setText("No video selected!")
                return
            self.cap = cv2.VideoCapture(self.video_path)
        elif self.source == "camera":
            self.cap = cv2.VideoCapture(0)  # Default camera
        else:
            self.ui_recognition.feed_view.setText("No source selected!")
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

        self.frame_count += 1  # Increment frame count
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape

        if self.frame_count % 5 == 0:  # Only recognize every 5 frames
            self.tm.reset()
            self.tm.start()
            self.last_faces = self.face_system.recognize_faces(
                rgb_frame, method="cosine", threshold=0.4
            )
            self.tm.stop()
            self.last_fps = self.tm.getFPS()

        # Use last_faces and last_fps for drawing
        faces = self.last_faces
        fps = self.last_fps

        if faces:
            for name, bbox in faces:
                label = (
                    f"Recognized: {name}" if name != "Unrecognized" else "Unrecognized"
                )
                color = (0, 255, 0) if name != "Unrecognized" else (0, 0, 255)
                cv2.rectangle(
                    rgb_frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2,
                )
                cv2.putText(
                    rgb_frame,
                    label,
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Display FPS
        cv2.putText(
            rgb_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        qimage = QImage(rgb_frame.data, width, height, QImage.Format.Format_RGB888)
        callback(qimage)

    def stop_camera(self):
        """Stop the camera feed."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()

    def capture_image(self):
        """Capture an image from the current frame."""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if ret:
            self.captured_images.append(frame)
            return frame
        return None

    def default_callback(self, qimage):
        """Default callback to update the UI feed."""
        self.ui_recognition.feed_view.setPixmap(QPixmap.fromImage(qimage))
