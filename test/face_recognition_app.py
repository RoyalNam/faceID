import sys
import os
from facenet_pytorch import InceptionResnetV1
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

# Thêm đường dẫn tới các module trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import FaceDetector, FaceRecognizer, FaceSystem, LiveViewLogic


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live View Test")
        self.setGeometry(100, 100, 640, 480)

        # Tạo widget chính và layout
        main_widget = QWidget(self)

        # Khởi tạo FaceSystem
        detect_model_path = "data/models/detect/model1.onnx"
        face_detector = FaceDetector(model_file=detect_model_path)
        emb_model = InceptionResnetV1(pretrained="vggface2").eval()
        face_recognizer = FaceRecognizer(emb_model)
        face_system = FaceSystem(
            face_detector, face_recognizer, "data/embeddings/embeddings.pkl"
        )

        # Khởi tạo LiveViewLogic
        self.live_view_logic = LiveViewLogic(face_system, main_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = FaceRecognitionApp()
    main_win.show()
    sys.exit(app.exec())
