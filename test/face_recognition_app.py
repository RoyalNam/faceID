import sys
import os
from facenet_pytorch import InceptionResnetV1
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import FaceDetector, FaceRecognizer, FaceSystem, LiveViewLogic, RegisterLogic


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        tab_widget = QTabWidget(main_widget)

        detect_model_path = "data/models/detect/model1.onnx"
        face_detector = FaceDetector(model_file=detect_model_path)
        emb_model = InceptionResnetV1(pretrained="vggface2").eval()
        face_recognizer = FaceRecognizer(emb_model)
        face_system = FaceSystem(
            face_detector, face_recognizer, "data/embeddings/embeddings.pkl"
        )

        self.live_view_logic = LiveViewLogic(face_system, tab_widget)
        self.recognition = RegisterLogic(face_system, tab_widget)

        tab_widget.addTab(self.live_view_logic, "Live View")
        tab_widget.addTab(self.recognition, "Register")

        layout = QVBoxLayout(main_widget)
        layout.addWidget(tab_widget)
        main_widget.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = FaceRecognitionApp()
    main_win.show()
    sys.exit(app.exec())
