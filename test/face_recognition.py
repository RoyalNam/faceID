import cv2
import os
import sys
import time
from facenet_pytorch import InceptionResnetV1

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import FaceDetector, FaceRecognizer, FaceSystem


def main(image_path=None, video_source=0):
    detect_model_path = "data/models/detect/model1.onnx"
    face_detector = FaceDetector(model_file=detect_model_path)

    emb_model = InceptionResnetV1(pretrained="vggface2").eval()

    face_recognizer = FaceRecognizer(emb_model)
    face_system = FaceSystem(
        face_detector, face_recognizer, "data/embeddings/embeddings.pkl"
    )

    # Mở video hoặc webcam
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    tm = cv2.TickMeter()
    frame_skip = 5
    frame_count = 0
    s0 = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        tm.start()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Nhận diện khuôn mặt
        faces = face_system.recognize_faces(img, method="cosine", threshold=0.4)
        tm.stop()

        if faces:
            for name, bbox in faces:
                if name == "Unrecognized":
                    label = "Unrecognized"
                    color = (0, 0, 255)  # Red for unrecognized
                else:
                    label = f"Recognized: {name}"
                    color = (0, 255, 0)  # Green for recognized
                    print(f"Recognized: {name}")

                # Vẽ hộp bao quanh khuôn mặt và ghi chú tên
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        fps = tm.getFPS()

        # Hiển thị FPS trên video
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Hiển thị video
        cv2.imshow("Live Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Processing time: {time.time() - s0}s")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main(video_source="data/video/output.mp4")
    main(video_source="data/video/testvid.mp4")
    # main()  # Uncomment to use webcam (video_source=0)
