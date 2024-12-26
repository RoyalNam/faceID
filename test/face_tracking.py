import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import FaceTracking


def main(video_source=0, model_file="data/models/detect/model1.onnx"):
    face_tracking = FaceTracking(model_file=model_file)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to access the video.")
        return
    tm = cv2.TickMeter()
    frame_skip = 5
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  #

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Xử lý khung hình với SCRFD và DeepSort
        tm.start()
        frame = face_tracking.process_frame(
            frame_rgb, input_size=(352, 256), detect_thresh=0.6
        )
        tm.stop()

        # Hiển thị FPS trên khung hình
        cv2.putText(
            frame,
            f"FPS: {int(tm.getFPS())}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        # Hiển thị khung hình
        cv2.imshow("Live Face Detection & Tracking", frame)

        # Dừng chương trình nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(video_source="data/video/test.mp4")
