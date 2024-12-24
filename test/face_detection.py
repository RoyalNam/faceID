import cv2
import time
import os
import sys

# Add parent directory to the path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_detector import FaceDetector
from src.utils import are_coordinates_in_frame


def main(image_path=None, video_source=0):
    face_detection_model_path = "data/models/detect/model1.onnx"
    face_detector = FaceDetector(model_file=face_detection_model_path)
    tick_meter = cv2.TickMeter()

    if image_path:
        # Process a static image
        image = cv2.imread(image_path)
        bounding_boxes, landmarks = face_detector.detect(image, thresh=0.5)
        print("Detected faces:", bounding_boxes)
        for box in bounding_boxes:
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2,
            )
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)

    else:
        # Process a video or webcam feed
        video_capture = cv2.VideoCapture(video_source)
        if not video_capture.isOpened():
            print("Error: Unable to access the camera or video file.")
            return

        start_time = time.time()
        frame_skip_interval = 5
        current_frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            current_frame_count += 1
            if current_frame_count % frame_skip_interval != 0:
                continue

            # Convert frame to RGB format for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tick_meter.start()
            try:
                bounding_boxes, face_landmarks = face_detector.detect(
                    rgb_frame, input_size=(352, 256)
                )
            except Exception as error:
                print(f"Error in detection: {error}")
                continue
            tick_meter.stop()

            if bounding_boxes.shape[0] > 0:
                for i in range(len(bounding_boxes)):
                    x_min, y_min, x_max, y_max = map(int, bounding_boxes[i][:4])
                    cv2.rectangle(
                        frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=2
                    )
                    if face_landmarks is not None and len(face_landmarks) > i:
                        for j in range(face_landmarks[i].shape[0]):
                            x, y = map(int, face_landmarks[i][j])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), thickness=-1)
                cv2.putText(
                    frame,
                    f"FPS={int(tick_meter.getFPS())}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    thickness=1,
                )
                for i in range(len(bounding_boxes)):
                    if are_coordinates_in_frame(
                        frame, bounding_boxes[i], face_landmarks[i]
                    ):
                        roll, yaw, pitch = face_detector.find_pose(face_landmarks[i])

                        # Draw arrow and angle info
                        nose_landmark = tuple(
                            map(int, face_landmarks[i][2])
                        )  # Using the nose landmark
                        end_point = (
                            nose_landmark[0] - int(yaw),
                            nose_landmark[1] - int(pitch),
                        )
                        cv2.arrowedLine(frame, nose_landmark, end_point, (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            f"Roll: {int(roll)}, Yaw: {int(yaw)}, Pitch: {int(pitch)}",
                            (20, 60 + i * 30),  # Adjust position for multiple faces
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            thickness=1,
                        )
                        print(f"Face {i}: Roll={roll}, Yaw={yaw}, Pitch={pitch}")

            cv2.imshow("Live Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time} seconds")

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(video_source="data/video/testvid.mp4")
