import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class FaceTracking:
    def __init__(self, face_detector, max_age=30):
        self.face_detector = face_detector
        self.tracker = DeepSort(max_age=max_age)
        self.track_colors = {}

    def process_frame(self, frame, input_size=(480, 480), detect_thresh=0.7):
        """Process each video frame to detect and track faces."""
        try:
            # Step 1: Detect faces using SCRFD
            bboxes, lmarks = self.face_detector.detect(
                frame, input_size=input_size, thresh=detect_thresh
            )

            if len(bboxes) == 0:
                return frame  # No faces detected, return original frame

            # Convert to DeepSort format
            dets_deepsort = [
                ([x1, y1, x2 - x1, y2 - y1], conf, 0) for x1, y1, x2, y2, conf in bboxes
            ]

            # Step 2: Update tracker
            tracks = self.tracker.update_tracks(dets_deepsort, frame=frame)

            # Step 3: Draw bounding boxes and track IDs
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                if track_id not in self.track_colors:
                    self.track_colors[track_id] = np.random.randint(
                        0, 255, size=3
                    ).tolist()

                color = self.track_colors[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        except Exception as e:
            print(f"Error during frame processing: {e}")

        return frame
