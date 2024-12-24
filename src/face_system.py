import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cosine
import logging
import cv2
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    load_pickle,
    save_pickle,
    save_image,
    load_json,
    save_json,
    load_images_from_folder,
    is_valid_face,
)
from src import FaceDetector, FaceRecognizer


class FaceSystem:
    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        embeddings_file,
        angles_file="data/saved_angles.json",
        image_dir="data/images",
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.embeddings_file = embeddings_file
        self.angles_file = angles_file
        self.image_dir = image_dir

        self.embeddings = self.load_embeddings()
        self.saved_angles = self.load_saved_angles()

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.update_all_stored_embeddings()

    def register_face(
        self, user_id, image_input, detect_faces=True, input_size=(352, 256)
    ):
        """Register a face or multiple faces to the system."""
        if user_id not in self.embeddings:
            self.embeddings[user_id] = []

        images = self._load_images(image_input)
        if not images:
            self.logger.warning(
                f"No valid images provided for user {user_id}. Skipping."
            )
            return

        cropped_faces_dir = os.path.join(self.image_dir, "known_faces", user_id)
        next_idx = self.get_next_index(cropped_faces_dir)

        for image in images:
            next_idx = self._process_registration(
                image, user_id, cropped_faces_dir, next_idx, detect_faces, input_size
            )

        self.save_embeddings()
        self.update_all_stored_embeddings()

    def register_from_video(self, video_path, user_id, tolerance=20, max_angle=100):
        """Process video frames and register faces."""
        cap = cv2.VideoCapture(video_path)

        cropped_faces_dir = os.path.join("data/images/unknown_faces", user_id)
        raw_faces_dir = os.path.join("data/images/raw_images", user_id)
        next_idx = self.get_next_index(raw_faces_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            next_idx = self._extract_and_save_face(
                frame_rgb,
                user_id,
                next_idx,
                cropped_faces_dir,
                raw_faces_dir,
                tolerance,
                max_angle,
            )

        cap.release()
        self.save_saved_angles()

        self._register_faces_from_video(cropped_faces_dir, user_id)

    def _process_registration(
        self, image, user_id, cropped_faces_dir, next_idx, detect_faces, input_size
    ):
        """Process and register a single face."""
        try:
            if detect_faces:
                bboxes, _ = self.detector.detect(
                    image, input_size=input_size, max_num=1
                )
                if not bboxes:
                    self.logger.warning(
                        f"No face detected for user {user_id}. Skipping."
                    )
                    return next_idx
                face = self.get_face(image, bboxes[0])
            else:
                face = image

            new_embedding = self.recognizer.get_embeddings([face])[0]
            if not self.is_duplicate_embedding(new_embedding, user_id):
                self.embeddings[user_id].append(new_embedding)
                cropped_path = os.path.join(cropped_faces_dir, f"{next_idx}.jpg")
                self.save_face(cropped_path, face)
                next_idx += 1
            else:
                self.logger.info(
                    f"Duplicate face detected for user {user_id}. Skipping."
                )
        except Exception as e:
            self.logger.error(f"Error processing image for {user_id}: {e}")
        return next_idx

    def _load_images(self, image_input):
        """Load images based on input type."""
        if isinstance(image_input, str):
            if os.path.isdir(image_input):
                return load_images_from_folder(image_input)
            else:
                self.logger.error(
                    f"Invalid directory: {image_input}. Skipping registration."
                )
                return []
        elif isinstance(image_input, list):
            return image_input
        else:
            return [image_input]

    def _extract_and_save_face(
        self,
        image,
        user_id,
        next_idx,
        cropped_faces_dir,
        raw_faces_dir,
        tolerance,
        max_angle,
    ):
        """Extract faces from a video frame and save the cropped and raw images."""
        bboxes, kps = self.detector.detect(image, input_size=(352, 256), max_num=1)
        if len(bboxes) == 0:
            self.logger.warning(f"No face detected for user {user_id}. Skipping.")
            return next_idx

        face = self.get_face(image, bboxes[0])

        # Check if the face is valid (not blurry or too small)
        if not is_valid_face(face, laplacian_threshold=80):
            # self.logger.warning(f"Invalid face detected. Skipping.")
            return next_idx

        # Get pose angles for the detected face
        roll, yaw, pitch = self.detector.find_pose(kps[0])
        if not self.is_angle_unique(roll, yaw, pitch, user_id, tolerance, max_angle):
            # self.logger.info(f"Face skipped: roll={roll:.2f}, yaw={yaw:.2f}, pitch={pitch:.2f} exceeds limits.")
            return next_idx
        self.saved_angles.setdefault(user_id, []).append((roll, yaw, pitch))
        # Save the cropped face image and raw frame image
        cropped_path = os.path.join(cropped_faces_dir, f"{next_idx}.jpg")
        raw_path = os.path.join(raw_faces_dir, f"{next_idx}.jpg")

        self.save_face(cropped_path, face)
        self.save_face(raw_path, image)

        return next_idx + 1

    def _register_faces_from_video(self, cropped_faces_dir, user_id):
        """Register faces from the video if any were detected."""
        if os.path.exists(cropped_faces_dir) and os.listdir(cropped_faces_dir):
            self.register_face(user_id, cropped_faces_dir, detect_faces=False)
            self.logger.info(f"Registered faces from video frames for user {user_id}.")
        else:
            self.logger.warning(f"No faces registered for user {user_id} from video.")

        # Clean up temporary folder
        if os.path.exists(cropped_faces_dir):
            shutil.rmtree(cropped_faces_dir)
            self.logger.info(f"Temporary folder deleted: {cropped_faces_dir}")

    def recognize_faces(self, image, method="cdist", threshold=0.8):
        """Recognize multiple faces in an image without using a loop."""
        bboxes, _ = self.detector.detect(image, input_size=(320, 256))
        face_images = [self.get_face(image, bbox) for bbox in bboxes]
        if len(bboxes) == 0:
            return []

        recognized_names = self.recognizer.recognize_faces(
            face_images, (self.all_names, self.all_stored_embeddings), method, threshold
        )
        return list(zip(recognized_names, bboxes))

    def get_face(self, image, bbox):
        x1, y1, x2, y2, _ = bbox
        return image[int(y1) : int(y2), int(x1) : int(x2)]

    def is_duplicate_embedding(self, new_embedding, name):
        return any(
            cosine(new_embedding, emb) < 0.01 for emb in self.embeddings.get(name, [])
        )

    def save_face(self, path, image):
        """Asynchronously save a cropped face image."""
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.executor.submit(save_image, path, image)

    def get_next_index(self, folder):
        """Get the next available index for naming files in a folder."""
        if os.path.exists(folder):
            existing_files = os.listdir(folder)
            # Extract valid numeric indices from file names
            existing_idxs = []
            for f in existing_files:
                if f.endswith(".jpg"):
                    try:
                        idx = int(f.split(".")[0])
                        existing_idxs.append(idx)
                    except ValueError:
                        self.logger.warning(f"Skipping invalid file name: {f}")
            return max(existing_idxs, default=0) + 1
        return 0

    def load_embeddings(self):
        """Load embeddings from file."""
        return load_pickle(self.embeddings_file)

    def save_embeddings(self):
        """Save embeddings to file."""
        save_pickle(self.embeddings, self.embeddings_file)

    def load_saved_angles(self):
        """Load saved angles from file."""
        return load_json(self.angles_file)

    def save_saved_angles(self):
        """Save angles to file."""
        save_json(self.saved_angles, self.angles_file)

    def delete_user(self, name):
        """Delete a user and their associated data."""
        if name in self.embeddings:
            del self.embeddings[name]
            self.save_embeddings()
            self.update_all_stored_embeddings()

            if name in self.saved_angles:
                del self.saved_angles[name]
                self.save_saved_angles()

            user_image_dir = os.path.join(self.image_dir, "known_faces", name)
            user_raw_image_dir = os.path.join(self.image_dir, "raw_images", name)
            user_unknown_faces_dir = os.path.join(self.image_dir, "unknown_faces", name)

            for folder in [user_image_dir, user_raw_image_dir, user_unknown_faces_dir]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    self.logger.info(f"Deleted folder: {folder}")

            self.logger.info(
                f"User {name} and all associated data deleted successfully."
            )
        else:
            self.logger.info(f"User {name} not found.")

    def update_all_stored_embeddings(self):
        """Update all_stored_embeddings and all_names."""
        self.all_stored_embeddings = []
        self.all_names = []

        for name, stored_embeddings in self.embeddings.items():
            if stored_embeddings is not None:
                self.all_stored_embeddings.append(np.array(stored_embeddings))
                self.all_names.extend([name] * len(stored_embeddings))

        if self.all_stored_embeddings:
            self.all_stored_embeddings = np.vstack(self.all_stored_embeddings)
        else:
            self.all_stored_embeddings = np.empty((0, self.recognizer.output_size))

    def is_angle_acceptable(self, roll, yaw, pitch, max_angle=100):
        """Check if the face angles are within acceptable limits."""
        return (
            abs(roll) <= max_angle and abs(yaw) <= max_angle and abs(pitch) <= max_angle
        )

    def is_angle_unique(self, roll, yaw, pitch, name, tolerance=20, max_angle=100):
        """Check if the face angles are unique and within the tolerance limit."""
        if not self.is_angle_acceptable(roll, yaw, pitch, max_angle=max_angle):
            return False
        user_angles = self.saved_angles.get(name, [])
        for saved_roll, saved_yaw, saved_pitch in user_angles:
            if (
                abs(roll - saved_roll) <= tolerance
                and abs(yaw - saved_yaw) <= tolerance
                and abs(pitch - saved_pitch) <= tolerance
            ):
                return False
        return True
