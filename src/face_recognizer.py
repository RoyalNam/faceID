import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist


class FaceRecognizer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def recognize_faces(
        self, face_images, all_embeddings, method="cdist", threshold=0.8
    ):
        """Recognize multiple faces in an image without using a loop."""

        face_embeddings = self.get_embeddings(face_images)
        recognized_names = self.recognize_from_embeddings_batch(
            face_embeddings, all_embeddings, method, threshold
        )
        return recognized_names

    def get_embeddings(self, face_images):
        """Extract embeddings for multiple images at once."""
        preprocessed_faces = torch.stack(
            [self.preprocess_face(face) for face in face_images]
        )
        with torch.no_grad():
            embeddings = self.embedding_model(preprocessed_faces).detach().cpu().numpy()
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def recognize_from_embeddings_batch(
        self, face_embeddings, all_embeddings, method, threshold
    ):
        """Recognize multiple embeddings at once."""
        all_names, all_stored_embeddings = all_embeddings
        if all_stored_embeddings.size == 0:
            return ["Unrecognized"] * len(face_embeddings)

        # Compute distances or similarities
        if method == "cdist":
            distances = cdist(face_embeddings, all_stored_embeddings, metric="cosine")
        elif method == "cosine":
            distances = 1 - np.dot(face_embeddings, all_stored_embeddings.T)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Find the closest name based on minimum distance
        min_distances = distances.min(axis=1)
        closest_indices = distances.argmin(axis=1)
        recognized_names = [
            all_names[idx] if min_distances[i] < threshold else "Unrecognized"
            for i, idx in enumerate(closest_indices)
        ]
        print(f"min dis: {min_distances}")
        return recognized_names

    def preprocess_face(self, face_image, size=(160, 160)):
        """Preprocess the face image for embedding extraction."""
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(face_image)

        if face_image.mode != "RGB":
            face_image = face_image.convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        return transform(face_image)
