import os
import pickle
from PIL import Image
import numpy as np
import cv2
import json


def load_pickle(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Returning empty data.")
        return {}

    with open(file_path, "rb") as f:
        try:
            return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading pickle file {file_path}: {e}")
            return {}


def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    """Load JSON data from a file."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Returning empty data.")
        return {}

    with open(file_path, "r") as f:
        try:
            content = f.read().strip()
            if not content:
                print(f"File {file_path} is empty. Returning empty data.")
                return {}
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return {}


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def save_image(path, image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


def load_images_from_folder(folder_path, output_format="np"):
    """Load all images from a folder and return them as a list of PIL Images or numpy arrays."""
    image_list = []

    if not os.path.exists(folder_path):
        print(f"Input folder {folder_path} does not exist.")
        return image_list

    image_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_path in image_paths:
        try:
            if output_format == "PIL":
                image = Image.open(image_path)
            elif output_format == "np":
                image = np.array(Image.open(image_path))
            else:
                raise ValueError("Unsupported output format. Use 'PIL' or 'np'.")

            image_list.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return image_list


def are_coordinates_in_frame(frame, bbox, landmarks):
    """
    Check if the bounding box and landmarks are within the frame boundaries.

    Parameters:
    - frame: The input image frame (numpy array).
    - bbox: The bounding box (array-like of shape [4]).
    - landmarks: The landmarks (array-like of shape [N, 2]).

    Returns:
    - bool: True if all coordinates are within the frame, False otherwise.
    """
    height, width = frame.shape[:2]

    # Check bounding box
    if not (
        0 <= bbox[0] < width
        and 0 <= bbox[1] < height
        and 0 <= bbox[2] < width
        and 0 <= bbox[3] < height
    ):
        return False

    # Check landmarks
    for x, y in landmarks:
        if not (0 <= x < width and 0 <= y < height):
            return False

    return True


# Adjusted variable name and frame visualization
def visualize(image, boxes, lmarks, scores, fps=0):
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = map(int, boxes[i][:4])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        if lmarks is not None and len(lmarks) > i:
            for j in range(lmarks[i].shape[0]):
                x, y = map(int, lmarks[i][j])
                cv2.circle(image, (x, y), 1, (0, 255, 0), thickness=-1)
        cv2.putText(
            image,
            str(round(scores[i].item(), 3)),
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            thickness=1,
        )
    cv2.putText(
        image,
        f"FPS={int(fps)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        thickness=1,
    )
    return image


def is_image_blurry(image, threshold=80):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute the Laplacian of the image
    variance_of_laplacian = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    # Check if the variance is below the threshold
    return variance_of_laplacian < threshold


def is_valid_face(face, laplacian_threshold=80, size_threshold=80):
    # Check if face is too small
    h, w = face.shape[:2]
    if h < size_threshold or w < size_threshold:
        return False
    # Check if image is blurry
    if is_image_blurry(face, laplacian_threshold):
        return False

    return True
