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
