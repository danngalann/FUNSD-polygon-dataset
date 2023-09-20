import os
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.dto.Annotation import Annotation


# Transform [xleft, ytop, xright, ytop, xright, ybottom, xleft, ybottom] to [[xleft, ytop], [xright, ytop], [xright, ybottom], [xleft, ybottom]]
def get_points(polygon: list):
    return np.array(polygon, dtype=np.int32).reshape((-1, 2))


# Function to recursively parse the dictionary and draw the boxes
def draw_boxes(annotation: Annotation, image, color):
    points = get_points(annotation.polygon)
    cv2.polylines(image, [points], True, color, 2)

    for word in annotation.words:
        points = get_points(word.polygon)
        # Color a bit lighter
        color = tuple([int(c * 0.8) for c in color])
        cv2.polylines(image, [points], True, color, 2)

    return image


def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def generate_bounding_boxes(image_path: str, annotations: List[Annotation]):
    image_name = os.path.basename(image_path)
    destination_path = Path.cwd() / 'datasets' / 'FUNSD_polygon_augmented' / 'dataset' / 'bounding_boxes' / f'{image_name}.png'

    # Create the destination folder if it doesn't exist
    if not os.path.exists(os.path.dirname(destination_path)):
        os.makedirs(os.path.dirname(destination_path))

    # Load the corresponding image
    image = cv2.imread(image_path)

    # Get the labels
    labels = [annotation.label for annotation in annotations]

    # Assign a color to each label in a dictionary
    color_map = {}
    for label in labels:
        if label not in color_map:
            color_map[label] = get_random_color()

    # Iterate over the sections and draw the boxes
    for annotation in annotations:
        image = draw_boxes(annotation, image, color_map[annotation.label])

    # Save the resulting image
    cv2.imwrite(str(destination_path), image)
