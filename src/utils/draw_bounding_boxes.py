import cv2
import json
import os
import random
import numpy as np


# Function to recursively parse the dictionary and draw the boxes
def draw_boxes(item, image, color):
    points = np.array(item['polygon'], dtype=np.int32)

    cv2.polylines(image, [points], True, color, 2)

    return image

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_bounding_boxes(data_path, image_path):
    image_name = os.path.basename(image_path)
    destination_path = os.path.join('dataset', 'bounding_boxes', f'{image_name}.png')

    # Create the destination folder if it doesn't exist
    if not os.path.exists(os.path.dirname(destination_path)):
        os.makedirs(os.path.dirname(destination_path))

    # Load the data from the JSONL file
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Load the corresponding image
    image = cv2.imread(image_path)

    # Get the labels
    labels = [item['label'] for item in data]
    
    # Assign a color to each label in a dictionary
    color_map = {}
    for label in labels:
        if label not in color_map:
            color_map[label] = get_random_color()

    # Iterate over the sections and draw the boxes
    for coordinate in data:
        image = draw_boxes(coordinate, image, color_map[coordinate['label']])

    # Save the resulting image
    cv2.imwrite(destination_path, image)
