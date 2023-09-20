from fastai.vision.all import *
from fastcore.foundation import L
import cv2
import json
import numpy as np
from typing import Tuple, List, Union

class DataAugmenter:
    def __init__(self, config: dict = {}):
        default_config = {
            "transform_perspective": True,
            "add_lines": True,
            "random_contrast": True,
            "random_brightness": True,
            "random_saturation": True,
            "random_reflections": True,
            "fit_transformed_images": True,
        }

        # Merge the default config with the user config
        config = {**default_config, **config}
        self.config = config

        # Define fastai augmentations based on config
        aug_transform_list = []

        if self.config["transform_perspective"]:
            aug_transform_list.append(Warp(magnitude=0.2))
        if self.config["random_contrast"]:
            aug_transform_list.append(Contrast(max_lighting=0.2, p=1., draw=self.draw))
        if self.config["random_brightness"]:
            aug_transform_list.append(Brightness(max_lighting=0.2, p=1., draw=self.draw))
        if self.config["random_saturation"]:
            aug_transform_list.append(Saturation(max_lighting=0.2, p=1., draw=self.draw))

        # TODO reflections and lines

        self.augs = aug_transforms(mult=1.0, do_flip=False) + aug_transform_list

    # This function allows control over the range of augmentation changes
    def draw(self, b, c, clamp=None, do_slice=True):
        return random.uniform(b - c, b + c)

    def augment(self, image: np.ndarray, coordinates: List[dict]) -> Tuple[np.ndarray, List[dict]]:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.create(image)

        # Convert coordinates to a format suitable for fastai's PointScaler
        points = [coord['polygon'] for coord in coordinates]
        pnts = L(points)

        # Apply augmentations
        tfms = setup_aug_tfms(self.augs)
        xb, yb = [pil_img], [pnts]
        xb, yb = Pipeline(tfms)(xb, yb)

        # Convert back to numpy and coordinates format
        augmented_img = np.array(xb[0])
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

        # Extract augmented points and format as original
        augmented_coords = []
        for original, augmented in zip(coordinates, yb[0]):
            augmented_polygon = [list(augmented[i]) for i in range(4)]
            original["polygon"] = augmented_polygon
            augmented_coords.append(original)

        return augmented_img, augmented_coords
