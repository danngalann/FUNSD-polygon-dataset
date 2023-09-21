from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import random

from src.dto.Annotation import Annotation


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

        self.bg_images = [cv2.imread(str(bg_path)) for bg_path in Path.cwd().glob('datasets/backgrounds/*.jpg')]

    # This function allows control over the range of augmentation changes
    def draw(self, b, c, clamp=None, do_slice=True):
        return random.uniform(b - c, b + c)

    def coin_flip(self) -> bool:
        return random.randint(0, 1) == 1

    def augment(
            self,
            image: np.ndarray,
            annotations: List[Annotation]
    ) -> Tuple[np.ndarray, List[Annotation]]:
        if self.config['transform_perspective']:
            bg_image = None
            if len(self.bg_images) > 0:
                bg_image = random.choice(self.bg_images)

            image, annotations = self.apply_perspective(image, annotations, bg_image)

        if self.config['add_lines'] and self.coin_flip():
            image = self.add_lines(image)

        if self.config['random_contrast'] and self.coin_flip():
            image = self.random_contrast(image)

        if self.config['random_brightness'] and self.coin_flip():
            image = self.random_brightness(image)

        if self.config['random_saturation'] and self.coin_flip():
            image = self.random_saturation(image)

        if self.config['random_reflections'] and self.coin_flip():
            image = self.random_reflections(image)

        return image, annotations

    def apply_perspective(
            self,
            image: np.ndarray,
            annotations: List[Annotation],
            bg_image: np.ndarray = None
    ) -> Tuple[np.ndarray, List[Annotation]]:
        # Compute the size of the image
        height, width = image.shape[:2]

        # Define points for a simple perspective transformation
        src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

        # Define a function to generate random shift for each corner
        def random_shift(max_shift):
            return random.uniform(-max_shift, max_shift)

        # Generate random destination points by shifting each corner by a random amount
        max_shift = min(height, width) * 0.2  # for example, allow shift up to 20% of the smaller dimension
        dst_points = np.float32([[random_shift(max_shift), random_shift(max_shift)],
                                 [width - 1 + random_shift(max_shift), random_shift(max_shift)],
                                 [random_shift(max_shift), height - 1 + random_shift(max_shift)],
                                 [width - 1 + random_shift(max_shift), height - 1 + random_shift(max_shift)]])

        if self.config["fit_transformed_images"]:
            # Adjust destination points to ensure all points fit into the image dimensions
            min_x = min(point[0] for point in dst_points)
            min_y = min(point[1] for point in dst_points)
            max_x = max(point[0] for point in dst_points)
            max_y = max(point[1] for point in dst_points)

            # Shift all points by the negative of these minimum values
            if min_x < 0:
                dst_points[:, 0] -= min_x
                max_x -= min_x
            if min_y < 0:
                dst_points[:, 1] -= min_y
                max_y -= min_y

            # Scale the points to fit into the original image dimensions
            scale_x = width / max_x
            scale_y = height / max_y
            dst_points[:, 0] *= scale_x
            dst_points[:, 1] *= scale_y

        # Generate the perspective transformation matrix
        perspective_mat = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(image, perspective_mat, (width, height))

        if bg_image is not None:
            # If a background image is provided, resize it to match the dimensions of the original image
            bg_image = cv2.resize(bg_image, (width, height))

            # Create a white image and warp it using the same transformation matrix
            white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            warped_white_image = cv2.warpPerspective(white_image, perspective_mat, (width, height))

            # Create a mask identifying where the warped_white_image has introduced new black pixels
            mask = (warped_white_image == 0).all(axis=2)

            warped_image[mask] = bg_image[mask]

        # Transform all coordinates
        new_annotations = []
        for annotation in annotations:
            annotation = self.transform_coords(annotation, perspective_mat)
            new_annotations.append(annotation)

        return warped_image, new_annotations

    def transform_polygon(self, polygon, perspective_mat):
        # Initializing a new polygon with the same size
        new_polygon = [0] * len(polygon)
        # Since the polygon is defined as [x1, y1, x2, y2, ...], we can pair them by iterating two steps at a time
        for i in range(0, len(polygon), 2):
            coordinate = np.array([[polygon[i], polygon[i + 1]]], dtype='float32')
            coordinate = np.array([coordinate])
            transformed_coordinate = cv2.perspectiveTransform(coordinate, perspective_mat)[0][0]
            new_polygon[i] = float(transformed_coordinate[0])
            new_polygon[i + 1] = float(transformed_coordinate[1])
        return new_polygon

    def transform_coords(self, annotation: Annotation, perspective_mat):
        # Transform main polygon of Annotation
        annotation.polygon = self.transform_polygon(annotation.polygon, perspective_mat)

        # Transform each WordAnnotation's polygon
        for word in annotation.words:
            word.polygon = self.transform_polygon(word.polygon, perspective_mat)

        return annotation

    def random_reflections(self, image: np.ndarray) -> np.ndarray:
        row, col, ch = image.shape

        # Multiple random rectangles
        for _ in range(np.random.randint(1, 6)):
            rect_mask = np.zeros((row, col, ch), dtype=np.uint8)
            pt1 = (np.random.randint(0, row // 2), np.random.randint(0, col // 2))
            pt2 = (pt1[0] + np.random.randint(0, row // 4), pt1[1] + np.random.randint(0, col // 4))
            cv2.rectangle(rect_mask, pt1, pt2,
                          (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                          -1)
            rect_mask = rect_mask.astype(image.dtype)
            image = cv2.addWeighted(image, 1, rect_mask, 0.3, 0)

        # Multiple random hexagons
        for _ in range(np.random.randint(1, 4)):
            hex_mask = np.zeros((row, col, ch), dtype=np.uint8)
            hex_pts = np.array([[np.random.randint(0, row), np.random.randint(0, col)] for _ in range(6)], np.int32)
            cv2.fillConvexPoly(hex_mask, hex_pts,
                               (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            hex_mask = cv2.GaussianBlur(hex_mask, (23, 23), 30)  # blur to simulate reflection
            hex_mask = hex_mask.astype(image.dtype)  # ensure the mask is the same type as the image
            image = cv2.addWeighted(image, 1, hex_mask, 0.3, 0)

        return image

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

        return noisy

    def add_lines(self, image: np.ndarray) -> np.ndarray:
        row, col, ch = image.shape

        # Multiple random lines
        for _ in range(np.random.randint(1, 6)):
            cv2.line(image, (np.random.randint(0, row), np.random.randint(0, col)),
                     (np.random.randint(0, row), np.random.randint(0, col)),
                     (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                     np.random.randint(1, 3))

        return image

    def random_contrast(self, image: np.ndarray) -> np.ndarray:
        alpha = 1.0 + 0.7 * (np.random.rand() - 0.5)  # Contrast control (1.0-3.0)
        gamma = np.random.randint(0, 2)  # Randomly generate gamma value
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

        return image

    def random_brightness(self, image: np.ndarray) -> np.ndarray:
        beta = np.random.randint(0, 100)
        image = cv2.add(image, beta)

        return image

    def random_saturation(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s = cv2.add(s, np.random.randint(0, 100))
        s = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        return s