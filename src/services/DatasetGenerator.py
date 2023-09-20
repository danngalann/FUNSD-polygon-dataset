import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path

from src.dto.Annotation import Annotation
from src.services.DataAugmenter import DataAugmenter
from src.utils.draw_bounding_boxes import generate_bounding_boxes


class DatasetGenerator:
    def __init__(self, multiplier: int = 5, generate_debug_bounding_boxes: bool = False):
        self.generate_debug_bounding_boxes = generate_debug_bounding_boxes
        self.dataset_root = Path.cwd() / 'datasets' / 'FUNSD_polygon' / 'dataset'
        self.test_images_path = self.dataset_root / 'testing_data' / 'images'
        self.test_data_path = self.dataset_root / 'testing_data' / 'annotations'
        self.train_data_path = self.dataset_root / 'training_data' / 'annotations'
        self.train_images_path = self.dataset_root / 'training_data' / 'images'

        self.augmented_dataset_root = Path.cwd() / 'datasets' / 'FUNSD_polygon_augmented' / 'dataset'
        self.augmented_test_images_path = self.augmented_dataset_root / 'testing_data' / 'images'
        self.augmented_test_data_path = self.augmented_dataset_root / 'testing_data' / 'annotations'
        self.augmented_train_data_path = self.augmented_dataset_root / 'training_data' / 'annotations'
        self.augmented_train_images_path = self.augmented_dataset_root / 'training_data' / 'images'


        self.createFolders()
        self.data_augmenter = DataAugmenter()
        self.multiplier = multiplier

        self.train_images = {img_path.name: cv2.imread(str(img_path)) for img_path in self.train_images_path.iterdir() if
                             img_path.suffix in ['.jpg', '.png']}
        self.test_images = {img_path.name: cv2.imread(str(img_path)) for img_path in self.test_images_path.iterdir() if
                            img_path.suffix in ['.jpg', '.png']}

    def createFolders(self) -> None:
        # Create folders for augmented images
        if not os.path.exists(self.augmented_dataset_root):
            os.makedirs(self.augmented_dataset_root)

        if not os.path.exists(self.augmented_test_images_path):
            os.makedirs(self.augmented_test_images_path)

        if not os.path.exists(self.augmented_test_data_path):
            os.makedirs(self.augmented_test_data_path)

        if not os.path.exists(self.augmented_train_data_path):
            os.makedirs(self.augmented_train_data_path)

        if not os.path.exists(self.augmented_train_images_path):
            os.makedirs(self.augmented_train_images_path)

    def augment(self) -> None:
        with tqdm(total=len(self.train_images) * self.multiplier, desc='Augmenting training images') as pbar:
            for image_filename, image_data in self.train_images.items():
                json_filename = image_filename.split('.')[0] + '.json'
                json_path = self.train_data_path / json_filename

                with open(json_path, 'r') as f:
                    annotation = json.loads(f.read())
                    annotations = [Annotation.from_dict(annotation) for annotation in annotation['form']]

                for i in range(self.multiplier):
                    # Augment image
                    augmented_image, augmented_annotations = self.data_augmenter.augment(image_data, annotations)

                    # Save augmented image
                    augmented_image_filename = f'{image_filename.split(".")[0]}_augmented_{i}.png'
                    augmented_image_path = self.augmented_train_images_path / augmented_image_filename
                    cv2.imwrite(str(augmented_image_path), augmented_image)

                    # Save augmented data
                    augmented_json_filename = f'{json_filename.split(".")[0]}__augmented_{i}.json'
                    augmented_json_path = self.augmented_train_data_path / augmented_json_filename

                    with augmented_json_path.open('w') as augmented_jsonl_file:
                        augmented_form = {
                            "form": [annotation.to_dict() for annotation in augmented_annotations]
                        }
                        augmented_jsonl_file.write(json.dumps(augmented_form))

                    if self.generate_debug_bounding_boxes:
                        # For debugging purposes, draw bounding boxes
                        generate_bounding_boxes(str(augmented_image_path), augmented_annotations)

                    pbar.update(1)
