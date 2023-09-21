import copy
import os
from concurrent.futures import ThreadPoolExecutor

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

    def worker(self, image_filename, image_data, i, image_save_path, annotations_path, annotation_save_path):
        json_filename = image_filename.split('.')[0] + '.json'
        json_path = annotations_path / json_filename

        with open(json_path, 'r') as f:
            annotation = json.loads(f.read())
            original_annotations = [Annotation.from_dict(annotation) for annotation in annotation['form']]

        annotations_to_augment = copy.deepcopy(original_annotations)

        # Augment image
        augmented_image, augmented_annotations = self.data_augmenter.augment(image_data, annotations_to_augment)

        # Save augmented image
        augmented_image_filename = f'{image_filename.split(".")[0]}_augmented_{i}.png'
        augmented_image_path = image_save_path / augmented_image_filename
        cv2.imwrite(str(augmented_image_path), augmented_image)

        # Save augmented data
        augmented_json_filename = f'{json_filename.split(".")[0]}__augmented_{i}.json'
        augmented_json_path = annotation_save_path / augmented_json_filename

        with augmented_json_path.open('w') as augmented_jsonl_file:
            augmented_form = {
                "form": [annotation.to_dict() for annotation in augmented_annotations]
            }
            augmented_jsonl_file.write(json.dumps(augmented_form))

        if self.generate_debug_bounding_boxes:
            # For debugging purposes, draw bounding boxes
            generate_bounding_boxes(str(augmented_image_path), augmented_annotations)

    def augment_images(self, images, image_save_path, annotations_path, annotation_save_path, pbar):
        with ThreadPoolExecutor() as executor:
            futures = []
            for image_filename, image_data in images.items():
                for i in range(self.multiplier):
                    future = executor.submit(self.worker, image_filename, image_data, i, image_save_path, annotations_path, annotation_save_path)
                    futures.append(future)

            for future in futures:
                future.result()
                pbar.update(1)

    def augment(self) -> None:
        with tqdm(total=(len(self.train_images) + len(self.test_images)) * self.multiplier, desc='Augmenting images') as pbar:
            self.augment_images(self.train_images, self.augmented_train_images_path, self.train_data_path, self.augmented_train_data_path, pbar)
            self.augment_images(self.test_images, self.augmented_test_images_path, self.test_data_path, self.augmented_test_data_path, pbar)
