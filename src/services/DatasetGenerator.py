import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path

from src.services.DataAugmenter import DataAugmenter
from src.utils.draw_bounding_boxes import generate_bounding_boxes


class DatasetGenerator:
    def __init__(self, multiplier: int = 100):
        self.dataset_root = Path.cwd() / 'dataset'
        self.images_path = self.dataset_root / 'images'
        self.data_path = self.dataset_root / 'data'

        self.__check_folders()
        self.data_augmenter = DataAugmenter()
        self.multiplier = multiplier

        # Load all images into memory
        self.images = {img_path.name: cv2.imread(str(img_path)) for img_path in self.images_path.iterdir() if
                       img_path.suffix in ['.jpg', '.png']}

    def __check_folders(self) -> None:
        if not self.data_path.exists():
            raise Exception(f'Data folder does not exist in {self.data_path}')
        if not self.images_path.exists():
            raise Exception(f'Images folder does not exist in {self.images_path}')

    def augment(self) -> None:
        with tqdm(total=len(self.images) * self.multiplier, desc='Augmenting images') as pbar:
            for image_filename, image_data in self.images.items():
                jsonl_filename = image_filename.split('.')[0] + '.jsonl'
                jsonl_path = self.data_path / jsonl_filename

                with open(jsonl_path, 'r') as f:
                    coordinates_data = [json.loads(line) for line in f]

                for i in range(self.multiplier):
                    # Augment image
                    augmented_image, coordinates = self.data_augmenter.augment(image_data, coordinates_data)

                    # Save augmented image
                    augmented_image_filename = f'{image_filename.split(".")[0]}_augmented_{i}.png'
                    augmented_image_path = self.images_path / augmented_image_filename
                    cv2.imwrite(str(augmented_image_path), augmented_image)

                    # Save augmented data
                    augmented_jsonl_filename = f'{jsonl_filename.split(".")[0]}__augmented_{i}.jsonl'
                    augmented_jsonl_path = self.data_path / augmented_jsonl_filename

                    with augmented_jsonl_path.open('w') as augmented_jsonl_file:
                        for line in coordinates:
                            augmented_jsonl_file.write(json.dumps(line) + '\n')

                    if os.environ.get('CREATE_BOUNDING_BOXES', 'false') == 'true':
                        # For debugging purposes, draw bounding boxes
                        generate_bounding_boxes(str(augmented_jsonl_path), str(augmented_image_path))

                    pbar.update(1)
