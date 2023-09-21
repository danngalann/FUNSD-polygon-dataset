from src.services.DatasetGenerator import DatasetGenerator
import argparse

args = argparse.ArgumentParser()
args.add_argument('--multiplier', type=int, required=False, default=5, help='Number of augmented images per image')
args.add_argument('--generate-bounding-boxes', type=bool, required=False, default=False)

arguments = args.parse_args()


def augment_dataset():
    dataset_augmenter = DatasetGenerator(arguments.multiplier, arguments.generate_bounding_boxes)
    dataset_augmenter.augment()


if __name__ == '__main__':
    augment_dataset()
