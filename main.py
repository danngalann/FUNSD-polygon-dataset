from src.services.DatasetGenerator import DatasetGenerator
import argparse

args = argparse.ArgumentParser()
args.add_argument('--multiplier', type=int, required=False, default=5, help='Number of augmented images per image')

arguments = args.parse_args()


def augment_dataset():
    dataset_augmenter = DatasetGenerator(arguments.multiplier)
    dataset_augmenter.augment()


if __name__ == '__main__':
    augment_dataset()
