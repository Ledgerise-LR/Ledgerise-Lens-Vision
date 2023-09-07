# from preprocess.preprocess import *
import os
from data_setup import create_dataloaders, investigate_data
from utils import save_model
from torchvision import datasets, transforms

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()


def main():
    transform = transforms.Compose(
        [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    )

    train_data_loader, test_data_loader, class_names = create_dataloaders(
        "data/train",
        "data/test",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,  # type: ignore
    )

    # print(
    #     f"Train data loader: {train_data_loader} | Length of train_dataloader {len(train_data_loader)} | {len(train_data_loader.dataset)} total samples."  # type: ignore
    # )
    # print(
    #     f"Test data loader: {test_data_loader} | Length of test_dataloader {len(test_data_loader)} | {len(test_data_loader.dataset)} total samples."  # type: ignore
    # )
    # print(class_names)


if __name__ == "__main__":
    main()
