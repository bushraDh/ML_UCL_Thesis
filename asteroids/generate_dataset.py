import torch
import pandas as pd
import os
from torch.utils.data import Dataset, ConcatDataset, random_split
from PIL import Image
from torchvision import transforms
import random



class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir,type, transform=None, target_size=(256, 256)):
        self.labels_df = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.type=type

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        if self.type=='CoB':
            label = self.labels_df.iloc[idx, -2:].values.astype(float)  # Ensure label is of type float
        if self.type=='CoM':
            label = self.labels_df.iloc[idx, -4:-2].values.astype(float)  # Ensure label is of type float
        label = torch.tensor(label, dtype=torch.float32)
        # Scale the labels according to the resizing
        original_size = (512, 512)  # Original size of your images
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]
        label[0] *= scale_x
        label[1] *= scale_y

        if self.transform:
            image = self.transform(image)

        return image, label


def load_datasets( image_dirs, runs, type='CoB', transform=None):
    datasets = []
    for i in range(runs):
        image_dir = image_dirs + str(i)
        csv_file = 'data.csv'
        csv_path = os.path.join(image_dir, csv_file)
        if os.path.exists(csv_path) and csv_path.endswith(".csv"):
            file = pd.read_csv(csv_path)  # Skip the first row
            dataset = CustomDataset(csv_file=file, root_dir=image_dir,type=type, transform=transform)
            datasets.append(dataset)
    return datasets


def split_datasets_by_runs(datasets, train_ratio=0.7, val_ratio=0.10, test_ratio=0.15):
    # random.shuffle(datasets)

    total_images = sum(len(dataset) for dataset in datasets)
    train_images = int(train_ratio * total_images)
    val_images = int(val_ratio * total_images)
    test_images = total_images - train_images - val_images

    train_datasets, val_datasets, test_datasets = [], [], []
    cumulative_images = 0

    for dataset in datasets:
        if cumulative_images < train_images:
            train_datasets.append(dataset)
        elif cumulative_images < train_images + val_images:
            val_datasets.append(dataset)
        else:
            test_datasets.append(dataset)
        cumulative_images += len(dataset)

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)


if __name__ == "__main__":
    # Define the transform with resizing to 256x256 and normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization values for grayscale
    ])

    # label_dirs = os.path.join(os.getcwd(), 'data')
    image_dirs = os.path.join(os.getcwd(), 'asteroids\cnn_MC_data_50-100km\\run')

    datasets = load_datasets(image_dirs, runs=473, type='CoB', transform=transform)

    train_dataset, val_dataset, test_dataset = split_datasets_by_runs(datasets, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Save the datasets
    torch.save(train_dataset, 'train_dataset.pth')
    torch.save(val_dataset, 'val_dataset.pth')
    torch.save(test_dataset, 'test_dataset.pth')
    print("Data paths saved")
