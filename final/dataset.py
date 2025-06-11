import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

image_folder = 'utkcropped'

image_paths, ages = [], []
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpg.chip") or filename.endswith(".jpg.chip.jpg"):
        try:
            age = int(filename.split('_')[0])
            ages.append(age)
            image_paths.append(os.path.join(image_folder, filename))
        except ValueError:
            continue

X_train, X_test, y_train, y_test = train_test_split(image_paths, ages, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class AgeDataset(Dataset):
    def __init__(self, image_paths, ages, transform=None):
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.ages[index], dtype=torch.float32)

def get_dataloader():
    train_loader = DataLoader(AgeDataset(X_train, y_train, transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(AgeDataset(X_val, y_val, transform), batch_size=32)
    test_loader = DataLoader(AgeDataset(X_test, y_test, transform), batch_size=32)

    return train_loader, val_loader, test_loader