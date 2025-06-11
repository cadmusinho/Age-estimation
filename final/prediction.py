import torch
from PIL import Image
from torchvision import transforms
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_path = 'age_estimation_model.pth'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class AgeEstimationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(256 * 8 * 8, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

model = AgeEstimationModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_age(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image).item()
    return prediction

if __name__ == '__main__':
    sample_image_path = "D:/test/cubarsi.jfif"
    if os.path.exists(sample_image_path):
        predicted_age = predict_age(sample_image_path)
        print(f"Przewidywany wiek: {predicted_age:.1f}")
    else:
        print(f"Błąd: Ścieżka do pliku {sample_image_path} nie istnieje.")
