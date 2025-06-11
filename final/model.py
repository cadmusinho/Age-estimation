import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


class AgeEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

def train(train_loader, val_loader, test_loader):
    model = AgeEstimationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    train_losses, val_losses, mae_scores = [], [], []

    for epoch in range(100):
        model.train()
        total_loss = 0
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss, preds, actuals = 0, [], []
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, ages)
                val_loss += loss.item() * images.size(0)
                preds.extend(outputs.cpu().numpy())
                actuals.extend(ages.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        mae = mean_absolute_error(actuals, preds)
        mae_scores.append(mae)

        print(f"Epoch {epoch + 1}/100 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'age_estimation_model.pth')
            print("Zapisano nowy najlepszy model!")

    model.load_state_dict(torch.load('age_estimation_model.pth'))
    model.eval()
    test_preds, test_ages = [], []

    with torch.no_grad():
        for images, ages in test_loader:
            images, ages = images.to(device), ages.to(device).view(-1, 1)
            outputs = model(images)
            test_preds.extend(outputs.cpu().numpy())
            test_ages.extend(ages.cpu().numpy())

    mae = mean_absolute_error(test_ages, test_preds)
    rmse = np.sqrt(mean_squared_error(test_ages, test_preds))
    r2 = r2_score(test_ages, test_preds)

    print(f"\n Test MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

    return train_losses, val_losses, mae_scores

def show_plot(train_losses, val_losses, mae_scores):
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(mae_scores, label="MAE")
    plt.title("MAE per epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
