from model import train, show_plot
from dataset import get_dataloader

def main():
    train_loader, val_loader, test_loader = get_dataloader()

    train_losses, val_losses, mae_scores = train(train_loader, val_loader, test_loader)
    show_plot(train_losses, val_losses, mae_scores)


if __name__ == '__main__':
    main()