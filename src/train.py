import torch
import torchvision

import logging
logging.basicConfig(level=logging.INFO)

from models import build_model
from utils import seed_everything

def train(model, train_loader, val_loader, optimizer, criterion, epochs, patience, model_path, seed):

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    seed_everything(seed)

    model.eval()
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_losses.append(running_train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

            val_losses.append(running_val_loss / len(val_loader))
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                torch.save(model.state_dict(), model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info(f'Early stopping at epoch {epoch+1}')
                break
        logging.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses
