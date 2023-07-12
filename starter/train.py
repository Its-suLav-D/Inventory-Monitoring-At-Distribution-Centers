import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO)

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 5)

    def forward(self, x):
        return self.base_model(x)


def train(model, train_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Training Loss: {epoch_loss}')


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.info(f'Test accuracy: {accuracy}')
    return accuracy


def create_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CustomNet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    train(model, train_loader, criterion, optimizer, device, args.epochs)
    accuracy = test(model, test_loader, device)

    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    main(args)
