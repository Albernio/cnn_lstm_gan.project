from models.cnn_classifier import CNNClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys

# Añadimos el path base para importar el modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train():
    # 1. Configuraciones básicas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    batch_size = 64
    learning_rate = 0.001

    print(f"Entrenando en: {device}")

    # 2. Transformaciones (Data Augmentation para robustez)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 3. Datasets y Loaders
    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Inicializar Modelo, Loss y Optimizer
    model = CNNClassifier(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    # Usamos Adam para una convergencia más rápida en VGG
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 5. Bucle de Entrenamiento
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        # Validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy en Test tras Epoch {epoch+1}: {accuracy:.2f}%')
        scheduler.step()

        # Guardar el mejor modelo
        if accuracy > best_acc:
            best_acc = accuracy
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/cnn_best.pth')
            print("--- Mejor modelo guardado ---")


if __name__ == "__main__":
    train()
