from models.lstm_generator import LSTMGenerator, TextProcessor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

# Ajuste de path para importar módulos locales
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


class DescriptionDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.processor = processor
        self.categories = list(self.data.keys())
        self.samples = []

        # Mapeo de nombre de categoría a índice (0-9)
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        for cat, descs in self.data.items():
            for d in descs:
                self.samples.append((self.cat_to_idx[cat], d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat_idx, desc = self.samples[idx]
        tokens = self.processor.tokenize(desc)
        return tokens, torch.tensor(cat_idx)


def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100  # Las LSTMs con pocos datos necesitan más épocas para converger
    batch_size = 32

    # 1. Preparar procesador y datos
    processor = TextProcessor(max_length=25)
    with open('data/descriptions.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    processor.build_vocab(raw_data)

    dataset = DescriptionDataset('data/descriptions.json', processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Inicializar modelo
    model = LSTMGenerator(vocab_size=processor.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignoramos el <PAD>
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Iniciando entrenamiento de LSTM en {device}...")

    # 3. Bucle de entrenamiento
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            # El input es la secuencia hasta la penúltima palabra
            # El target es la secuencia desde la segunda palabra
            optimizer.zero_grad()
            outputs = model(texts[:, :-1], labels)

            loss = criterion(
                outputs.reshape(-1, processor.vocab_size), texts[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

    # 4. Guardar checkpoint y vocabulario
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'vocab': processor.word2idx,
        'idx2word': processor.idx2word
    }, 'checkpoints/lstm_best.pth')
    print("Modelo LSTM y vocabulario guardados.")


if __name__ == "__main__":
    train_lstm()
