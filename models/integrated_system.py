
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys

from models.cnn_classifier import CNNClassifier
from models.lstm_generator import LSTMGenerator, TextProcessor
from models.conditional_gan import Generator


class MultimodalSystem:
    def __init__(self, checkpoints_dir=None, device=None):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Si no pasas ruta, busca 'checkpoints' en la raíz del proyecto
        if checkpoints_dir is None:
            # Esto obtiene la ruta de la carpeta del proyecto (un nivel arriba de 'models')
            base_path = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            checkpoints_dir = os.path.join(base_path, 'checkpoints')

        self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

        self.processor = TextProcessor(max_length=25)
        self._load_models(checkpoints_dir)

    def _load_models(self, path):
        # Cargar LSTM y su Vocabulario
        lstm_ckpt = torch.load(f'{path}/lstm_best.pth',
                               map_location=self.device)
        self.processor.word2idx = lstm_ckpt['vocab']
        self.processor.idx2word = lstm_ckpt['idx2word']

        self.lstm = LSTMGenerator(vocab_size=len(
            self.processor.word2idx)).to(self.device)
        self.lstm.load_state_dict(lstm_ckpt['model_state'])
        self.lstm.eval()

        # Cargar GAN (Generador)
        self.gan = Generator(latent_dim=100).to(self.device)
        self.gan.load_state_dict(torch.load(
            f'{path}/gan_generator.pth', map_location=self.device))
        self.gan.eval()

        # Cargar CNN (Clasificador para validación)
        self.cnn = CNNClassifier(num_classes=10).to(self.device)
        self.cnn.load_state_dict(torch.load(
            f'{path}/cnn_best.pth', map_location=self.device))
        self.cnn.eval()

    def generate_and_verify(self, category_name):
        """
        Genera una imagen, la describe y usa la CNN para ver si la GAN lo hizo bien.
        """
        if category_name not in self.categories:
            return "Categoría no válida", None, 0.0

        cat_idx = self.categories.index(category_name)
        label_tensor = torch.LongTensor([cat_idx]).to(self.device)

        with torch.no_grad():
            # A. Generar Imagen con GAN
            noise = torch.randn(1, 100).to(self.device)
            fake_img_tensor = self.gan(noise, label_tensor)

            # B. Validar con CNN (¿Qué opina el clasificador?)
            # Re-normalizamos para la CNN si es necesario
            cnn_output = self.cnn(fake_img_tensor)
            probs = F.softmax(cnn_output, dim=1)
            confidencia = probs[0][cat_idx].item() * 100

            # C. Generar Texto con LSTM
            description = self._generate_text(label_tensor)

        # Convertir tensor a imagen PIL
        img_np = (fake_img_tensor.squeeze().cpu().permute(
            1, 2, 0).numpy() * 0.5 + 0.5)
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        return description, img_pil, confidencia

    def _generate_text(self, label_tensor):
        generated = []
        input_tokens = torch.tensor(
            [[self.processor.word2idx["<SOS>"]]]).to(self.device)

        for _ in range(20):
            outputs = self.lstm(input_tokens, label_tensor)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1).item()
            if next_token == self.processor.word2idx["<EOS>"]:
                break

            word = self.processor.idx2word.get(next_token, "")
            if word not in ["<PAD>", "<SOS>"]:
                generated.append(word)

            input_tokens = torch.cat(
                (input_tokens, torch.tensor([[next_token]]).to(self.device)), dim=1)

        return " ".join(generated).capitalize() + "."


if __name__ == "__main__":
    # Prueba rápida del sistema
    system = MultimodalSystem()
    desc, img, score = system.generate_and_verify("cat")
    print(f"Descripción: {desc}")
    print(f"Confidencia de la CNN: {score:.2f}%")
    img.show()
