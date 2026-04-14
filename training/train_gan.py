from models.conditional_gan import Generator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def train_gan():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.0002
    beta1 = 0.5
    latent_dim = 100
    epochs = 500
    batch_size = 128

    # Discriminator steps por iteración
    n_critic = 2

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(beta1, 0.999))

    # Fixed noise para evaluar progreso
    fixed_noise = torch.randn(25, latent_dim, device=device)
    fixed_labels = torch.arange(0, 10).repeat(3)[:25].to(device)

    print(f"Entrenando en {device}")

    for epoch in range(epochs):

        for i, (imgs, labels) in enumerate(dataloader):

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            batch_size_i = imgs.size(0)

            # Label smoothing
            valid = torch.empty(
                batch_size_i, 1, device=device).uniform_(0.8, 1.0)
            fake = torch.zeros(batch_size_i, 1, device=device)

            # ============================================================
            # ENTRENAR DISCRIMINADOR
            # ============================================================

            for _ in range(n_critic):

                optimizer_D.zero_grad(set_to_none=True)

                z = torch.randn(batch_size_i, latent_dim, device=device)

                gen_labels = torch.randint(
                    0, 10, (batch_size_i,), device=device)

                with torch.no_grad():
                    fake_imgs = generator(z, gen_labels)

                real_pred = discriminator(imgs, labels)
                fake_pred = discriminator(fake_imgs, gen_labels)

                real_loss = adversarial_loss(real_pred, valid)
                fake_loss = adversarial_loss(fake_pred, fake)

                d_loss = (real_loss + fake_loss) * 0.5

                d_loss.backward()
                optimizer_D.step()

            # ============================================================
            # ENTRENAR GENERADOR
            # ============================================================

            optimizer_G.zero_grad(set_to_none=True)

            z = torch.randn(batch_size_i, latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (batch_size_i,), device=device)

            gen_imgs = generator(z, gen_labels)

            validity = discriminator(gen_imgs, gen_labels)

            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

        # ============================================================
        # LOG + VISUALIZACIÓN
        # ============================================================

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"[D loss: {d_loss.item():.4f}] "
            f"[G loss: {g_loss.item():.4f}]"
        )

        if epoch % 10 == 0:

            os.makedirs('outputs', exist_ok=True)

            with torch.no_grad():
                sample_imgs = generator(fixed_noise, fixed_labels)

            utils.save_image(
                sample_imgs,
                f"outputs/epoch_{epoch}.png",
                nrow=5,
                normalize=True
            )

    os.makedirs('checkpoints', exist_ok=True)

    torch.save(generator.state_dict(), "checkpoints/gan_generator.pth")
    torch.save(discriminator.state_dict(), "checkpoints/gan_discriminator.pth")

    print("Entrenamiento finalizado")


if __name__ == "__main__":
    train_gan()
