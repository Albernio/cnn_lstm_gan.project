import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.init_size = img_shape[1] // 4  # 8 para 32x32

        # embedding más expresivo
        self.label_emb = nn.Embedding(num_classes, 50)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + 50, 256 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):

        gen_input = torch.cat((noise, self.label_emb(labels)), dim=1)

        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)

        img = self.conv_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, 50)

        def block(in_filters, out_filters, bn=True):
            layers = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            return layers

        self.conv_blocks = nn.Sequential(

            *block(img_shape[0], 64, bn=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
        )

        ds_size = img_shape[1] // 2**4

        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size + 50, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):

        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)

        d_in = torch.cat((out, self.label_emb(labels)), dim=1)

        validity = self.adv_layer(d_in)

        return validity
