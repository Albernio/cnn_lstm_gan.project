import torch
import torch.nn as nn
import json
import re
from collections import Counter


class TextProcessor:
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def build_vocab(self, descriptions_dict):
        all_words = []
        for category in descriptions_dict:
            for desc in descriptions_dict[category]:
                words = self.clean_text(desc).split()
                all_words.extend(words)

        # Solo palabras que aparecen al menos 1 vez
        counts = Counter(all_words)
        for word, count in counts.items():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        print(f"Vocabulario construido: {self.vocab_size} palabras únicas.")

    def tokenize(self, text):
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"])
                  for w in self.clean_text(text).split()]
        # Añadir Start of Sentence y End of Sentence
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
        # Padding o Truncado
        if len(tokens) < self.max_length:
            tokens += [self.word2idx["<PAD>"]] * \
                (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens)


class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=10):
        super(LSTMGenerator, self).__init__()

        self.hidden_dim = hidden_dim

        # Capa de Embedding: convierte índices en vectores densos
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capa para procesar la categoría (label) e inyectarla como contexto
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        # La LSTM recibe el embedding de la palabra + el contexto de la categoría
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.2)

        # Capa de salida: predice la siguiente palabra en el vocabulario
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text_seq, labels):
        # text_seq: [batch, seq_len]
        # labels: [batch]

        batch_size = text_seq.size(0)
        seq_len = text_seq.size(1)

        # Embeddings de palabras
        word_embeds = self.embedding(text_seq)  # [batch, seq_len, emb_dim]

        # Embedding de la categoría (expandido para cada paso de tiempo)
        class_embeds = self.label_emb(labels).unsqueeze(
            1).repeat(1, seq_len, 1)  # [batch, seq_len, emb_dim]

        # Concatenamos palabra + categoría
        # [batch, seq_len, emb_dim * 2]
        combined = torch.cat((word_embeds, class_embeds), dim=2)

        lstm_out, _ = self.lstm(combined)
        logits = self.fc(lstm_out)

        return logits
