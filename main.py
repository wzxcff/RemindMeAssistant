import nltk
import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits


class Assistant:
    def __init__(self, intents_path):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.label_vocab = {"O": 0}
        self.intents_path = intents_path

    def tokenize_and_lemmatize(self, text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"File not found: {self.intents_path}")

        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f)

        dataset = []

        for intent in intents_data["intents"]:
            tokens = self.tokenize_and_lemmatize(intent["pattern"][0])
            labels = ["O"] * len(tokens)

            # Assign entity labels
            for ent in intent["entities"]:
                ent_word = ent["word"]
                ent_label = ent["label"]

                for i, token in enumerate(tokens):
                    if token.lower() == ent_word.lower():
                        labels[i] = ent_label

            # Add tokens to vocab
            for token in tokens:
                if token.lower() not in self.vocab:
                    self.vocab[token.lower()] = len(self.vocab)

            # Add labels to label_vocab
            for label in labels:
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)

            dataset.append({"tokens": tokens, "labels": labels})

        return dataset

    def encode_dataset(self, dataset):
        X = []
        y = []

        for item in dataset:
            token_ids = [self.vocab.get(tok.lower(), self.vocab["<UNK>"]) for tok in item["tokens"]]
            label_ids = [self.label_vocab[lbl] for lbl in item["labels"]]

            X.append(token_ids)
            y.append(label_ids)

        return X, y

    def pad_sequences(self, sequences, pad_value):
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded_seqs)

    def train(self, embedding_dim=64, hidden_dim=64, epochs=20, lr=0.001):
        dataset = self.parse_intents()
        X, y = self.encode_dataset(dataset)
        X_padded = self.pad_sequences(X, self.vocab["<PAD>"])
        y_padded = self.pad_sequences(y, self.label_vocab["O"])

        self.model = Model(len(self.vocab), embedding_dim, hidden_dim, len(self.label_vocab))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.label_vocab["O"])

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_padded)
            loss = criterion(outputs.view(-1, len(self.label_vocab)), y_padded.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, text):
        tokens = self.tokenize_and_lemmatize(text)
        token_ids = [self.vocab.get(tok.lower(), self.vocab["<UNK>"]) for tok in tokens]
        X = self.pad_sequences([token_ids], self.vocab["<PAD>"])
        with torch.no_grad():
            logits = self.model(X)
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        pred_labels = [list(self.label_vocab.keys())[list(self.label_vocab.values()).index(i)] for i in pred_ids[:len(tokens)]]
        return list(zip(tokens, pred_labels))


assistant = Assistant("dataset.json")
assistant.train(epochs=100)
print(assistant.predict("Remind me about meeting today at 18:00"))

