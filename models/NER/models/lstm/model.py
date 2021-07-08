import torch
import torch.nn as nn
import os


class LSTMFixedLen(nn.Module):
    def __init__(self, vocab_size=33, embedding_dim=16, hidden_dim=128, max_len=14):
        super().__init__()

        self.max_len = max_len

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)

        self.fc_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embeddings(x)

        # x = self.dropout(x)

        lstm_out, (ht, ct) = self.lstm(x)
        last_ht = ht[-1]

        y = self.fc_block(last_ht)

        y = self.fc_block2(y)

        y = self.fc(y)

        return y

    def encode(self, word: str) -> torch.Tensor:
        """Encode token with russian dictionary

        Parameters:
            word: word to encode

        Returns:
            Encoded with russian dictionary word

        """
        cur_word = []
        a = ord('а')
        chars = [chr(i) for i in range(a, a + 32)]
        nums = [i + 1 for i in range(0, 33)]
        rus_dict = dict(zip(chars, nums))
        for i in word.lower():
            cur_word.append(rus_dict[i])
        while len(cur_word) < self.max_len:
            cur_word.append(1)
        return torch.tensor(cur_word)

    def prediction(self, word: str) -> torch.Tensor:
        encoded = self.encode(word)
        encoded = torch.reshape(encoded, (1, 14))
        probability = self.forward(encoded)
        return probability

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(model_path: str):
        path = os.path.abspath(model_path)
        model = LSTMFixedLen()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
