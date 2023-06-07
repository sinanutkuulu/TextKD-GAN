import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512):
        super(Autoencoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, 1)
        self.decoder = nn.LSTM(hidden_dim, input_dim, 1)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)

        # Repeat the hidden state for each time step in the sequence
        repeat_hidden = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        # Decoder
        outputs, _ = self.decoder(repeat_hidden)

        outputs = nn.Softmax(outputs)

        return outputs