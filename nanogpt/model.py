import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(1024, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)

        x = self.embed(x) + self.pos_embed(positions)
        x = self.transformer(x)
        return self.fc(x)
