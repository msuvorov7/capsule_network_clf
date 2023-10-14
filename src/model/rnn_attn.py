import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self, num_input: int):
        super().__init__()
        self.linear = nn.Linear(num_input, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        weights = self.softmax(self.linear(x))
        attn = x.transpose(1, 2).bmm(weights).squeeze()
        if batch_size == 1:
            attn = attn.unsqueeze(0)
        return attn, weights


class RNNAttention(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=True,
                          )

        self.attn = DotProductAttention(hidden_dim * 2)

        self.out_fc = nn.Linear(2 * hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)

        # x = [batch_size, sent_len, emb_dim]
        packed_output, _ = self.rnn(embedded)
        # packed_output = [batch_size, seq_len, 2 * hid_dim]
        attn, weights = self.attn(packed_output)
        # attn = [batch_size, 2 * hid_dim]

        attn_cat_fc = F.leaky_relu(self.dropout(attn))

        return self.out_fc(attn_cat_fc)
