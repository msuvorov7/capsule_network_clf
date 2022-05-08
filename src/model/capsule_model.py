import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1.0 + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=6):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=1,
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=0,
                      ) for _ in range(num_capsules)
        ])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 4 * 3, -1)
        return squash(u)


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 4 * 3, in_channels=338, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x, use_cuda=False):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = torch.autograd.Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if use_cuda:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)


class CapsNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim: int, output_dim: int):
        super(CapsNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layer = ConvLayer(embedding_dim)
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.linear = nn.Linear(10 * 16 * 1, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(x)))
        return self.linear(output.view(output.size(0), -1))
