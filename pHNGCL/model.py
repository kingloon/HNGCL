from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random

from torch_geometric.nn import GCNConv
import gc

from torch_geometric.utils import num_nodes



class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class HNGCL(torch.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau=0.5, alpha=0.6):
        super(HNGCL, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden
        self.alpha = alpha

    def generate_hard_neg_samples(self, z: torch.Tensor, x: torch.Tensor, hard_sample_num: int):
        device = z.device
        num_nodes = x.size()[0]
        index = np.arange(num_nodes)
        random.shuffle(index)
        mask = torch.tensor(index[:hard_sample_num], dtype=torch.long).to(device)
        z = torch.unsqueeze(z, 1).to(device)
        source = torch.ones(z.size()[0], hard_sample_num, 1).to(device)
        source = torch.bmm(source, z)
        hard_neg_samples = self.alpha * source + (1 - self.alpha) * x[mask]
        hard_neg_samples = hard_neg_samples.to(z.device)
        return hard_neg_samples

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def batched_semi_loss_hard_neg(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            batch_sample = z1[mask]
            refl_sim = f(self.sim(batch_sample, z1))  # [B, N]
            between_sim = f(self.sim(batch_sample, z2))  # [B, N]
            hard_neg_samples = self.generate_hard_neg_samples(z1[mask], z2, between_sim.size()[0])  # [B, num_batch, d]
            batch_sample = F.normalize(batch_sample)
            batch_sample = torch.unsqueeze(batch_sample, 1)
            batch_sample = batch_sample.permute(0, 2, 1)
            hard_neg_samples = F.normalize(hard_neg_samples)
            hard_neg_sim = f(torch.bmm(hard_neg_samples, batch_sample))  # [B, num_batch]
            hard_neg_sim = hard_neg_sim.squeeze(-1)
            cur_loss = -torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (between_sim.sum(1) + hard_neg_sim.sum(1)))
            losses.append(cur_loss)
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
    
        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            if hasattr(torch.cuda, 'empty_cache'):
              gc.collect()
              torch.cuda.empty_cache()
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
    
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
    
        return ret

    def loss_neg(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size=64):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
    
        l1 = self.batched_semi_loss_hard_neg(h1, h2, batch_size)
        l2 = self.batched_semi_loss_hard_neg(h2, h1, batch_size)
    
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
    
        return ret

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
