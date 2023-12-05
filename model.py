import torch
import torch.nn as nn
import torch.nn.functional as F


class MNN(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, alpha, v):
        super(MNN, self).__init__()
        self.encode0 = nn.Linear(node_size, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size)
        self.alpha = alpha
        self.v = v

    def forward(self, adj_batch, c_mat, w_mat):
        encode = F.leaky_relu(self.encode0(adj_batch))
        encode = F.leaky_relu(self.encode1(encode))
        decode = F.leaky_relu(self.decode0(encode))
        decode = F.leaky_relu(self.decode1(decode))

        encode_norm = torch.sum(encode * encode, dim=1, keepdim=True)
        L_1st = self.alpha * torch.sum(
            w_mat * (encode_norm - 2 * torch.mm(encode, torch.transpose(encode, dim0=0, dim1=1))
                     + torch.transpose(encode_norm, dim0=0, dim1=1)))

        temp = (decode - adj_batch) * c_mat
        L_kth = torch.linalg.norm(temp, ord='nuc')

        L_reg = 0
        for param in self.parameters():
            L_reg += self.v * torch.sum(param * param)

        return L_1st + L_kth + L_reg

    def savector(self, adj):
        encode = F.leaky_relu(self.encode0(adj))
        encode = F.leaky_relu(self.encode1(encode))
        decode = F.leaky_relu(self.decode0(encode))
        decode = F.leaky_relu(self.decode1(decode))
        return encode, decode
