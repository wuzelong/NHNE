import numpy as np
import torch


def getKatz(adj, beta):
    # (I-beta*A)^-1 -I
    n = len(adj)
    I = torch.eye(n)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    I = I.to(device)
    adj = adj.to(device)
    A = beta * adj
    IA = I - A
    if torch.det(IA) == 0:
        s_matrix = torch.pinverse(IA) - I
    else:
        s_matrix = torch.inverse(IA) - I
    return s_matrix


def train(model, Data, Adj, opt, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Adj = Adj.to(device)
    model.train()
    model_opt = model
    loss_opt = -1
    if args.model == "NHNE":
        S = getKatz(Adj, args.beta)
        W = torch.softmax(S, dim=1)
    for epoch in range(1, args.epochs + 1):
        loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
        for index in Data:
            adj_batch = Adj[index]
            if args.model == "NHNE":
                w_batch = W[index]
                w_mat = w_batch[:, index]
            else:
                w_mat = adj_batch[:, index]
            c_mat = torch.ones_like(adj_batch)
            c_mat[adj_batch != 0] = args.gamma
            opt.zero_grad()
            Loss = model(adj_batch, c_mat, w_mat)
            Loss.backward()
            opt.step()
            loss_sum += Loss
        if loss_opt == -1 or loss_opt > loss_sum:
            loss_opt = loss_sum
            model_opt = model

    model_opt.eval()
    model_opt = model_opt.to(device)
    encode, decode = model_opt.savector(Adj)
    return encode, decode


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
