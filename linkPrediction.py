import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data.dataloader import DataLoader
from data import dataset
import model
import utils
import eval


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--model', default="NHNE", choices=["NHNE", "NNE"], help="NHNE or NNE model")
    parser.add_argument('--input', default='./data/GrQc.edge', help="the path of input data (edge list)")
    parser.add_argument('--input_observe', default='./data/GrQc1.5.observe', help="the path of observe data (edge list), hidden 15% to 80% for training")
    parser.add_argument('--epochs', default=100, type=int, help="the number of training epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="the learning rate of optimizer")
    parser.add_argument('--beta', default=1e-3, type=float, help="the decay parameter of Katz index")
    parser.add_argument('--gamma', default=7., type=float, help="the weight of non-zero elements")
    parser.add_argument('--bs', default=128, type=int, help="the batch size during training")
    parser.add_argument('--nhid0', default=500, type=int, help="the size of first hidden layer")
    parser.add_argument('--nhid1', default=100, type=int, help="the size of second hidden layer")
    parser.add_argument('--alpha', default=1e-4, type=float, help="the weight of L_1st loss")
    parser.add_argument('--v', default=1e-4, type=float, help="the weight of L_reg loss")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # init
    utils.setup_seed(20230317)
    args = parse_args()

    # data processing
    _, Adj, Node = dataset.Read_graph(args.input_observe)
    Data = dataset.Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True)

    # model init
    model = model.MNN(Node, args.nhid0, args.nhid1, args.alpha, args.v)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # model training
    encode, decode = utils.train(model, Data, Adj, opt, args)

    # model evaluation
    _, Adj2, _ = dataset.Read_graph(args.input)
    Ek, Eq = eval.get_EkEq(decode.cpu().detach().numpy(), Adj.cpu().detach().numpy(), Adj2.cpu().detach().numpy())
    eval.get_Pk(Ek, Eq, [2, 10, 100, 200, 300, 500, 800, 1000, 1500, 2000, 10000])
    eval.get_MAP(Ek, Eq)
