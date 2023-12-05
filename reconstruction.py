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
    parser.add_argument('--epochs', default=100, type=int, help="the number of training epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="the learning rate of optimizer")
    parser.add_argument('--beta', default=1e-2, type=float, help="the decay parameter of Katz index")
    parser.add_argument('--gamma', default=2., type=float, help="the weight of non-zero elements")
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
    _, Adj, Node = dataset.Read_graph(args.input)
    Data = dataset.Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True)

    # model init
    model = model.MNN(Node, args.nhid0, args.nhid1, args.alpha, args.v)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # model training
    encode, decode = utils.train(model, Data, Adj, opt, args)

    # model evaluation
    # Arixv GR-QC
    eval.get_Pk(decode.cpu().detach().numpy(), Adj.cpu().detach().numpy(),
                [2, 10, 50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000])
    # Blogcatalog
    # get_Pk(decode.cpu().detach().numpy(), Adj.cpu().detach().numpy(),
    #          [2, 100, 1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000])

    eval.get_MAP(decode.cpu().detach().numpy(), Adj.cpu().detach().numpy())
