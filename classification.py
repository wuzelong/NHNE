import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data.dataloader import DataLoader

import eval
from data import dataset
import model
import utils
from sklearn.preprocessing import MultiLabelBinarizer


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--model', default="NHNE", choices=["NHNE", "NNE"], help="NHNE or NNE model")
    parser.add_argument('--input', default='./data/Blog.edge', help="the path of input data (edge list)")
    parser.add_argument('--label', default='./data/Blog.groups', help="the labels of node")
    parser.add_argument('--epochs', default=100, type=int, help="the number of training epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="the learning rate of optimizer")
    parser.add_argument('--beta', default=1e-2, type=float, help="the decay parameter of Katz index")
    parser.add_argument('--gamma', default=3., type=float, help="the weight of non-zero elements")
    parser.add_argument('--bs', default=128, type=int, help="the batch size during training")
    parser.add_argument('--nhid0', default=1000, type=int, help="the size of first hidden layer")
    parser.add_argument('--nhid1', default=100, type=int, help="the size of second hidden layer")
    parser.add_argument('--alpha', default=3e-1, type=float, help="the weight of L_1st loss")
    parser.add_argument('--v', default=1e-1, type=float, help="the weight of L_reg loss")
    parser.add_argument('--seed', default=20230317, type=int, help="random seed for different classifications")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # ! pip install -U liblinear-official
    # ! pip install iterative-stratification

    # init
    utils.setup_seed(20230317)
    args = parse_args()

    # data processing
    _, Adj, Node = dataset.Read_graph(args.input)
    Data = dataset.Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True)

    # label processing (one-hot)
    with open(args.label, 'r') as f:
        lines = f.readlines()
    y = []
    for line in lines:
        labels = line.strip().split(' : ')[1]
        y.append([int(label) for label in labels.split()])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    # model init
    model = model.MNN(Node, args.nhid0, args.nhid1, args.alpha, args.v)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # model training
    encode, decode = utils.train(model, Data, Adj, opt, args)
    x = encode.cpu().detach().numpy()

    # model evaluation
    # ten fold cross validation
    Mi_F1, Ma_F1 = eval.ten_fold_cross_validation(x, y, args.seed)
    print(Mi_F1, Ma_F1)

    # F1 with train radio from 10% to 90%
    # (Execute five times using different seeds to obtain the mean)
    # F1 = eval.F1_with_different_train_radio(x, y, args.seed)
    # for i, row in enumerate(F1):
    #     print("training radio:", i)
    #     print("Mi_F1 & Ma_F1:", row)
