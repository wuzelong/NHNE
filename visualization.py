import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from data import dataset
import numpy as np
import model
import utils
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--model', default="NHNE", choices=["NHNE", "NNE"], help="NHNE or NNE model")
    parser.add_argument('--epochs', default=100, type=int, help="the number of training epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="the learning rate of optimizer")
    parser.add_argument('--beta', default=1e-2, type=float, help="the decay parameter of Katz index")
    parser.add_argument('--gamma', default=2.9, type=float, help="the weight of non-zero elements")
    parser.add_argument('--bs', default=128, type=int, help="the batch size during training")
    parser.add_argument('--nhid0', default=200, type=int, help="the size of first hidden layer")
    parser.add_argument('--nhid1', default=100, type=int, help="the size of second hidden layer")
    parser.add_argument('--alpha', default=1e-4, type=float, help="the weight of L_1st loss")
    parser.add_argument('--v', default=1e-1, type=float, help="the weight of L_reg loss")
    parser.add_argument('--threshold', default=0.05, type=float,
                        help="the threshold for converting cosine similarity matrix into adjacency matrix")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # init
    utils.setup_seed(20230317)
    args = parse_args()

    # data processing
    categories = ['comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                    remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(newsgroups.data)
    cosine_similarities = cosine_similarity(X)
    Adj = cosine_similarities.copy()
    np.fill_diagonal(Adj, 1)
    adj = torch.tensor(Adj, dtype=torch.float)
    adj = torch.where(adj > args.threshold, torch.tensor(1.), torch.tensor(0.))
    Node = len(adj)
    Data = dataset.Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=False)

    # model init
    model = model.MNN(Node, args.nhid0, args.nhid1, args.alpha, args.v)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # model training
    encode, decode = utils.train(model, Data, adj, opt, args)

    # model evaluation
    encode = encode.cpu().detach().numpy()
    tsne = TSNE(n_components=2, metric='cosine', perplexity=30, random_state=20230317)
    X_tsne = tsne.fit_transform(encode)
    print("KL divergence:", round(tsne.kl_divergence_, 4))

    target_names = newsgroups.target_names
    target_colors = ['r', 'g', 'b']
    target_labels = [target_names[i] for i in newsgroups.target]
    for target, color in zip(range(len(target_names)), target_colors):
        indices = [i for i, label in enumerate(target_labels) if label == target_names[target]]
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=color, label=target_names[target],
                    s=3)
    plt.axis('off')
    plt.savefig('20News_visualization.pdf')
    plt.show()
