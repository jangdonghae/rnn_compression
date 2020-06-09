"""Module using IndRNNCell to solve the sequential MNIST task.
The hyper-parameters are taken from that paper as well.

"""
from indrnn import IndRNN
from indrnn import IndRNNv2
from myrnn import myRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from time import time

parser = argparse.ArgumentParser(description='PyTorch IndRNN sequential MNIST test')
# Default parameters taken from https://arxiv.org/abs/1803.04831
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--n-layer', type=int, default=6,
                    help='number of layer of IndRNN (default: 6)')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='number of hidden units in one IndRNN layer(default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disable frame-wise batch normalization after each layer')
parser.add_argument('--log_epoch', type=int, default=1,
                    help='after how many epochs to report performance')
parser.add_argument('--log_iteration', type=int, default=-1,
                    help='after how many iterations to report performance, deactivates with -1 (default: -1)')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='enable bidirectional processing')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--max-steps', type=int, default=2000,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--model', type=str, default="IndRNN",
                    help='if either IndRNN or LSTM cells should be used for optimization')
parser.add_argument('--layer_sizes', type=int, nargs='+', default=None,
                    help='list of layers')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 24  # 28x28 pixels
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)


cuda = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size=None, n_layer=2, layer_sizes=None, model=IndRNN):
        super(Net, self).__init__()
        recurrent_inits = []
        
        if layer_sizes is None:
            for _ in range(n_layer - 1):
                recurrent_inits.append(
                    lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX)
                )
            recurrent_inits.append(lambda w: nn.init.uniform_(
                w, RECURRENT_MIN, RECURRENT_MAX))
            self.rnn = model(
                input_size, hidden_size, n_layer, batch_norm=args.batch_norm,
                hidden_max_abs=RECURRENT_MAX, batch_first=True,
                bidirectional=args.bidirectional, recurrent_inits=recurrent_inits,
                gradient_clip=5
            )
            self.lin = nn.Linear(
                hidden_size * 2 if args.bidirectional else hidden_size, 10)

            self.lin.bias.data.fill_(.1)
            self.lin.weight.data.normal_(0, .01)
            
        else:
            n_layer = len(layer_sizes) + 1
            for _ in range(n_layer - 1):
                recurrent_inits.append(
                    lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX)
                )
            recurrent_inits.append(lambda w: nn.init.uniform_(
                w, RECURRENT_MIN, RECURRENT_MAX))
            
            self.rnn = model(
                input_size, hidden_layer_sizes = layer_sizes,
                batch_first=True, recurrent_inits=recurrent_inits
            )
            self.lin = nn.Linear(layer_sizes[-1], 10)

            self.lin.bias.data.fill_(.1)
            self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y, _ = self.rnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)


def main():
    # build model
    if args.model.lower() == "indrnn":
        model = Net(9, hidden_size = args.hidden_size, n_layer = args.n_layer)
    elif args.model.lower() == "indrnnv2":
        model = Net(9, hidden_size = args.hidden_size, n_layer = args.n_layer, model = IndRNNv2)
    elif args.model.lower() == "myrnn":
        model = Net(9, layer_sizes = args.layer_sizes, model = myRNN)
    else:
        raise Exception("unsupported cell model")

    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load data
    train_data, test_data = UCI_dataloader(args.batch_size)

    # Train the model
    model.train()
    step = 0
    epochs = 0
    while step < args.max_steps:
        losses = []
        start = time()
        for data, target in train_data:
            if cuda:
                data, target = data.cuda(), target.cuda()
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().item())
            step += 1

            if step % args.log_iteration == 0 and args.log_iteration != -1:
                print(
                    "\tStep {} cross_entropy {}".format(
                        step, np.mean(losses)))
            if step >= args.max_steps:
                break
        if epochs % args.log_epoch == 0:
            print(
                "Epoch {} cross_entropy {} ({} sec.)".format(
                    epochs, np.mean(losses), time()-start))
        epochs += 1

    # get test error
    model.eval()
    correct = 0
    for data, target in test_data:
        if cuda:
            data, target = data.cuda(), target.cuda()
        out = model(data)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return np.squeeze(y_ - 1)


class UCIDataset(Dataset):
    def __init__(self, data_path, mode):
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
        import numpy as np
        x_path = [data_path + mode + '/' + "Inertial Signals/" + signal + mode + '.txt' for signal in
                  INPUT_SIGNAL_TYPES]
        y_path = data_path + mode + '/' + 'y_' + mode + '.txt'
        self.X = load_X(x_path)
        self.y = load_y(y_path)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

def UCI_dataloader(batch_size, dataset_folder='./UCI HAR Dataset/'):

    dataset_train = UCIDataset(dataset_folder, 'train')
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=64,
                              shuffle=True, drop_last=True
                              )

    dataset_test = UCIDataset(dataset_folder, 'test')
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=64,
                             shuffle=False
                             )

    return (train_loader, test_loader)



if __name__ == "__main__":
    main()
