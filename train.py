import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import argparse

from utils import Experiment, update_lr
from model import CNN
from loss import nll_loss
from data_loader import get_wiki_data_loader


##################
# Todo: 
# - (done) pass the network parameters as arguments
# - (done) reorganize the stuff, loss fn and gaussian in dif files
# - (done) write function descriptions/parameters
# - (done) save the model each epoch/n-steps
# - resume experiment
# - optimizer as a parameter?
# - fix the learning rate thing from lluis' code
# - (done)lr decay
##################


def main(args):
    num_epochs = args.ne
    batch_size = args.bs

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    dataset_path = args.dataset_path
    json_labels_path = args.json_labels_path

    data_loader = get_wiki_data_loader(dataset_path, json_labels_path, 
                                       transform, batch_size, shuffle=True,
                                       num_workers=args.n_workers)

    if args.out_dim == -1:
        out_dim = None
    else:
        out_dim = args.out_dim
    if args.ttn:
        cnn = CNN(args.n_topics, args.n_kernels, mixture_model=False, args.cnn)
    else:
        cnn = CNN(args.n_topics, args.n_kernels, out_dim=out_dim, args.cnn)

    if torch.cuda.is_available():
        cnn.cuda()
    cnn.train()

    optimizer = optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.mm)
    # optimizer = optim.Adam(cnn.parameters())
    # optimizer = optim.RMSprop(cnn.parameters(), lr= args.lr, momentum=args.mm)

    exp = Experiment(args, args.exp_name)

    if args.ttn:
        loss_fn = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fn = nll_loss

    learning_rate = args.lr
    losses = []

    for epoch in range(num_epochs):
        if (epoch%args.decay_step == 0 and epoch > 0):  # move this to the utils file
            learning_rate = learning_rate * args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        for step, (images, ts) in enumerate(data_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                ts = ts.cuda()
            cnn.zero_grad()

            if not args.ttn:
                alpha, sigma, mu = cnn(images)
                loss = loss_fn(alpha, sigma, mu, ts)
            else:
                out = cnn.alexnet(images)
                loss = loss_fn(out, ts)
            loss.backward()
            optimizer.step()

            losses.append(float(loss))
            exp.save_loss(epoch, step, loss)
            print('Epoch ' + str(epoch+1) + '/' + str(num_epochs) + ' - Step ' + str(step+1) + '/' +
                  str(len(data_loader)) + ' - Loss: ' + str(float(loss)))
        exp.save_loss_epoch(epoch, losses)
        losses = []
        if (epoch%args.save_step == 0 and epoch > 0):
            exp.save_model(epoch, cnn)
    
    exp.save_model('last', cnn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_topics', type=int, default=40,
                        help='Number of topics of the LDA space')
    parser.add_argument('--n_kernels', type=int, default=10,
                        help='Number of Gaussian kernels used to compute the likelihood')
    parser.add_argument('--dataset_path', type=str, default='./../datasets/ImageCLEF_wikipedia/',
                        help='Path to the wikipedia dataset')
    parser.add_argument('--json_labels_path', type=str, default='LDA/training_labels40_original.json',
                        help='Path to the labels of the wikipedia dataset')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of subprocesses used for data loading')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate, might be dependent on the optimizer')
    parser.add_argument('-mm', type=float, default=0.9,
                        help='Momentum, might be dependent on the optimizer')
    parser.add_argument('--out_dim', type=int, default=-1,
                        help='Number size of the output vector of the CNN')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='Decay of the learning rate')
    parser.add_argument('--decay_step', type=int, default=20,
                        help='')
    parser.add_argument('--save_step', type=int, default=20,
                        help='')
    parser.add_argument('-ne', type=int, default=100, help='Number of epochs')
    parser.add_argument('-bs', type=int, default=1024, help='Size of the batch')
    parser.add_argument('-ttn', action='store_true', help='If true, replicates the TextTopicNet \
                        architecture')
    parser.add_argument('-cnn', type=str, help='Name of the cnn used to extract the features, can \
                        be alexnet or resnet', default="alexnet")
    args = parser.parse_args()

    main(args)

