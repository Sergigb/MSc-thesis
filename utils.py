import os

import torch
import numpy as np


# try using https://pytorch.org/docs/stable/distributions.html
def gaussian(t, mu, sigma):
    return (1/torch.sqrt(2*np.pi*sigma)) * torch.exp((-1/(2*sigma)) * torch.norm((t-mu), 2, 1)**2)


def update_lr(lr, decay, epoch, optimizer):
    """
    :param lr: learning rate
    :param decay: decay
    :param epoch: current epoch
    :param optimizer: model optimizer
    """
    lr_ = lr*(1/(1+decay*epoch))
    print(lr_)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


class Experiment:
    def __init__(self, args, exp_name):
        """
        :param args: arguments of the main file
        :param exp_name: Name of the experiment
        """

        if not os.path.exists(os.path.join('exps', args.exp_name)):
            os.mkdir(os.path.join('exps', args.exp_name))
        if not os.path.exists(os.path.join('exps', args.exp_name, 'models')):
            os.mkdir(os.path.join('exps', args.exp_name, 'models')) 

        self.exp_path = os.path.join('exps', args.exp_name)
        self.models_path = os.path.join('exps', args.exp_name, 'models')

        with open(os.path.join('exps', args.exp_name, 'parameters.txt'), 'w+') as f:
            f.write('Name of the experiment: ' + args.exp_name + '\n')
            f.write('Number of topics: ' + str(args.n_topics) + '\n')
            if args.ttn:
                f.write('Number of kernels: n/a\n')
            else:
                f.write('Number of kernels: ' + str(args.n_kernels) + '\n')
            f.write('Learning rate: ' + str(args.lr) + '\n')
            f.write('Momentum: ' + str(args.mm) + '\n')
            f.write('Optimizer: ' + 'SGD' + '\n')
            if args.ttn:
                f.write('Out_dim: default\n')
            else:
                f.write('Out_dim: ' + str(args.out_dim) + '\n')
            f.write('Decay: ' + str(args.decay) + '\n')
            f.write('Decay step: ' + str(args.decay_step) + '\n')
            f.write('Batch size: ' + str(args.bs) + '\n')
            f.write('Labels path: ' + str(args.json_labels_path) + '\n')
            f.write('TextTopicNetwork architecture: ' + str(args.ttn) + '\n')

    def save_loss(self, epoch, step, loss):
        """
        Saves the loss of the current step and epoch in a text file.
        :param epoch: current epoch
        :param step: current step
        :param loss: loss of the current step
        """
        file_path = os.path.join(self.exp_path, 'losses.txt')
        with open(file_path, 'a+') as f:
            f.write('Epoch: ' + str(epoch) + ', step: ' + str(step) + ', loss: ' + str(float(loss)) + '\n')

    def save_loss_epoch(self, epoch, losses):
        """
        Saves the mean loss of the current epoch in a text file.
        :param epoch: current epoch
        :param losses: list of losses of the current epoch
        """
        file_path = os.path.join(self.exp_path, 'mean_losses.txt')
        with open(file_path, 'a+') as f:
            f.write('Epoch: ' + str(epoch) + ', loss: ' + str(float(np.mean(np.array(losses)))) + '\n')

    def save_model(self, epoch, model):
        """
        Saves the model at the current epoch in the model folder.
        :param epoch: current epoch
        :param model: model to be saved
        """
        filename = 'model-epoch-' + str(epoch) + '.pth'
        model_path = os.path.join(self.models_path, filename)
        torch.save(model.state_dict(), model_path)
