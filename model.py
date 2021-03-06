import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, t_dim, k, out_dim=None, mixture_model=True, cnn='alexnet', pretrained=False):
        """
        :param t_dim: dimensionality of the target space (number of LDA topics)
        :param k: number of Gaussian kernels
        :param out_dim: Number of dimensions of the tensor outputted by the cnn. Should be set to None if we are using
        the TextTopicNet architecture.
        :param mixture_model: if false, replicates the TextTopicNet architecture instead of using the mixture model
        network
        """
        super(CNN, self).__init__()

        if not mixture_model:
            out_dim = t_dim
        elif out_dim is None:
            out_dim = k*t_dim

        self.mixture_model = mixture_model
        if cnn == 'alexnet':
            if pretrained:
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                self.cnn = nn.Sequential(model, nn.Linear(4096, out_dim))
            else:  # kept for compatibility with older models
                self.cnn = models.alexnet(pretrained=False, num_classes=out_dim)

        elif cnn == 'resnet':
            if pretrained:
                model = models.resnet50(pretrained=True, num_classes=out_dim)
                self.cnn = nn.Sequential(list(model.children())[:-1], nn.Linear(2048, out_dim))
            else:
                self.cnn = models.resnet50(pretrained=False, num_classes=out_dim)
        else:
            print("wrong cnn option")
            exit(0)

        if mixture_model:
            self.relu = nn.ReLU()
            self.alpha_out = nn.Linear(out_dim, k)
            self.sigma_out = nn.Linear(out_dim, k)
            self.mu_out = nn.Linear(out_dim, k*t_dim)

    def forward(self, x):
        """
        :param x: batch of images
        :return: alpha (mix coefficients), sigma (covariances), mu (expected values)
        """
        if self.mixture_model:
            out = self.cnn(x)
            out = self.relu(out)
            alpha = torch.softmax(self.alpha_out(out), 1)
            sigma = torch.exp(self.sigma_out(out))
            mu = self.mu_out(out)

            return alpha, sigma, mu
        else:
            out = self.cnn(x)
            if not self.training:
                out = torch.sigmoid(out)

            return out
