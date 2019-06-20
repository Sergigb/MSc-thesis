import torch
from torch import sigmoid, log, exp, max, abs
from torch.nn.functional import logsigmoid
from utils import gaussian


def nll_loss(alpha, sigma, mu, t):
    """
    Loss function, minimizes the negative log-likelihood.
    :param alpha: mixing coefficients (priors)
    :param sigma: covariances, one per kernel
    :param mu: expected value of each kernel
    :param t: batch of target vectors
    :return: loss
    """
    batch_size = alpha.shape[0]
    k = alpha.shape[1]
    t_dim = int(mu.shape[1] / k)

    loss = torch.zeros(batch_size)
    if torch.cuda.is_available():
        loss = loss.cuda()

    for i in range(k):
        likelihood_t = gaussian(t, mu[:, i*t_dim:(i+1)*t_dim], sigma[:, i])
        loss += alpha[:, i] * likelihood_t  # posterior
    loss = torch.mean(-torch.log(loss))
    return loss


def cross_entropy(pred, soft_targets):
    # return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    cont = torch.mean(soft_targets * logsigmoid(pred) + (1 - soft_targets) * log(1 - sigmoid(pred)), 1)
    # scalar = torch.FloatTensor([0]).cuda()
    # cont = torch.sum(max(pred, scalar.expand_as(pred)) - pred*soft_targets + log(1 + exp(-pred)), 1)
    return -torch.sum(cont)

