import pickle
import os
import json
import sys

import numpy as np
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import kde
from matplotlib.lines import Line2D

sys.path.insert(1, '../')

from model import CNN
from data_loader import get_wiki_data_loader

mixture_model = True
use_lda_labels = False  # use the labels generated by the LDA instead of the neural network
n_kernels = 3
n_topics = 40
batch_size = 128
model_path = '../models/mdn-3kernel6.pth'

emb_filename = '../data/2d-embeddings-3k6-test.npy'
ids_filename = '../data/ids-embeddings-3k6-test.pkl'
if mixture_model:
    alpha_filename = '../data/alpha-3k6-test.npy'
    sigma_filename = '../data/sigma-3k6-test.npy'


def rand_jitter(arr):
    stdev = .0005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def imscatter(x, y, image, ax=None, zoom=2.):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        pass
    im = OffsetImage(image, zoom=zoom, cmap=plt.get_cmap('gray'))
    im.set_zorder(10)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        ab.set_zorder(10)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if os.path.isfile(emb_filename):
    X_embedded = np.load(emb_filename)
    if mixture_model:
        alpha_values = np.load(alpha_filename)
        sigma_values = np.load(sigma_filename)
    with open(ids_filename, 'rb') as handle:
        ids = pickle.load(handle)
else:
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    dataset_path = './../../datasets/ImageCLEF_wikipedia/'
    json_labels_path = '../LDA/training_labels40_original.json'

    data_loader = get_wiki_data_loader(dataset_path, json_labels_path,
                                       transform, batch_size, shuffle=True,
                                       num_workers=8, return_ids=True)

    model = CNN(n_topics, n_kernels, out_dim=256, mixture_model=mixture_model, cnn='alexnet')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    X = []
    sigma_values = []
    alpha_values = []
    ids = []

    for step, (images, _, image_ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        if mixture_model:
            alphas, sigmas, out = model(images)
        else:
            out = model(images)

        #ids.extend(list(image_ids))
        image_ids = list(image_ids)
        for i in range(out.shape[0]):
            embeddings = out[i, :].cpu().detach().numpy()
            if mixture_model:
                alphas_ = alphas[i, :].cpu().detach().numpy()
                sigmas_ = sigmas[i, :].cpu().detach().numpy()
            for j in range(n_kernels):
                ids.append(image_ids[i])
                X.append(embeddings[j*n_topics:(j+1)*n_topics])
                if mixture_model:
                    alpha_values.append(alphas_[j])
                    sigma_values.append(sigmas_[j])

        print('Step ' + str(step+1) + '/' + str(len(data_loader)))
    print(len(X))
    X = np.array(X)
    if mixture_model:
        alpha_values = np.array(alpha_values)
        sigma_values = np.array(sigma_values)
    print(X.shape)
    print(len(ids))
    if mixture_model:
        print(alpha_values.shape)
        print(sigma_values.shape)

    with open(ids_filename, 'wb') as handle:
        pickle.dump(ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_embedded = TSNE(n_components=2, perplexity=50, early_exaggeration=64.0, verbose=1,
                      learning_rate=500.0, n_iter=4000).fit_transform(X)
    print(X_embedded.shape)
    np.save(emb_filename, X_embedded)
    if mixture_model:
        np.save(alpha_filename, alpha_values)
        np.save(sigma_filename, sigma_values)

    print("finished!")


# scatter random images
# fig, ax = plt.subplots()
# idxs = []
# for i in range(20000):
#     idx = int(np.random.choice(len(ids), 1))
#     idxs.append(idx)
#     image_filename = ids[idx]
#     image_filename = os.path.join('./../datasets/ImageCLEF_wikipedia/', image_filename)
#     imscatter(X_embedded[idx, 0], X_embedded[idx, 1], image_filename, zoom=0.20)
#
# plt.show()


# plot consecutive pairs
# fig, ax = plt.subplots()
# i = 0
# offset = 550  # must be even
# while i < 10000:
#    i += 2
#    image_filename = ids[i+offset ]
#    image_filename = os.path.join('./../datasets/ImageCLEF_wikipedia/', image_filename)
##    imscatter(X_embedded[i+offset, 0], X_embedded[i+offset, 1], image_filename, zoom=0.05)
#    image_filename2 = ids[i+offset+1 ]
#    image_filename2 = os.path.join('./../datasets/ImageCLEF_wikipedia/', image_filename2)
#    #imscatter(X_embedded[i+offset+1, 0], X_embedded[i+offset+1, 1], image_filename2, zoom=0.05)
##    plt.plot(X_embedded[i+offset, 0], X_embedded[i+offset+1, 0], X_embedded[i+offset, 1], X_embedded[i+offset+1, 1])
#    ax.add_line(Line2D([X_embedded[i+offset, 0], X_embedded[i+offset+1, 0]],[X_embedded[i+offset, 1],
#                        X_embedded[i+offset+1, 1]], linewidth=1, color='blue', zorder=10))
#    ax.scatter([X_embedded[i+offset, 0], X_embedded[i+offset+1, 0]],[X_embedded[i+offset, 1],
#                        X_embedded[i+offset+1, 1]], color='red', marker='*', zorder=0, s=8)
#    print(i)
# plt.autoscale()
# plt.show()


# sigma visualization
# sigma_norm = (sigma_values-min(sigma_values)) / (max(sigma_values)-min(sigma_values))
plt.scatter(rand_jitter(X_embedded[:, 0]), rand_jitter(X_embedded[:, 1]), c=sigma_values[:], s=4)
plt.clim(sigma_values.min(), sigma_values.max())
plt.colorbar()
plt.show()

# alpha visualization
# alpha_norm = (alpha_values-min(alpha_values)) / (max(alpha_values)-min(alpha_values))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=alpha_values, s=8)
plt.clim(alpha_values.min(),alpha_values.max())
plt.colorbar()
plt.show()


# density visualization
# nbins = 1500
# k = kde.gaussian_kde([X_embedded[:, 0], X_embedded[:, 1]])
# xi, yi = np.mgrid[X_embedded[:, 0].min():X_embedded[:, 0].max():nbins * 1j,
#                   X_embedded[:, 0].min():X_embedded[:, 0].max():nbins * 1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
# plt.show()


#  scatter random images in a matrix form
# fig, ax = plt.subplots()
# x_min_width = np.min(X_embedded[:, 0])
# y_min_width = np.min(X_embedded[:, 1])
#
# X_embedded[:, 0] += abs(x_min_width)
# X_embedded[:, 1] += abs(y_min_width)
#
# fig, ax = plt.subplots()
# idxs = []
#
# for i in range(5000):
#     idx = int(np.random.choice(len(ids), 1))
#     idxs.append(idx)
#     image_filename = ids[idx]
#     image_filename = os.path.join('./../datasets/ImageCLEF_wikipedia/', image_filename)
#     imscatter(int(X_embedded[idx, 0]/4.5), int(X_embedded[idx, 1]/8), image_filename, zoom=0.2)
#
# plt.show()
