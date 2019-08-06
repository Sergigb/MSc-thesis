import pickle
import os
import json

import numpy as np
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import kde

from model import CNN
from data_loader import get_wiki_data_loader

mixture_mode = True
n_kernels = 2

emb_filename = 'data/3d-embeddings-2mdn5.npy'
ids_filename = 'data/ids-embeddings-2mdn5-3d.pkl'
alpha_filename = 'data/alpha-2mdn5-3d.npy'
sigma_filename = 'data/sigma-2mdn5-3d.npy'

if os.path.isfile(emb_filename):
    X_embedded = np.load(emb_filename)
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
    batch_size = 512

    dataset_path = './../datasets/ImageCLEF_wikipedia/'
    json_labels_path = 'LDA/training_labels40_original.json'

    data_loader = get_wiki_data_loader(dataset_path, json_labels_path,
                                       transform, batch_size, shuffle=True,
                                       num_workers=4, return_ids=True)

    model_path = 'models/mdn-2kernel5.pth'
    model = CNN(40, n_kernels, out_dim=256, mixture_model=mixture_mode)
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

        if mixture_mode:
            alphas, sigmas, out = model(images)
        else:
            out = model(images)

        #ids.extend(list(image_ids))
        image_ids = list(image_ids)
        for i in range(out.shape[0]):
            embeddings = out[i, :].cpu().detach().numpy()
            alphas_ = alphas[i, :].cpu().detach().numpy()
            sigmas_ = sigmas[i, :].cpu().detach().numpy()
            for j in range(n_kernels):
                ids.append(image_ids[i])
                X.append(embeddings[j*40:(j+1)*40])
                alpha_values.append(alphas_[j])
                sigma_values.append(sigmas_[j])

        print('Step ' + str(step+1) + '/' + str(len(data_loader)))
    print(len(X))
    X = np.array(X)
    alpha_values = np.array(alpha_values)
    sigma_values = np.array(sigma_values)
    print(X.shape)
    print(len(ids))
    print(alpha_values.shape)
    print(sigma_values.shape)

    with open(ids_filename, 'wb') as handle:
        pickle.dump(ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_embedded = TSNE(n_components=3, perplexity=50, early_exaggeration=64.0, verbose=1,
                      learning_rate=500.0).fit_transform(X)
    print(X_embedded.shape)
    np.save(emb_filename, X_embedded)
    np.save(alpha_filename, alpha_values)
    np.save(sigma_filename, sigma_values)

    print("finished!")

# fig, ax = plt.subplots()

# sigma visualization

fig = plt.figure()
ax = Axes3D(fig)

num = (len(X_embedded))
num = num/32

sigma_norm = (sigma_values-min(sigma_values)) / (max(sigma_values)-min(sigma_values))
ax.scatter(X_embedded[:num, 0], X_embedded[:num, 1], X_embedded[:num, 2], c=sigma_norm[:num])
plt.show()

# alpha visualization
# alpha_norm = (alpha_values-min(alpha_values)) / (max(alpha_values)-min(alpha_values))
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=alpha_norm)
# plt.show()

