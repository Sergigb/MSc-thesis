import pickle
import os
import json

import numpy as np
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import kde
import gensim

from model import CNN
from data_loader import get_wiki_data_loader
from preprocess_text import preprocess_imageclef

mixture_mode = True
n_kernels = 3
n_topics = 40


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

emb_filename = 'data/2d-embeddings-mdn-3kernel-embeddings.npy'
ids_filename = 'data/ids-embeddings-mdn-3kernel-embeddings.pkl'
alpha_filename = 'data/alpha-mdn-3kernel-embeddings.npy'
sigma_filename = 'data/sigma-mdn-3kernel-embeddings.npy'

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
                                       num_workers=8, return_ids=True)

    model_path = 'models/mdn-3kernel6.pth'

    model = CNN(n_topics, n_kernels, out_dim=256, mixture_model=mixture_mode)
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
                X.append(embeddings[j*n_topics:(j+1)*n_topics])
                alpha_values.append(alphas_[j])
                sigma_values.append(sigmas_[j])

        print('Step ' + str(step+1) + '/' + str(len(data_loader)))
    print(len(X))

    words = ['airplane', 'car', 'universe', 'planet', 'race car', 'football', 'sport', 'stadium',
             'jet', 'gun', 'computer', 'map', 'sky', 'building', 'train', 'transport']

    dictionary = gensim.corpora.Dictionary.load('./LDA/dictionary_original.dict')
    ldamodel = gensim.models.ldamulticore.LdaMulticore.load('./LDA/ldamodel' + str(n_topics) + '_original.lda',
                                                            mmap='r')

    for word in words:
        process = preprocess_imageclef(word)

        if process[1] != '':
            tokens = process[0]
            bow_vector = dictionary.doc2bow(tokens)
            lda_vector = ldamodel.get_document_topics(bow_vector, minimum_probability=None)

            lda_vector = sorted(lda_vector, key=lambda x: x[1], reverse=True)
            topic_prob = {}
            for instance in lda_vector:
                topic_prob[instance[0]] = float(instance[1])
            labels = []
            for topic_num in range(0, n_topics):
                if topic_num in topic_prob.keys():
                    labels.append(topic_prob[topic_num])
                else:
                    labels.append(0)

    X = np.array(X)
    alpha_values = np.array(alpha_values)
    sigma_values = np.array(sigma_values)
    print(X.shape)
    print(len(ids))
    print(alpha_values.shape)
    print(sigma_values.shape)

    with open(ids_filename, 'wb') as handle:
        pickle.dump(ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_embedded = TSNE(n_components=2, perplexity=50, early_exaggeration=64.0, verbose=1,
                      learning_rate=500.0, n_iter=4000).fit_transform(X)
    print(X_embedded.shape)
    np.save(emb_filename, X_embedded)
    np.save(alpha_filename, alpha_values)
    np.save(sigma_filename, sigma_values)

    print("finished!")

fig, ax = plt.subplots()
idxs = []

# scatter random images
for i in range(20000):
    idx = int(np.random.choice(len(ids), 1))
    idxs.append(idx)
    image_filename = ids[idx]
    image_filename = os.path.join('./../datasets/ImageCLEF_wikipedia/', image_filename)
    imscatter(X_embedded[idx, 0], X_embedded[idx, 1], image_filename, zoom=0.20)

plt.show()




