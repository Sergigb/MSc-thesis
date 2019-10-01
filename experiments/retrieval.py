from __future__ import division

import sys
import os
import json
import argparse

import numpy as np
from PIL import Image
import gensim
import scipy.stats as sp
import torch
import torchvision.transforms as transforms
from scipy.spatial import distance
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(1, '../')

from preprocess_text import preprocess_imageclef
from model import CNN
from utils import likelihood

type_data_list = ['test']


# Function to compute average precision for text retrieval given image as input
def get_AP_img2txt(sorted_scores, given_image, top_k):
    consider_top = sorted_scores[:top_k]
    top_text_classes = [GT_txt2img[i[0]][1] for i in consider_top]
    class_of_image = GT_img2txt[given_image][1]
    T = top_text_classes.count(class_of_image)
    R = top_k
    sum_term = 0
    for i in range(0, R):
        if top_text_classes[i] != class_of_image:
            pass
        else:
            p_r = top_text_classes[:i + 1].count(class_of_image)
            sum_term = sum_term + float(p_r / len(top_text_classes[:i + 1]))
    if T == 0:
        return 0
    else:
        return float(sum_term / T)


# Function to compute average precision for image retrieval given text as input
def get_AP_txt2img(sorted_scores, given_text, top_k):
    consider_top = sorted_scores[:top_k]
    top_image_classes = [GT_img2txt[i[0]][1] for i in consider_top]
    class_of_text = GT_txt2img[given_text][1]
    T = top_image_classes.count(class_of_text)
    R = top_k
    sum_term = 0
    for i in range(0, R):
        if top_image_classes[i] != class_of_text:
            pass
        else:
            p_r = top_image_classes[:i + 1].count(class_of_text)
            sum_term = sum_term + float(p_r / len(top_image_classes[:i + 1]))
    if T == 0:
        return 0
    else:
        return float(sum_term / T)


### Start : Generating image representations of wikipedia dataset for performing multi modal retrieval
text_dir_wd = '../../datasets/Wikipedia/texts_wd/' # Path to wikipedia dataset text files
images_root = '../../datasets/Wikipedia/images_wd_256/'

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('-k', type=int)
parser.add_argument('-cnn', type=str, default="alexnet")
parser.add_argument('--mixture_model', action='store_true')
parser.add_argument('--n_topics', type=int, default=40)
parser.add_argument('--distance', type=str, default="probability")
parser.add_argument('--query_type', type=str)
args = parser.parse_args()

model_path = args.model_path
feat_root = '../data/features/' + model_path.split("/")[-1].split(".")[0] + "/"
cnn = args.cnn
mixture_model = args.mixture_model
n_kernels = args.k
out_dim = 256
num_topics = args.n_topics
dist = args.distance # distance used in the retrieval part, 'entropy', 'euclidean' or 'probability'
visualize_results = False

if not mixture_model:
    n_kernels = 1

print(model_path)

if not os.path.isdir('../data/features'):
    os.mkdir('../data/features')

if not os.path.isdir(feat_root):  # extract features
    os.mkdir(feat_root)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = CNN(num_topics, n_kernels, mixture_model=mixture_model, out_dim=out_dim, cnn=cnn)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    im_txt_pair_wd = open('../../datasets/Wikipedia/wikipedia_dataset/' +
                          str(type_data_list[0])+'set_txt_img_cat.list', 'r').readlines() # Image-text pairs
    img_files = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd] # List of image files in wikipedia dataset

    progress = 0
    for filename in img_files:
        image_path = os.path.join(images_root, filename)
        im = Image.open(image_path).convert('RGB')
        im = transform(im)
        im = im.unsqueeze(0)
        if torch.cuda.is_available():
            im = im.cuda()

        if not mixture_model:
            representation = model(im)
        else:
            alpha, sigma, representation = model(im)

        np.save(os.path.join(feat_root, filename), representation.cpu().detach().numpy().squeeze())
        if mixture_model:
            np.save(os.path.join(feat_root, filename + '-alpha'), alpha.cpu().detach().numpy().squeeze())
            np.save(os.path.join(feat_root, filename + '-sigma'), sigma.cpu().detach().numpy().squeeze())

        progress += 1
        sys.stdout.write("\rCompleted:  " + str(progress) + "/" + str(len(img_files)))
        sys.stdout.flush()
print("")

### Start : Generating text representation of wikipedia dataset for performing multi modal retrieval

choose_set_list = type_data_list

# IMPORTANT ! Specify minimum probability for LDA
min_prob_LDA = None

# load id <-> term dictionary
dictionary = gensim.corpora.Dictionary.load('../LDA/dictionary_original.dict')
#dictionary = gensim.corpora.Dictionary.load('../LDA/dictionary.dict')
#dictionary = gensim.corpora.Dictionary.load('../wiki/dictionary.dict')
#dictionary.filter_extremes(no_below=20, no_above=0.5)  # comment this when using the original dictionary

# load LDA model
ldamodel = gensim.models.ldamulticore.LdaMulticore.load('../LDA/ldamodel' + str(num_topics) + '_original.lda', mmap='r')
#ldamodel = gensim.models.ldamulticore.LdaMulticore.load('../LDA/ldamodel' + str(num_topics) + '.lda', mmap='r')

for choose_set in choose_set_list:
    # Read image-text document pair ids
    im_txt_pair_wd = open('../../datasets/Wikipedia/wikipedia_dataset/' + str(choose_set) + 'set_txt_img_cat.list', 'r').readlines()
    text_files_wd = [text_dir_wd + i.split('\t')[0] + '.xml' for i in im_txt_pair_wd]
    output_path_root = '../data/multi_modal_retrieval/text/'
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)
    output_file_path = 'wd_txt_' + str(num_topics) + '_' + str(choose_set) + '.json'
    output_path = output_path_root + output_file_path

    # transform ALL documents into LDA space
    TARGET_LABELS = {}

    for i in text_files_wd:
        sys.stdout.write('\rGenerating text representation for document number : ' + str(len(TARGET_LABELS.keys())))
        sys.stdout.flush()
        raw = open(i, 'r').read()
        process = preprocess_imageclef(raw)
        if process[1] != '':
            tokens = process[0]
            bow_vector = dictionary.doc2bow(tokens)
            lda_vector = ldamodel.get_document_topics(bow_vector, minimum_probability=None)
            # lda_vector = ldamodel[bow_vector]
            lda_vector = sorted(lda_vector, key=lambda x: x[1], reverse=True)
            topic_prob = {}
            for instance in lda_vector:
                topic_prob[instance[0]] = float(instance[1])
            labels = []
            for topic_num in range(0, num_topics):
                if topic_num in topic_prob.keys():
                    labels.append(topic_prob[topic_num])
                else:
                    labels.append(0)
            list_name = list_name = i.split('/')
            TARGET_LABELS[list_name[len(list_name) - 1].split('.xml')[0]] = labels
    print("")
    # Save thi as json.
    json.dump(TARGET_LABELS, open(output_path, 'w'))

### End : Generating text representation of wikipedia dataset for performing multi modal retrieval

### Start : Perform multi-modal retrieval on wikipedia dataset.

for type_data in type_data_list:
    # Wikipedia data paths
    im_txt_pair_wd = open('../../datasets/Wikipedia/wikipedia_dataset/' + str(type_data) + 'set_txt_img_cat.list', 'r').readlines()
    image_files_wd = [i.split('\t')[1] + '.jpg' for i in im_txt_pair_wd]

    # Read the required Grount Truth for the task.
    GT_img2txt = {}  # While retrieving text, you need image as key.
    GT_txt2img = {}  # While retrieving image, you need text as key.
    for i in im_txt_pair_wd:
        GT_img2txt[i.split('\t')[1]] = (i.split('\t')[0], i.split('\t')[2])  # (Corresponding text, class)
        GT_txt2img[i.split('\t')[0]] = (i.split('\t')[1], i.split('\t')[2])  # (Corresponding image, class)

    # Load image representation
    image_rep = feat_root

    # Load text representation
    data_text = json.load(
        open('../data/multi_modal_retrieval/text/wd_txt_' + str(num_topics) + '_' + str(type_data) + '.json',
             'r'))

    image_ttp = {}
    image_alphas = {}
    image_sigmas = {}
    for i in GT_img2txt.keys():
        sample = i
        value = np.load(image_rep + i + '.jpg.npy')
        image_ttp[sample] = value
        if mixture_model:
            alpha = np.load(image_rep + i + '.jpg-alpha.npy')
            sigma = np.load(image_rep + i + '.jpg-sigma.npy')
            image_alphas[sample] = alpha
            image_sigmas[sample] = sigma

    # Convert text_rep to numpy format
    text_ttp = {}
    for i in data_text.keys():
        text_ttp[i] = np.asarray(data_text[i], dtype='f')
    # IMPORTANT NOTE : always sort the images and text in lexi order !!
    # If Query type is input=image, output=text
    mAP = 0
    order_of_images = sorted(image_ttp.keys())
    order_of_texts = sorted(text_ttp.keys())
    counter = 0
    if args.query_type == 'img2txt':
        for given_image in order_of_images:
            sys.stdout.write('\rPerforming retrieval for document number : ' + str(counter))
            sys.stdout.flush()

            score_texts = []
            image_reps = image_ttp[given_image]
            if mixture_model:
                image_alpha = image_alphas[given_image]
                image_sigma = image_sigmas[given_image]
            for given_text in order_of_texts:
                text_reps = text_ttp[given_text]
                if dist == 'euclidean':
                    min_score = 99999999999
                    for j in range(n_kernels):
                        image_rep = image_reps[j * num_topics:(j + 1) * num_topics]
                        given_score = distance.euclidean(text_reps, image_rep)
                        if given_score < min_score:
                            min_score = given_score
                    score_texts.append((given_text, min_score))
                elif dist == 'entropy':
                    min_score = 99999999999
                    for j in range(n_kernels):
                        image_rep = image_reps[j * num_topics:(j + 1) * num_topics]
                        given_score = sp.entropy(text_reps, image_rep)
                        if given_score < min_score:
                            min_score = given_score
                    score_texts.append((given_text, min_score))
                elif dist == 'probability':
                    given_score = 1 - likelihood(image_alpha, image_sigma, image_reps, text_reps)
                    score_texts.append((given_text, given_score))
            sorted_scores = sorted(score_texts, key=lambda x: x[1], reverse=False)
            mAP = mAP + get_AP_img2txt(sorted_scores, given_image, top_k=len(order_of_texts))
            counter += 1
        print('MAP img2txt : ' + str(float(mAP / len(image_ttp.keys()))), 'red')
    if args.query_type == 'txt2img':
        for given_text in order_of_texts:
            sys.stdout.write('\rPerforming retrieval for document number : ' + str(counter))
            sys.stdout.flush()

            score_images = []
            text_reps = text_ttp[given_text]
            for given_image in order_of_images:
                image_reps = image_ttp[given_image]
                if mixture_model:
                    image_alpha = image_alphas[given_image]
                    image_sigma = image_sigmas[given_image]
                if dist == 'euclidean':
                    min_score = 99999999999
                    for j in range(n_kernels):
                        image_rep = image_reps[j*num_topics:(j+1)*num_topics]
                        given_score = distance.euclidean(text_reps, image_rep)
                        if given_score < min_score:
                            min_score = given_score
                    score_images.append((given_image, min_score))
                elif dist == 'entropy':
                    min_score = 99999999999
                    for j in range(n_kernels):
                        image_rep = image_reps[j*num_topics:(j+1)*num_topics]
                        given_score = sp.entropy(text_reps, image_rep)
                        if given_score < min_score:
                            min_score = given_score
                    score_images.append((given_image, min_score))
                elif dist == 'probability':
                    given_score = 1 - likelihood(image_alpha, image_sigma, image_reps, text_reps)
                    score_images.append((given_image, given_score))
            sorted_scores = sorted(score_images, key=lambda x: x[1], reverse=False)

            if visualize_results:
                image_categories = ['art', 'biology', 'geography', 'history', 'literature', 'media', 'music', 'royalty',
                                    'sport', 'warfare']  # List of document (image-text) categories in wikipedia dataset
                f, axarr = plt.subplots(1, 5)
                for i in range(5):
                    img = Image.open('../../datasets/Wikipedia/images_wd_256/' + sorted_scores[i+1][0] + '.jpg')
                    img = img.resize((512, 512), Image.ANTIALIAS)
                    axarr[i].imshow(np.array(img), cmap='gray')
                    axarr[i].axis('off')
                    axarr[i].set_title(image_categories[int(GT_img2txt[sorted_scores[i+1][0]][1])-1] + "("
                                            + str(1 - sorted_scores[i+1][1])[:4] + ")")
                plt.suptitle("Text query topic: " + image_categories[int(GT_txt2img[given_text][1])-1])
                plt.show()

            mAP = mAP + get_AP_txt2img(sorted_scores, given_text, top_k=len(order_of_images))

            counter += 1
        print('MAP txt2img : ' + str(float(mAP / len(text_ttp.keys()))), 'red')
print("")
        ### End : Perform multi modal retrieval on wikipedia dataset
