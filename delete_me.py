# import gensim
# import numpy as np
#
# from preprocess_text import preprocess_imageclef
#
# n_topics = 40
#
# words = ['airplane', 'car', 'universe', 'planet', 'race car', 'football', 'sport', 'stadium',
#          'jet fighter', 'gun', 'computer', 'map', 'sky', 'building', 'train', 'public transport']
#
# dictionary = gensim.corpora.Dictionary.load('./LDA/dictionary_original.dict')
# ldamodel = gensim.models.ldamulticore.LdaMulticore.load('./LDA/ldamodel' + str(n_topics) + '_original.lda',
#                                                         mmap='r')
#
# for word in words:
#     process = preprocess_imageclef(word)
#
#     if process[1] != '':
#         tokens = process[0]
#         bow_vector = dictionary.doc2bow(tokens)
        # lda_vector = ldamodel.get_document_topics(bow_vector, minimum_probability=None)
        # print(word)
        # print(lda_vector)


        # vector = []
        # for topic in lda_vector:
        #     vector.append(topic[1])
        # print(word)
        # print(np.array(vector))
        # print(np.array(vector).sum())



import gensim
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import CNN

LDAmodel = gensim.models.ldamulticore.LdaMulticore.load('LDA/ldamodel40_original.lda')
dictionary = gensim.corpora.Dictionary.load('./LDA/dictionary_original.dict')

to_tensor = transforms.ToTensor()
resize = transforms.Resize((256, 256))

model_path = 'models/mdn-3kernel6.pth'
num_kernels = 3
n_topics = 40
model = CNN(n_topics, num_kernels, out_dim=256)
model.load_state_dict(torch.load(model_path))
model.eval()

# image_path = 'images/race.jpg'
image_path = './../datasets/ImageCLEF_wikipedia/images/10/93590.jpg'
print(image_path)
image = Image.open(image_path)
image = image.convert('RGB')
image = resize(image)
tensor_image = to_tensor(image).unsqueeze(0)

alpha, sigma, mu = model(tensor_image)

max_coeff, idx = torch.max(alpha, 1)
print("max mixing coeff: ", str(float(max_coeff)))
topics = mu[:, n_topics*idx:n_topics*(idx+1)]

for i in range(num_kernels):
    topics = mu[:, n_topics * i:n_topics * (i + 1)]
    _, idx2 = torch.max(topics, 1)
    print(alpha[:, i], LDAmodel.show_topic(idx2))
    # print(mu[:, n_topics * i:n_topics * (i + 1)])
plt.imshow(plt.imread(image_path))
plt.show()

exit(0)