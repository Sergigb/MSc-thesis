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

model_path = 'exps/1kernel_ttn_labels/models/model-epoch-5.pth'
num_kernels = 1
n_topics = 40
model = CNN(n_topics, num_kernels, bn=True)
model.load_state_dict(torch.load(model_path))
model.eval()

# image_path = 'images/race.jpg'
image_path = './../datasets/ImageCLEF_wikipedia/images/8/79963.jpg'
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
    print(mu[:, n_topics * i:n_topics * (i + 1)])

exit(0)

