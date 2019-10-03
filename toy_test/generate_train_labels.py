import os, sys
import json

import gensim
from gensim import corpora, models

sys.path.insert(1, '../LDA/')
from preprocess_text import preprocess

NUM_TOPICS = 2

# load id <-> term dictionary
if not os.path.isfile('./dictionary.dict'):
    sys.exit('ERR: ID <-> Term dictionary file ./dictionary.dict not found!')

print 'Loading id <-> term dictionary from ./dictionary.dict ...',
sys.stdout.flush()
dictionary = corpora.Dictionary.load('./dictionary.dict')
print ' Done!'
# ignore words that appear in less than 20 documents or more than 50% documents
dictionary.filter_extremes(no_below=20, no_above=0.5)

# load document-term matrix
if not os.path.isfile('./bow.mm'):
    sys.exit('ERR: Document-term matrix file ./bow.mm not found!')

print 'Loading document-term matrix from ./bow.mm ...',
sys.stdout.flush()
corpus = gensim.corpora.MmCorpus('./bow.mm')
print ' Done!'

# load LDA model
if not os.path.isfile('ldamodel'+str(NUM_TOPICS)+'.lda'):
    sys.exit('ERR: LDA model file ./ldamodel'+str(NUM_TOPICS)+'.lda not found!')

print 'Loading LDA model from file ./ldamodel'+str(NUM_TOPICS)+'.lda ...',
sys.stdout.flush()
ldamodel = models.LdaModel.load('ldamodel'+str(NUM_TOPICS)+'.lda')
print ' Done!'

# transform ALL documents into LDA space
target_labels = {}
ignored = 0
with open('data_pairs.json') as f:
    data_pairs = json.load(f)
for data_pair in data_pairs:
    raw = str(data_pair['text'])
    img_path = data_pair['img']
    if img_path.find('.webp') != -1 or img_path.find('.tif') != -1 or img_path.find('.tiff') != -1:
        ignored += 1
        continue
    img_path = img_path.replace("../../datasets/ali/", "")
    tokens = preprocess(raw)
    bow_vector = dictionary.doc2bow(tokens)
    lda_vector = ldamodel.get_document_topics(bow_vector, minimum_probability=None)
    lda_vector = sorted(lda_vector,key=lambda x:x[1],reverse=True)
    topic_prob = {}
    for instance in lda_vector:
        topic_prob[instance[0]] = instance[1]
    labels = []
    for topic_num in range(0,NUM_TOPICS):
        if topic_num in topic_prob.keys():
            labels.append(float(topic_prob[topic_num]))
        else:
            labels.append(0)
    if img_path in target_labels.keys():
        print("collision with image ", img_path)
    target_labels[img_path] = labels
    sys.stdout.write('\rDocuments processed: ' + str(len(target_labels.keys()))
                     + ", ignored articles: " + str(ignored))
    sys.stdout.flush()

sys.stdout.write(' Done!\n')
print("Ignored articles: " + str(ignored))
# save key,labels pairs into json format file
with open('./training_labels'+str(NUM_TOPICS)+'.json','w') as fp:
    json.dump(target_labels, fp)
