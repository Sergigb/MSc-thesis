import os,sys,re
import json

from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import gensim
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import remove_markup

import sys
sys.path.insert(1, '../LDA/')
from preprocess_text import preprocess

NUM_TOPICS = 40

data_pairs_files = ['data_pairs_1.json', 'data_pairs_2.json', 'data_pairs_3.json', 'data_pairs_4.json']  # about 690k sections

print('Learning LDA topic model with '+str(NUM_TOPICS)+' topics')

if not os.path.isfile('./dictionary.dict') or not os.path.isfile('./bow.mm'):
    # list for tokenized documents in loop
    texts = []
    for data_pairs_file in data_pairs_files:
        with open(data_pairs_file) as f:
            data_pairs = json.load(f)
        for data_pair in data_pairs:
            texts.append(preprocess(str(data_pair['text'])))
            sys.stdout.write("\rNum texts processed: " + str(len(texts)))
            sys.stdout.flush()
        del data_pairs
    sys.stdout.write(' Done!\n')

# turn our tokenized documents into a id <-> term dictionary
if not os.path.isfile('./dictionary.dict'):
    print 'Turn our tokenized documents into a id <-> term dictionary ...',
    sys.stdout.flush()
    dictionary = corpora.Dictionary(texts)
    dictionary.save('./dictionary.dict')
else:
    print 'Loading id <-> term dictionary from ./dictionary.dict ...',
    sys.stdout.flush()
    dictionary = corpora.Dictionary.load('./dictionary.dict')
print ' Done!'

# ignore words that appear in less than 20 documents or more than 50% documents
dictionary.filter_extremes(no_below=20, no_above=0.5)
    
# convert tokenized documents into a document-term matrix
if not os.path.isfile('./bow.mm'):
    print 'Convert tokenized documents into a document-term matrix ...',
    sys.stdout.flush()
    corpus = [dictionary.doc2bow(text) for text in texts]
    gensim.corpora.MmCorpus.serialize('./bow.mm', corpus)
else:
    print 'Loading document-term matrix from ./bow.mm ...',
    sys.stdout.flush()
    corpus = gensim.corpora.MmCorpus('./bow.mm')
print ' Done!'

#del texts

# Learn the LDA model
print 'Learning the LDA model ...',
sys.stdout.flush()
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word = dictionary, passes=20)
#ldamodel = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary ,num_topics = NUM_TOPICS, workers=3, passes=3)
ldamodel.save('ldamodel'+str(NUM_TOPICS)+'.lda')
print ' Done!'
