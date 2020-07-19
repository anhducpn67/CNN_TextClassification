import pickle

import gensim as gensim
import numpy as np

import load_text

vocab = load_text.vocabulary_set
encoder = load_text.encoder
word2vec_path = "C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/TensorFlow/data/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
dict = {}
dict[0] = np.zeros(300, dtype='float32')
for elements in vocab:
    if elements in model:
        dict[encoder.encode(elements)[0]] = list(model.get_vector(elements))
    else:
        dict[encoder.encode(elements)[0]] = list(np.random.uniform(-0.25, 0.25, 300))

with open('dict.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
