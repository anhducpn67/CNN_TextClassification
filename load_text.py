import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import data_helpers
import pickle
import numpy as np

# Load and preprecess data

negative_data_file = '//data/rt-polaritydata/rt' \
                     '-polarity.neg '
positive_data_file = '//data/rt-polaritydata/rt' \
                     '-polarity.pos '
predict_data_file = '//data/rt-polaritydata/predict.txt'

x_text, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)
all_labeled_data = tf.data.Dataset.from_tensor_slices((x_text, y))

predict_data = data_helpers.load_predict_data(predict_data_file)

# Build vocabulary

# tokenizer = tfds.features.text.Tokenizer()
#
# vocabulary_set = set()
#
# for text_tensor, _ in all_labeled_data:
#     some_tokens = tokenizer.tokenize(text_tensor.numpy())
#     vocabulary_set.update(some_tokens)
#
# vocabulary_set = list(vocabulary_set)
#
# with open('vocab.pickle', 'wb') as handle:
#    pickle.dump(vocabulary_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('vocab.pickle', 'rb') as handle:
    vocabulary_set = pickle.load(handle)

vocab_size = len(vocabulary_set)

# Convert sentences to list of integers

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def do_encode(text, label):
    encoded_text = encoder.encode(str(text))
    return encoded_text[3:-3], label


def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(do_encode,
                                         inp=[text, label],
                                         Tout=(tf.int32, tf.int32))

    return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)
predict_data = [encoder.encode(s) for s in predict_data]

sentences_max_len = 0
for text, label in all_encoded_data.as_numpy_iterator():
    sentences_max_len = max(sentences_max_len, len(text))

for idx in range(len(predict_data)):
    while len(predict_data[idx]) < sentences_max_len:
        predict_data[idx].append(0)

predict_data = tf.data.Dataset.from_tensor_slices(predict_data)

# Split to train data and test data

BATCH_SIZE = 64
BUFFER_SIZE = 50000

all_encoded_data = all_encoded_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

train_data = all_encoded_data.skip(2000)
test_data = all_encoded_data.take(2000)
val_data = test_data.skip(1000)
test_data = test_data.take(1000)

train_data = train_data.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([sentences_max_len], [2]))
test_data = test_data.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([sentences_max_len], [2]))
val_data = val_data.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([sentences_max_len], [2]))

vocab_size += 1

with open('dict.pickle', 'rb') as handle:
    dict_word2vec = pickle.load(handle)

embedding_matrix = np.zeros((vocab_size, 300))

for index in range(0, vocab_size):
    embedding_matrix[index] = dict_word2vec[index]
