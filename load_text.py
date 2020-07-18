import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import tensorflow_datasets as tfds

import data_helpers

# Load and preprecess data

negative_data_file = 'C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/TensorFlow/data/rt-polaritydata/rt-polarity.neg'
positive_data_file = 'C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/TensorFlow/data/rt-polaritydata/rt-polarity.pos'

x_text, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)

all_labeled_data = tf.data.Dataset.from_tensor_slices((x_text, y))

# Build vocabulary

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()

for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

# Convert sentences to list of integers

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def do_encode(text, label):
    encoded_text = encoder.encode(str(text))
    return encoded_text[3:-3], label


#
def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(do_encode,
                                         inp=[text, label],
                                         Tout=(tf.int32, tf.int32))

    return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

sentences_max_len = 0

for text, label in all_encoded_data.as_numpy_iterator():
    sentences_max_len = max(sentences_max_len, len(text))

# Split to train data and test data

BATCH_SIZE = 64
TAKE_SIZE = 2000
BUFFER_SIZE = 50000

all_encoded_data = all_encoded_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

train_data = all_encoded_data.skip(TAKE_SIZE)
train_data = train_data.padded_batch(batch_size=64, padded_shapes=([sentences_max_len], [2]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(batch_size=64, padded_shapes=([sentences_max_len], [2]))

# for elements1 in test_data.as_numpy_iterator():
#     for elements2 in train_data.as_numpy_iterator():
#         if len(elements1[0]) != len(elements2[0]):
#             continue
#         if all(elements1[0] == elements2[0]) and all(elements1[1] == elements2[1]):
#             print(elements1[0], elements2[0], sep = "\n")
#             print("\n" "\n")

vocab_size += 1
