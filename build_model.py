import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import load_text

embedding_dim = 300
inputs = keras.Input(shape=(54,), name="input_text")
embedding_layer = layers.Embedding(load_text.vocab_size, embedding_dim, name="embedding")(inputs)
conv_3_layer = layers.Conv1D(100, 3, activation='relu', name="filter_size_3")(embedding_layer)
conv_4_layer = layers.Conv1D(100, 4, activation='relu', name="filter_size_4")(embedding_layer)
conv_5_layer = layers.Conv1D(100, 5, activation='relu', name="filter_size_5")(embedding_layer)
max_pool_3_layer = layers.MaxPool1D(pool_size=52, name="max_pool_3", padding="same")(conv_3_layer)
max_pool_4_layer = layers.MaxPool1D(pool_size=51, name="max_pool_4", padding="same")(conv_4_layer)
max_pool_5_layer = layers.MaxPool1D(pool_size=50, name="max_pool_5", padding="same")(conv_5_layer)
flatten_3_layer = layers.Flatten()(max_pool_3_layer)
flatten_4_layer = layers.Flatten()(max_pool_4_layer)
flatten_5_layer = layers.Flatten()(max_pool_5_layer)
concatenate_layer = layers.concatenate([flatten_3_layer, flatten_4_layer, flatten_5_layer])
dropout_layer = layers.Dropout(rate=0.5)(concatenate_layer)
outputs = layers.Dense(2, activation="softmax")(dropout_layer)

model = keras.Model(inputs=inputs, outputs=outputs, name="test_model")
keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_data = load_text.train_data
test_data = load_text.test_data

print("FD")

model.fit(train_data, epochs=2, verbose=1)
test_loss, test_acc = model.evaluate(test_data, verbose=2)
print(test_acc)
