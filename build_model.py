import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import load_text
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

train_data = load_text.train_data
test_data = load_text.test_data
val_data = load_text.val_data

vocab_size = load_text.vocab_size
embedding_matrix = load_text.embedding_matrix

embedding_matrix = tf.keras.initializers.Constant(embedding_matrix)

inputs = keras.Input(shape=(54,), name="input_text")
embedding_layer_static = layers.Embedding(vocab_size, 300, embeddings_initializer=embedding_matrix,
                                          name="embedding_static",
                                          trainable=False)(inputs)
embedding_layer_non_static = layers.Embedding(vocab_size, 300, embeddings_initializer=embedding_matrix,
                                              name="embedding_non_static",
                                              trainable=True)(inputs)
reshape_static = layers.Reshape((54, 300, 1))(embedding_layer_static)
reshape_non_static = layers.Reshape((54, 300, 1))(embedding_layer_non_static)
embedding_layer = layers.concatenate([reshape_static, reshape_non_static], axis=3)
conv_3_layer = layers.Conv2D(100, (3, 300), activation='relu', name="filter_size_3")(embedding_layer)
conv_4_layer = layers.Conv2D(100, (4, 300), activation='relu', name="filter_size_4")(embedding_layer)
conv_5_layer = layers.Conv2D(100, (5, 300), activation='relu', name="filter_size_5")(embedding_layer)
max_pool_3_layer = layers.MaxPool2D(pool_size=(52, 1), name="max_pool_3")(conv_3_layer)
max_pool_4_layer = layers.MaxPool2D(pool_size=(51, 1), name="max_pool_4")(conv_4_layer)
max_pool_5_layer = layers.MaxPool2D(pool_size=(50, 1), name="max_pool_5")(conv_5_layer)
flatten_3_layer = layers.Flatten()(max_pool_3_layer)
flatten_4_layer = layers.Flatten()(max_pool_4_layer)
flatten_5_layer = layers.Flatten()(max_pool_5_layer)
concatenate_layer = layers.concatenate([flatten_3_layer, flatten_4_layer, flatten_5_layer])
# dropout_layer = layers.Dropout(rate=0.6)(concatenate_layer)
outputs = layers.Dense(2, activation="softmax")(concatenate_layer)

model = keras.Model(inputs=inputs, outputs=outputs, name="test_model")
keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.load_weights('C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/TensorFlow/data')

model.fit(train_data,
          epochs=200,
          validation_data=val_data,
          callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min'),
          verbose=1, shuffle=True)

# model.save_weights('C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/TensorFlow/data')
# test_loss, test_acc = model.evaluate(test_data, verbose=2)

predict_data = load_text.predict_data
predict_data = predict_data.batch(batch_size=54)

predictions = model.predict(predict_data)
print(np.argmax(predictions, axis=1))
