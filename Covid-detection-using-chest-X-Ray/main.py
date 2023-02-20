import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet201, ResNet101V2
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

data_location = './data'
CLASS_NAMES = ['covid-19', 'healthy']
IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 8
EPOCHS = 10
DATASET_SIZE = sum([len(files) for r, d, files in os.walk(data_location)])


class DataProcessor():
    def __init__(self, data_location):
        self.labeled_dataset = tf.data.Dataset.list_files(f"{data_location}/*/*")

    def _get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == CLASS_NAMES

    def _decode_image(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [IMAGE_SHAPE[0], IMAGE_SHAPE[1]])

    def _pre_proces_images(self, file_path):
        label = self._get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self._decode_image(img)
        return img, label

    def prepare_dataset(self):
        self.labeled_dataset = self.labeled_dataset.map(self._pre_proces_images)
        self.labeled_dataset = self.labeled_dataset.cache()
        self.labeled_dataset = self.labeled_dataset.shuffle(buffer_size=10)
        self.labeled_dataset = self.labeled_dataset.repeat()
        self.labeled_dataset = self.labeled_dataset.batch(BATCH_SIZE)
        self.labeled_dataset = self.labeled_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_size = int(0.7 * DATASET_SIZE)
        val_size = int(0.15 * DATASET_SIZE)
        test_size = int(0.15 * DATASET_SIZE)

        train_dataset = self.labeled_dataset.take(train_size)
        test_dataset = self.labeled_dataset.skip(train_size)
        val_dataset = test_dataset.skip(test_size)
        test_dataset = test_dataset.take(test_size)

        return train_dataset, test_dataset, val_dataset


processor = DataProcessor(data_location)
train_dataset, test_dataset, val_dataset = processor.prepare_dataset()


class Wrapper(tf.keras.Model):
    def __init__(self, base_model):
        super(Wrapper, self).__init__()

        self.base_model = base_model
        self.average_pooling_layer = AveragePooling2D(name="polling")
        self.flatten = Flatten(name="flatten")
        self.dense = Dense(64, activation="relu")
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(2, activation="softmax")

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.average_pooling_layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output


base_learning_rate = 0.0001
steps_per_epoch = DATASET_SIZE // BATCH_SIZE
validation_steps = 20

mobile_net = MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
mobile_net.trainable = False
mobile = Wrapper(mobile_net)
mobile.compile(Adam(lr=base_learning_rate),
               loss='binary_crossentropy',
               metrics=['accuracy'])

res_net = ResNet101V2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
res_net.trainable = False
res = Wrapper(res_net)
res.compile(optimizer=Adam(lr=base_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy'])

dense_net = DenseNet201(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
dense_net.trainable = False
dense = Wrapper(dense_net)
dense.compile(optimizer=Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_mobile = mobile.fit(train_dataset,
                            epochs=EPOCHS,
                            validation_data=val_dataset,
                            validation_steps=validation_steps)

history_resnet = res.fit(train_dataset,
                         epochs=EPOCHS,
                         validation_data=val_dataset,
                         validation_steps=validation_steps)

history_densenet = dense.fit(train_dataset,
                             epochs=EPOCHS,
                             validation_data=val_dataset,
                             validation_steps=validation_steps)

plt.plot(history_mobile.history['accuracy'])
plt.plot(history_mobile.history['val_accuracy'])
plt.title('Model accuracy - Mobile Net')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_mobile.history['loss'])
plt.plot(history_mobile.history['val_loss'])
plt.title('Model loss - Mobile Net')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_resnet.history['accuracy'])
plt.plot(history_resnet.history['val_accuracy'])
plt.title('Model accuracy - ResNet')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_resnet.history['loss'])
plt.plot(history_resnet.history['val_loss'])
plt.title('Model loss - ResNet')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_densenet.history['accuracy'])
plt.plot(history_densenet.history['val_accuracy'])
plt.title('Model accuracy - Dense Net')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history_densenet.history['loss'])
plt.plot(history_densenet.history['val_loss'])
plt.title('Model loss - DenseNet')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

loss, accuracy = mobile.evaluate(test_dataset, steps=validation_steps)

print("--------MobileNet---------")
print("Loss: {:.2f}".format(loss))
print("Accuracy: {:.2f}".format(accuracy))
print("---------------------------")

loss, accuracy = res.evaluate(test_dataset, steps=validation_steps)

print("--------ResNet---------")
print("Loss: {:.2f}".format(loss))
print("Accuracy: {:.2f}".format(accuracy))
print("---------------------------")

loss, accuracy = dense.evaluate(test_dataset, steps=validation_steps)

print("--------DenseNet---------")
print("Loss: {:.2f}".format(loss))
print("Accuracy: {:.2f}".format(accuracy))
print("---------------------------")

tf.saved_model.save(mobile, './models/mobilenet/1')
tf.saved_model.save(res, './models/resnet/1')
tf.saved_model.save(dense, './models/densenet/1')
