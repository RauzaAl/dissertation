import numpy as np
from keras import utils
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import csv


class Classifier:
    BATCH_SIZE = 32
    EPOCHS = 150

    def __init__(self):
        self.X = None
        self.y = None

        self.X_test = None
        self.y_test = None

        self.model = None

        self.history = None

    def set_train_data(self, X, y):
        self.X = X
        self.y = y

    def set_test_data(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def set_model(self, model):
        self.model = model

    def load_model(self, model_path: str):
        self.model.load_weights(model_path)

    def train(self, output_model: str, classes=43):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=0.3, random_state=42, shuffle=True,
        )

        y_train = utils.to_categorical(y_train, classes)
        y_val = utils.to_categorical(y_val, classes)

        aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest",
        )

        self.history = self.model.fit(
            aug.flow(X_train, y_train, batch_size=self.BATCH_SIZE),
            epochs=self.EPOCHS,
            validation_data=(X_val, y_val),
        )

        self.model.save(output_model)

    def import_metrics(self, file_name: str):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'acc', 'val_acc', 'loss', 'val_loss'])
            for i in range(len(self.history.history['accuracy'])):
                writer.writerow(
                    [i+1,
                     self.history.history['accuracy'][i],
                     self.history.history['val_accuracy'][i],
                     self.history.history['loss'][i],
                     self.history.history['val_loss'][i],
                     ])

    def test(self):
        predict_x = self.model.predict(self.X_test)
        pred = np.argmax(predict_x, axis=1)

        return accuracy_score(self.y_test, pred) * 100

    def test_one(self, image_path: str):
        data = []
        image = Image.open(image_path).resize((32, 32))
        data.append(np.array(image))
        X_test = np.array(data)

        predict = self.model.predict(X_test)

        Y_pred = np.argmax(predict, axis=1)
        return Y_pred.item()
