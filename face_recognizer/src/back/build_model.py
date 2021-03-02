"""
This module contains all the functions used to build and train a deep learning model capable of identifying people's faces.
Transfert learning has been used to increase the model's performances.
The dataset used for training the models is `VGGFace2 <https://www.arxiv-vanity.com/papers/1710.08092/>`_. However, VGGFace2 has become the name to refer to the pre-trained models which has been trained on the dataset.
`ResNet50 <https://iq.opengenus.org/resnet50-architecture/>`_  architecture is used to train the model.
"""

import os
import cv2 as cv
import numpy as np

from face_recognizer.src.back.config import CFG, BuildModels_CFG
from face_recognizer.src.back.utils import create_directory, write

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def input_preprocessing(img):
    """
        Pre-processes the images before feeding them into the model.

        :param img: the image to pre-process
        :type img: OpenCv2 instance, basically a np.array()

        :return: the image pre-processed
        :rtype: OpenCv2 instance, basically a np.array()
    """
    # Ensure that the image size is CFG.IMG_SIZE
    img = cv.resize(img, CFG.IMG_SIZE[:-1])

    # Ensure that the elemets in the image are float32
    img = img.astype('float32')

    # the input layer shape in the model is: (None, 224, 224, 3)
    # Our input layer is (224, 224, 3)
    # We expand our dimension in the 0-th position (where the None is)
    img = np.expand_dims(img, axis=0)

    # the pixels values of the face image must be centered on each channel using the mean from the training dataset.
    # this mean is known by the preprocess_input function
    # version=2 refers to VGGFaces2 and not VGGFaces1
    img = preprocess_input(img, version=2)

    return img


def train_model(current_working_directory, model_name, verbose=False):
    """
        creates and trains the final model used to recognize faces.

        :param current_working_directory: path to the Current Working Directory (CWD).
        :param current_working_directory: str

        :param model_name: name of the model.
        :param model_name: str

        :param verbose: Is True, plots the training history
        :param verbose: bool
    """

    model_builder = BuildModels_CFG(
        current_working_directory, model_name)

    # instanciation of the image generator object
    data_generator = ImageDataGenerator(
        preprocessing_function=input_preprocessing)  # we specify the pre-processing function which will be applied to all images

    # train generator definition
    train_generator = data_generator.flow_from_directory(
        directory=model_builder.training_set,
        target_size=CFG.IMG_SIZE[:2],
        classes=model_builder.classes,
        batch_size=model_builder.train_generator_batch_size,
        class_mode=model_builder.class_mode
    )

    # validation generator definition
    validation_generator = data_generator.flow_from_directory(
        directory=model_builder.validation_set,
        target_size=CFG.IMG_SIZE[:2],
        classes=model_builder.classes,
        batch_size=model_builder.validation_generator_batch_size,
        class_mode=model_builder.class_mode
    )

    # test generator definition
    test_generator = data_generator.flow_from_directory(
        directory=model_builder.testing_set,
        target_size=CFG.IMG_SIZE[:2],
        classes=model_builder.classes,
        batch_size=model_builder.test_generator_batch_size,
        class_mode=model_builder.class_mode,
        shuffle=False
    )

    # instanciation of the pre-trained model
    vggface = VGGFace(model='resnet50', input_shape=CFG.IMG_SIZE,
                      include_top=False, pooling='avg')

    # customization of the pre-trained model
    custom_model = Sequential()  # create a new Sequential model
    # plug in the pre-trained model, without its last layer
    custom_model.add(vggface)
    custom_model.add(
        Dense(units=model_builder.output_classes_number, activation='softmax'))  # add a custom output layer

    # do not re-train the initial parameters of the pre-trained model
    custom_model.layers[0].trainable = False

    # set the model's compilation parameters
    custom_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit the model to the data (training phase)
    r = custom_model.fit(x=train_generator, epochs=model_builder.epochs,
                         validation_data=validation_generator, verbose=2)

    if verbose:
        import matplotlib.pyplot as plt

        plt.plot(r.history['loss'], label='training loss')
        plt.plot(r.history['val_loss'], label='validation loss')
        plt.legend(loc='best')
        plt.show()

        plt.plot(r.history['accuracy'], label='training accuracy')
        plt.plot(r.history['val_accuracy'], label='validation accuracy')
        plt.legend(loc='best')
        plt.show()

        predictions = custom_model.predict(x=test_generator, verbose=0)

        for element in range(len(predictions)):
            print("human readable real test labels : {}".format(
                model_builder.classes[test_generator.classes[element]]))
            print("human readable predicted test labels : {}".format(
                model_builder.classes[np.argmax(predictions[element])]))
            print('\n')

    # save the trained model (basically, its name and parameters)
    create_directory(model_builder.default_model_path)
    if os.path.isfile(model_builder.model_saving_path) is False:
        custom_model.save(model_builder.model_saving_path)
        write(os.path.join(model_builder.default_model_path,
                           'names_list.txt'), model_builder.classes)

    return model_builder.model_saving_path
