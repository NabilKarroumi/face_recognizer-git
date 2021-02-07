# coding: utf-8

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
        VGGFaces inputs should be preprocessed the same way
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

    model_builder = BuildModels_CFG(
        current_working_directory, model_name)

    # 1. Create the generator object and we specify how the images are in the VGGFaces2 model
    data_generator = ImageDataGenerator(
        preprocessing_function=input_preprocessing)

    # 2. Training generation
    train_generator = data_generator.flow_from_directory(
        directory=model_builder.training_set,
        target_size=CFG.IMG_SIZE[:2],
        classes=model_builder.classes,
        batch_size=model_builder.train_generator_batch_size,
        class_mode=model_builder.class_mode
    )

    # 3. Validation generation
    validation_generator = data_generator.flow_from_directory(
        directory=model_builder.validation_set,
        target_size=CFG.IMG_SIZE[:2],
        classes=model_builder.classes,
        batch_size=model_builder.validation_generator_batch_size,
        class_mode=model_builder.class_mode
    )

    # 3. Testing generation
    test_generator = data_generator.flow_from_directory(
        directory=model_builder.testing_set,
        target_size=CFG.IMG_SIZE[:2],
        classes=model_builder.classes,
        batch_size=model_builder.test_generator_batch_size,
        class_mode=model_builder.class_mode,
        shuffle=False
    )

    # imgs, labels = next(train_generator)

    # batch_size = 5
    # for element in range(batch_size):
    #     print("image n° " + str(element))
    #     plt.imshow(imgs[element])
    #     plt.show()

    #     print("related label : {}".format(labels[element]))
    #     print("human readable label : {}".format(
    #         model_builder.classes[np.argmax(labels[element])]))

    # 1. get the original model without the last layer
    vggface = VGGFace(model='resnet50', input_shape=CFG.IMG_SIZE,
                      include_top=False, pooling='avg')
    # vggface.summary()

    # 2. create a custom model
    custom_model = Sequential()
    custom_model.add(vggface)
    custom_model.add(
        Dense(units=model_builder.output_classes_number, activation='softmax'))
    # custom_model.summary()

    # Do not train all parameters, only those of the last layer
    custom_model.layers[0].trainable = False
    # custom_model.summary()

    # 3. train the model
    custom_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

        # test_generator.classes

        predictions = custom_model.predict(x=test_generator, verbose=0)
        # print(np.round(predictions))

        for element in range(len(predictions)):
            print("human readable real test labels : {}".format(
                model_builder.classes[test_generator.classes[element]]))
            print("human readable predicted test labels : {}".format(
                model_builder.classes[np.argmax(predictions[element])]))
            print('\n')

    # Models and classes saving (to make the next use of the tool easier and faster)
    create_directory(model_builder.default_model_path)
    if os.path.isfile(model_builder.model_saving_path) is False:
        # os.makedirs(model_builder.model_saving_path, exist_ok=True)
        custom_model.save(model_builder.model_saving_path)
        write(os.path.join(model_builder.default_model_path,
                           'names_list.txt'), model_builder.classes)

    return model_builder.model_saving_path


if __name__ == "__main__":

    current_working_directory = './tests'
    model_name = 'my_model'
    model_saving_path = train_model(
        current_working_directory, model_name, verbose=True)
