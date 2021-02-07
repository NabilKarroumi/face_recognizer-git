# coding: utf-8

import os
import glob
import random
import shutil
from face_recognizer.src.back.utils import delete_items_from_directory


class TrainValidationTestDatasetsGenerator(object):

    def __init__(self, parent_directory):
        """
            /!\ IMPORTANT /!\ 
            parent_directory: contains all images of all classes. Everything is mixed here
        """
        self.parent_directory = parent_directory

        # Make sure the cwd is empty before genertaing data because new data might be added and all datasets must be
        # rebuilt from the beginning
        self.clearDirectoryBeforeBuildingDatasets()

    def generateDatasets(self, classes, training_set_size, validation_set_size, testing_set_size, critial_samples_number_in_smallest_class):
        """
            classes: all classes of the classifier. In other words, all labels i.e. subdirectories we want in each dataset.
                     It's a list of str
            critial_samples_number_in_smallest_class: If all classes have not the same number of samples, the user have to 
                     precise the minimum amount of samples contained in the smallest class.
        """

        ###################################
        # WARNING: Choose intelligently the size of each dataset, as all classes have not the same number of samples!
        ###################################

        # Make sure datasets sizes contraints are met
        assert critial_samples_number_in_smallest_class > training_set_size, 'The size of the training set should be lesser or equal than the maximum size of the smallest class !'
        assert training_set_size > validation_set_size, 'The size of the training set should be greater than the validation one !'
        assert training_set_size > testing_set_size, 'The size of the training set should be greater than the testing one !'

        # Ensure we are in the desired working directory
        os.chdir(self.parent_directory)

        # Ensure classes is a list of string with lower letters
        classes = [i.lower() for i in classes]

        # Create all datasets
        if os.path.isdir(os.path.join('train', classes[0])) is False:
            for one_class in classes:
                os.makedirs(os.path.join(
                    'train', one_class), exist_ok=True)
                os.makedirs(os.path.join(
                    'validation', one_class), exist_ok=True)
                os.makedirs(os.path.join(
                    'test', one_class), exist_ok=True)

                # One may say:
                #   "random.sample chooses k elements from a sequence without replacement
                #   but here, the function is called three times and may
                #   select a same sample several times and put it in all training, validation and testing datasets
                #   which can create a bias in our future model !"

                # I answer:
                #   That's true !
                #   But here, before calling back random.sample, we MOVE the samples selected during its previous call
                #   Hence, we ensure that no sample is duplicated in all training, validation and testing datasets.
                for sample in random.sample(glob.glob(one_class+'*'), training_set_size):
                    shutil.move(sample, os.path.join(
                        'train', one_class))

                for sample in random.sample(glob.glob(one_class+'*'), validation_set_size):
                    shutil.move(sample, os.path.join(
                        'validation', one_class))

                for sample in random.sample(glob.glob(one_class+'*'), testing_set_size):
                    shutil.move(sample, os.path.join(
                        'test', one_class))

    def clearDirectoryBeforeBuildingDatasets(self):
        """
            Clears self.parent_directory
        """
        delete_items_from_directory(
            self.parent_directory, os.listdir(self.parent_directory))


if __name__ == "__main__":
    # pass
    generator = TrainValidationTestDatasetsGenerator(r'./tests/datasets')
    generator.generateDatasets(['asmaa', 'nabil'], 3, 1, 1, 5)
