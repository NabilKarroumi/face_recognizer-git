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
            :param parent_directory: contains all images of all classes. Everything is mixed up here.
            :type parent_directory: str
        """
        self.parent_directory = parent_directory

        # Before building any dataset, ensure the directory is empty to avoid duplicates files.
        self.clearDirectoryBeforeBuildingDatasets()

    def generateDatasets(self, classes, training_set_size, validation_set_size, testing_set_size, critial_samples_number_in_smallest_class):
        """
            Generates/Builds training, validation and test sets used for training the model.

            :param classes: all classes of the classifier. In other words, all labels i.e. subdirectories we want in each dataset.
            :type classes: list(str)

            :param training_set_size: number of samples in the training set
            :type training_set_size: int

            :param validation_set_size: number of samples in the validation set
            :type validation_set_size: int

            :param testing_set_size: number of samples in the test set
            :type testing_set_size: int

            :param critial_samples_number_in_smallest_class: If all classes have not the same number of samples, the user have to precise the minimum amount of samples contained in the smallest class.
            :type critial_samples_number_in_smallest_class: int
        """

        ###################################
        ############# WARNING #############
        # - Choose carefully the size of each dataset, as all classes have necessarily not the same number of samples!
        # - If no choice is made by the user, a default one will be applied (see face_recognizer.src.back.utils.set_datasets_sizes function).
        ###################################

        # Ensure datasets sizes constraints are respected
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

                # NOTE:
                #   "random.sample chooses k elements from a sequence without replacement
                #   but here, the function is called three times and may
                #   select a same sample several times and put it in all training, validation and testing datasets
                #   which can create a bias in our future model !"

                #   However here, before calling back random.sample, we MOVE the samples selected during its previous call
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
