# coding: utf-8

import os


class CFG():
    """
        Main configuration class.
    """

    # threshold to decide whether a MTCNN detection is considered as a face.
    FACE_DETECTION_CONFIDENCE = 0.90

    # images sizes.
    IMG_SIZE = (224, 224, 3)


class BuildDatasets_CFG():
    """
        Class configuring the creation of datasets (i.e. training, validation and test set).
    """

    def __init__(self,
                 current_working_directory,
                 resize_imgs=False,
                 rename_imgs=False,
                 find_faces=True,
                 generate_datasets=True,
                 copy_data_for_datasets_generation=True):
        """
            Dataset instance constructor.

            :param current_working_directory: path the to Current Working Directory (CWD).
            :param current_working_directory: str

            :param resize_imgs: If True, resizes the images (see face_recognizer.src.back.utils.resize_multiple_images function).
            :param resize_imgs: bool

            :param rename_imgs: If True, renames the images (see face_recognizer.src.back.utils.rename_multiple_files function).
            :param rename_imgs: bool

            :param find_faces: If True, finds faces in images (see face_recognizer.src.back.utils.find_all_faces_in_multiple_img function).
            :param find_faces: bool

            :param generate_datasets: If True, generates training, validation, test sets (see face_recognizer.src.back.generate_train_validation_test_sets.TrainValidationTestDatasetsGenerator class)
            :param generate_datasets: bool

            :param copy_data_for_datasets_generation: If True, copies all faces found from 'faces' folder to 'datasets' folder in order to create the final training, validation, test sets later on.
            :param copy_data_for_datasets_generation: bool
        """

        # Data processing configuration
        self.resize_imgs = resize_imgs
        self.rename_imgs = rename_imgs
        self.find_faces = find_faces
        self.generate_datasets = generate_datasets
        self.copy_data_for_datasets_generation = copy_data_for_datasets_generation

        # Paths definition
        self.current_working_directory = current_working_directory
        self.datasets_path = os.path.join(
            self.current_working_directory, r'datasets')
        self.classes_path = os.path.join(
            self.current_working_directory, r'initial_images')
        self.faces_path = os.path.join(
            self.current_working_directory, r'faces')

        # Classes definition
        self.classes = os.listdir(self.classes_path)


class BuildModels_CFG(BuildDatasets_CFG):

    def __init__(self,
                 current_working_directory,
                 model_name):
        """
        Dataset instance constructor.

            : param current_working_directory: path the to Current Working Directory(CWD).
            : param current_working_directory: str

            : param model_name: name of the model.
            : param model_name: str
        """

        # Call back BuildDatasets_CFG Constructor to access self.classes
        super().__init__(current_working_directory)

        # Paths definition
        self.current_working_directory = current_working_directory
        self.model_name = model_name
        self.training_set = os.path.join(
            self.current_working_directory, r'datasets/train')
        self.validation_set = os.path.join(
            self.current_working_directory, r'datasets/validation')
        self.testing_set = os.path.join(
            self.current_working_directory, r'datasets/test')

        # Number of output classes
        # classes from BuildDatasets_CFG
        self.output_classes_number = len(self.classes)

        # Learning parameters configuration
        self.train_generator_batch_size = 5
        self.validation_generator_batch_size = 2
        self.test_generator_batch_size = 1
        self.epochs = 5
        self.class_mode = 'categorical'

        # Models saving path
        self.default_model_path = os.path.join(
            self.current_working_directory, 'default')
        self.model_saving_path = os.path.join(
            self.default_model_path, self.model_name + '.h5')
