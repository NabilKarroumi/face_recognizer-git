# coding: utf-8

# import mtcnn
import os
# from save import read
# from collections import Counter


class CFG():

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # BASE_DIR = read('./working_directory.txt')

    # DETECTOR = mtcnn.MTCNN()
    FACE_DETECTION_CONFIDENCE = 0.90

    IMG_SIZE = (224, 224, 3)


class BuildDatasets_CFG():

    def __init__(self,
                 current_working_directory,
                 resize_imgs=False,
                 rename_imgs=False,
                 find_faces=True,
                 generate_datasets=True,
                 copy_data_for_datasets_generation=True):

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

    # # Variables Definition: To be asked in GUI ??
    # RESIZE_IMGS = False
    # RENAME_IMGS = False
    # FIND_FACES = True
    # GENERATE_DATASETS = True
    # COPY_DATA_FOR_DATASET_GENERATION = True

    # All paths needed
    # DATASETS_PATH = os.path.join(self.BASE_DIR, r'Datasets\family\datasets')

    # CLASSES_PATH = os.path.join(
    #     self.BASE_DIR, r'Datasets\family\initial_images')

    # CLASSES = os.listdir(CLASSES_PATH)

    # FACES_PATH = os.path.join(self.BASE_DIR, r'Datasets\family\faces')

    # sizes = find_number_of_elements_in_directories(FACES_PATH)
    # TRAINING_SET_SIZE = set_datasets_sizes(sizes)[0]
    # VALISATION_SET_SIZE = set_datasets_sizes(sizes)[1]
    # TESTING_SET_SIZE = set_datasets_sizes(sizes)[2]
    # CRITICAL_SAMPLES_NUMBER_IN_SMALLEST_CLASS = min(sizes)


class BuildModels_CFG(BuildDatasets_CFG):

    def __init__(self,
                 current_working_directory,
                 model_name):

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

    # # Paths to training/validation datasets
    # TRAINING_SET = os.path.join(
    #     self.BASE_DIR, r'./Datasets/family/datasets/train')
    # VALIDATION_SET = os.path.join(
    #     self.BASE_DIR, r'./Datasets/family/datasets/validation')
    # TESTING_SET = os.path.join(
    #     self.BASE_DIR, r'./Datasets/family/datasets/test')

    # OUTPUT_CLASSES_NUMBER = len(BuildDatasets_CFG.CLASSES)

    # TRAIN_GENERATOR_BATCH_SIZE = 5
    # VALIDATION_GENERATOR_BATCH_SIZE = 2
    # TEST_GENERATOR_BATCH_SIZE = 1

    # EPOCHS = 3

    # CLASS_MODE = 'categorical'

    # MODEL_SAVING_PATH = './models/custom_model'


# if __name__ == "__main__":
#     cfg = BuildModels_CFG(os.path.dirname(os.path.abspath(__file__)))
#     print(cfg.output_classes_number)
