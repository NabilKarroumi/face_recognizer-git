# coding: utf-8

import os
from face_recognizer.src.back.config import CFG, BuildDatasets_CFG
from face_recognizer.src.back.generate_train_validation_test_sets import TrainValidationTestDatasetsGenerator
from face_recognizer.src.back.utils import resize_multiple_images, rename_multiple_files, find_all_faces_in_multiple_img, count_files_in_subdirectories, set_datasets_sizes, delete_items_from_directory, remove_l2_from_l1, create_directory, copy_files


def process_data(current_working_directory):
    """
        Processes the data and organizes all folders needed to run the application.

        :param current_working_directory: path to the current working directory.
        :type current_working_directory: str
    """

    dataset_builder = BuildDatasets_CFG(current_working_directory)

    # If we need to detect faces in a photos, we'll have to import mtcnn.
    # The import is done here because it takes a lot of time. Hence, it must be done only if it is required.
    if dataset_builder.find_faces:
        import mtcnn
        detector = mtcnn.MTCNN()

    for one_class in dataset_builder.classes:

        one_class_path = os.path.join(
            dataset_builder.classes_path, one_class)
        create_directory(one_class_path)

        # Resizing all images
        if dataset_builder.resize_imgs:
            resize_multiple_images(
                one_class_path, one_class_path)

        # Renaming all images
        if dataset_builder.rename_imgs:
            rename_multiple_files(
                one_class_path, one_class.lower())

        # Exctracting faces from images
        if dataset_builder.find_faces:
            dst_path = os.path.join(
                dataset_builder.faces_path, one_class.lower())
            create_directory(dst_path)

            find_all_faces_in_multiple_img(
                one_class_path, detector, CFG.IMG_SIZE[:2], dst_path)

    # Generating training, testing, validation datasets
    if dataset_builder.generate_datasets:

        # Datasets sizes definition
        sizes = count_files_in_subdirectories(
            dataset_builder.faces_path)
        training_set_size = set_datasets_sizes(sizes)[0]
        validation_set_size = set_datasets_sizes(sizes)[1]
        testing_set_size = set_datasets_sizes(sizes)[2]
        critical_samples_number_in_smallest_class = min(sizes)

        create_directory(dataset_builder.datasets_path)

        datasets_generator = TrainValidationTestDatasetsGenerator(
            dataset_builder.datasets_path)  # instance creation here to empty the cwd before copying data

        # move data to split into training, validation, testing sets to the working directory
        if dataset_builder.copy_data_for_datasets_generation:
            copy_files(dataset_builder.faces_path,
                       dataset_builder.datasets_path)

        # generate datasets
        datasets_generator.generateDatasets(
            classes=dataset_builder.classes,
            training_set_size=training_set_size,
            validation_set_size=validation_set_size,
            testing_set_size=testing_set_size,
            critial_samples_number_in_smallest_class=critical_samples_number_in_smallest_class
        )

        # remove all folders except training, validation and testing sets
        l1 = os.listdir(dataset_builder.datasets_path)
        l2 = ['train', 'test', 'validation']
        move_to_junk = remove_l2_from_l1(l1, l2)
        delete_items_from_directory(
            dataset_builder.datasets_path, move_to_junk)


if __name__ == "__main__":
    cwd = r'D:\\Users\\KARROUMI Nabil\\Desktop\\ApprendrePython\\PROJECTS\\FacesRecognition\\Restructure\\tests'
    process_data(cwd)
