# coding: utf-8

import shutil
import re
import os
import glob
import pickle as pic
import cv2 as cv
from face_recognizer.src.back.face_recognition import faces_detector


def write(filename, data):
    """ 
    This function saves variables
    """
    with open(filename, "wb") as f:
        pic.dump(data, f)


def read(filename):
    """ 
    This function loads the variables saved by the function "write"
    """
    with open(filename, "rb") as f:
        data = pic.load(f)
    return data


# def show_img(img, title):
#     cv.imshow(title, img)
#     cv.waitKey(1000)
#     cv.destroyAllWindows()


def files_in_dir(dir_path):
    # print(dir_path)
    # print(os.listdir(dir_path))
    files = [f for f in os.listdir(
        dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return files


def find_specific_file_extension_in_dir(dir_path, extension):
    """
        extension type: .txt / .png etc...
    """
    return glob.glob(os.path.join(dir_path, r'*{}'.format(extension)))[0].replace('\\', '/').split('/')[-1]


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def remove_l2_from_l1(l1, l2):
    """
        Returns all elements of l1 that are not in l2.
    """
    return [element for element in l1 if element not in l2]


def copy_files(src_path, dst_path):
    """
        Copies files of src_path in dst_path
        src_path and dst_path are folders
    """
    for folder in os.listdir(src_path):
        for file in os.listdir(os.path.join(src_path, folder)):
            source = os.path.join(os.path.join(src_path, folder), file)
            dest = os.path.join(dst_path, file)
            shutil.copy(source, dest)


def delete_items_from_directory(dir_path, list_items):
    """
        Removes all items (files and/or directories) in items_list from the directory located at dir_path
    """
    # for item in os.listdir(dir_path):
    #     if item in list_items:
    #         # os.path.isitem(os.path.join(dir_path, item)):
    #         os.remove(os.path.join(dir_path, item))

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if filename in list_items:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def resize_multiple_images(src_path, dst_path, img_size=None):
    # Here src_path is the location where raw images are saved.
    # Here dst_path is the location where resized images are saved: MUST EXISTS BEFORE FUNCTION CALL
    # Here img_siez is the size of an image (tuple of length 2).

    # When src_path and dst_path are the same, images are overwritten
    for filename in os.listdir(src_path):
        try:
            img = cv.imread(os.path.join(src_path, filename))

            if img_size is None:
                # img_size = (np.shape(img)[1]//4, np.shape(img)[0]//4)
                # this shape should be the same as the one of the window used to take photos
                img_size = (640, 480)

            new_img = cv.resize(img, img_size)

            # create_directory(dst_path)

            filename = filename.split(sep='.')[0]+'.jpg'
            cv.imwrite(os.path.join(dst_path, filename), new_img)
            print('Resized and saved {} successfully.'.format(filename))
        except:
            continue


def rename_multiple_files(path, obj):
    # path = <Enter the path of objects to be renamed >
    # obj = <Enter the prefix to be added to each file. For ex. car, bike, cat, dog, etc. >
    i = 0
    for filename in os.listdir(path):
        try:
            f, extension = os.path.splitext(os.path.join(path, filename))
            src = os.path.join(path, filename)
            dst = os.path.join(path, obj+'_{:04}'.format(i)+extension)
            os.rename(src, dst)
            i += 1
            print('Rename successful.')
        except:
            i += 1


def count_files_in_one_directory(path_directory):

    files_list = files_in_dir(path_directory)
    return len(files_list)


def count_files_in_subdirectories(path_parent_directory):

    counter = []

    for subdirectory in os.listdir(path_parent_directory):
        files_list = files_in_dir(os.path.join(
            path_parent_directory, subdirectory))

        counter.append(len(files_list))

    return counter


def find_all_faces_in_one_img(img_path, detector, img_size, dst_path):
    """
        img_path: path of the image
        detector: mtcnn instance (capable of finding a face)
    """

    img_path = img_path.replace('\\', '/')
    img_name = img_path.split(sep='/')[-1].split(sep='.')[0]

    assert img_path.split(sep='/')[-1].split(sep='.')[1] in [
        'png', 'jpg'], 'files should be images with a ".jpg" or ".png" extension !'

    img_extension = '.jpg'

    # create_directory(dst_path)

    all_detected_faces, detection_status = faces_detector(
        img_path, detector, img_size, threshold_confidence=0.90)

    if detection_status == 'success':
        for faces in all_detected_faces:
            # cv.imwrite(os.path.join(dst_path, img_name +
            #                         str(all_detected_faces.index(faces)) + img_extension), faces)
            cv.imwrite(os.path.join(dst_path, img_name + img_extension), faces)
    elif detection_status == 'failure':
        os.remove(img_path)

    return detection_status


def find_all_faces_in_multiple_img(img_dir_path, detector, img_size, dst_path):
    """
        img_path: path of the image
        detector: mtcnn instance (capable of finding a face)

        WARNING: All images must have been renamed with respect to the convention set in the function 'rename_multiple_file' (e.g: 'name_000i.jpg')
        The function finds faces ONLY from the image indexed len(dst_path).
        This allows the user to add more when he desired without finding faces in ALL images everytime. The function will focus on new data only.
        If len(dst_path) == 0 (no old data found) the function will still work. 
    """

    number_of_faces_already_found = count_files_in_one_directory(dst_path)
    print('number_of_faces_already_found', number_of_faces_already_found)
    for filename in os.listdir(img_dir_path):
        try:
            # When the user decides to add new data to existing one, no need to deal again (refind faces) with the old data
            if list(map(int, re.findall(r'\d+', filename)))[0] < number_of_faces_already_found:
                continue
            img_path = os.path.join(img_dir_path, filename)
            print(filename + ': IN PROGRESS')
            detection_status = find_all_faces_in_one_img(
                img_path, detector, img_size, dst_path)
            print(filename + ': {}\n'.format(detection_status.upper()))
        except:
            continue


def set_datasets_sizes(list_of_sizes):
    """
    list_of_sizes: contains all sizes of directories of interest
    """
    minimum = min(list_of_sizes)

    train = int(minimum*0.7)
    valid = int(minimum*0.2)
    test = int(minimum*0.1)
    gap = minimum - (train + valid + test)

    if gap > 0 and test != 0:
        valid += gap
    elif gap > 0 and test == 0:
        test += gap
    elif gap < 0:
        train -= gap

    return train, valid, test


if __name__ == "__main__":
    # pass
    # img_size = (224, 224)

    # src_path_train = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\Train"
    # src_path_val = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\Validation"

    # dst_path_train = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\all_cars_preprocessed"
    # dst_path_val = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\Validation_preprocessed"

    # src_path_test = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\ttest\motorbike"
    # dst_path_test = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\test\motorbike"

    # resize_multiple_images(src_path_test, dst_path_test, img_size)

    # path = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\celebrities\willis_extension_jfif - Copy"
    # rename_multiple_files(path, 'jason')
    # 7/0

    # path_to_train_set = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\train"
    # path_to_test_set = r"D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\validation"
    # X_train, y_train = split_data_into_train_and_validation(
    #     get_data(path_to_train_set))
    # X_val, y_val = split_data_into_train_and_validation(
    #     get_data(path_to_test_set))

    # save.write('./data', {'X_train': X_train,
    #                       'y_train': y_train, 'X_val': X_val, 'y_val': y_val})
    # print('X_train set : ', X_train)
    # print('y_train set : ', y_train)
    # print('X_val set : ', X_val)
    # print('y_val set : ', y_val)

    # img_path = r'D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\celebrities\jason-statham\jason-statham29.jfif'
    # img_dir_path = r'D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\celebrities\bruce-willis'
    # detector = mtcnn.MTCNN()
    # img_size = (224, 224)
    # dst_path = r'D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\AI_models\Datasets\celebrities\bruce-willis\testeuh'
    # obj

    # img = cv.imread(img_path, 1)
    # show_img(img, 'JS29')
    # img = cv.resize(img, img_size)
    # show_img(img, 'JS29')
    # 7/0

    # find_resize_rename_and_save_all_faces_from_one_img(img_path, detector, img_size, dst_path)

    # find_resize_rename_and_save_all_faces_from_multiple_img(
    # img_dir_path, detector, img_size, dst_path)

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # # KALTOUM_DIR = os.path.join(BASE_DIR, r'Datasets\family\Kaltoum')
    # # KALTOUM_DIR_PREPROCESSED = os.path.join(KALTOUM_DIR, 'kaltoum')

    # # KALTOUM_IMAGES = [os.path.join(KALTOUM_DIR, img)
    # #                   for img in files_in_dir(KALTOUM_DIR)]

    # # KALTOUM_IMAGES_PREPROCESSED = [os.path.join(KALTOUM_DIR_PREPROCESSED, img)
    # #                                for img in files_in_dir(KALTOUM_DIR_PREPROCESSED)]

    # # resize_multiple_images(KALTOUM_DIR, KALTOUM_DIR_PREPROCESSED)
    # # rename_multiple_files(KALTOUM_DIR_PREPROCESSED, 'kaltoum')

    # family_members_path = os.path.join(BASE_DIR, r'Datasets\family')
    # family_members = os.listdir(os.path.join(BASE_DIR, r'Datasets\family'))
    # for family_member in family_members:

    #     family_member_path = os.path.join(family_members_path, family_member)
    #     family_member_preprocessed_path = os.path.join(
    #         family_members_path, family_member.lower())

    #     # family_member_images = [os.path.join(
    #     #     family_member_path, img) for img in files_in_dir(family_member_path)]
    #     # family_member_images = [os.path.join(
    #     # family_member_preprocessed_path, img) for img in files_in_dir(family_member_preprocessed_path)]
    #     print(family_member_path)
    #     print(family_member_preprocessed_path)
    #     7/0
    #     resize_multiple_images(
    #         family_member_path, family_member_preprocessed_path)
    #     rename_multiple_files(
    #         family_member_preprocessed_path, family_member.lower())

    # find_all_faces_in_one_img('./toto.txt',
    #   None, None, None)

    # name = 'Toto'
    # resize_multiple_images('./test/'+name, './test/' +
    #                        name.lower(), img_size=(100, 100))

    print(count_files_in_subdirectories('./tests/faces'))
