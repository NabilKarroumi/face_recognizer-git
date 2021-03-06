""" 
This module contains functions used to detect faces in an image.
Faces are detected using `MTCNN <https://github.com/ipazc/mtcnn>`_ .
"""

import cv2 as cv
import numpy as np
import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(
    0), user32.GetSystemMetrics(1)  # width, height


def show_img(img, title):
    """
        Displays an image using OpenCv2 built-in functions.

        :param img: image to display
        :type img: OpenCv2 instance, basically a np.array()

        :param title: title of the image to display.
        :type title: str
    """
    cv.imshow(title, img)
    cv.waitKey(1000)
    cv.destroyAllWindows()


def show_all_detected_faces(list_of_images):
    """
        Displays all images in list_of_images.
        See show_img() function.

        :param list_of_images: list of images to display
        :type list_of_images: list(np.array())
    """
    i = 0
    for img in list_of_images:
        show_img(img, 'face nb {}'.format(i))
        i += 1


def resize_img_to_fit_user_screen(img):
    """
        Resizes the image to fit monitor's screen (if the image is too big).

        :param img: image to display
        :type img: OpenCv2 instance, basically a np.array()
    """
    img = cv.resize(img, (screensize[1]//10, screensize[0]//10))


def is_empty(l):
    """
        Checks whether a list is empty or not.

        :param l: list of interest
        :param l: list()
    """
    if not bool(l):
        return True
    else:
        return False


def faces_detector(img_path, detector, img_size=(224, 224), threshold_confidence=0.90):
    """
        Detects faces in an image, using MTCNN.

        :param img_path: path to the image of interest.
        :type img_path: str

        :param detector: detector to use to detect faces
        :type detector: MTCNN instance

        :param img_size: desired size of image 
        :type img_size: tuple(int, int)

        :param threshold_confidence: threshold to decide whether a detection is considered as a face or not.
        :type threshold_confidence: float

    """
    all_detected_faces = []
    detection_status = 'success'

    # Read the image in color
    img = cv.imread(img_path, 1)
    if np.shape(img)[:2] > screensize:
        resize_img_to_fit_user_screen(img)

    detected_faces = detector.detect_faces(img)

    # Check whether faces are detected
    if is_empty(detected_faces):
        detection_status = 'failure'
    else:
        # At this stage, at least on face is detected
        for detected_face in detected_faces:
            # Check if the confidence is relevant
            if detected_face['confidence'] >= threshold_confidence:
                # the lower-left-corner is returned
                x_left, y_left, width, height = detected_face['box']
                x_right, y_right = x_left + width, y_left + height

                face = img[y_left:y_right, x_left:x_right]
                face = cv.resize(face, img_size)
                all_detected_faces.append(face)

    return all_detected_faces, detection_status
