""" 
            Links and References 

Tutorial webpage: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
mtcnn: https://github.com/ipazc/mtcnn

"""

""" Import Librairies """

# The following library will be usefull to check whether an image is over-sized


# from config import CFG
import cv2 as cv  # to manipulate images
import numpy as np
import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(
    0), user32.GetSystemMetrics(1)  # width, height


""" Function definition """
"""
    defining a function which:
        - takes an image as parameter
        - checks for faces in it
        - if faces are detected:
            - resize the image containing the faces (224x224) to match models inputs
            - return all the resized faces as array
"""


def show_img(img, title):
    cv.imshow(title, img)
    cv.waitKey(1000)
    cv.destroyAllWindows()


def show_all_detected_faces(list_of_images):
    i = 0
    for img in list_of_images:
        show_img(img, 'face nb {}'.format(i))
        i += 1


def resize_img_to_fit_user_screen(img):
    img = cv.resize(img, (screensize[1]//10, screensize[0]//10))


def is_empty(l):
    if not bool(l):
        return True
    else:
        return False


def faces_detector(img_path, detector, img_size=(224, 224), threshold_confidence=0.90):
    all_detected_faces = []
    detection_status = 'success'
    # detector = mtcnn.MTCNN()

    # Read the image in color
    img = cv.imread(img_path, 1)
    if np.shape(img)[:2] > screensize:
        resize_img_to_fit_user_screen(img)

    # Check whether faces are detected
    # a list of dict containing several information (face position, eyes position...)
    detected_faces = detector.detect_faces(img)

    # assert bool(detected_faces), 'No faces detected in the image {}'.format(
    #     img_path)  # assert is raised when the condition is False

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


if __name__ == "__main__":
    img_path = './Datasets/Baby_Face.jpg'
    # img_path = './Datasets/montagne6.jpg'
    # img_path = './Datasets/football_team.jpg'
    detector = mtcnn.MTCNN()
    faces, detection_status = faces_detector(img_path, detector)
    show_all_detected_faces(faces)
