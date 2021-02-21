# coding: utf-8

import cv2
import numpy as np
from face_recognizer.src.back.build_model import input_preprocessing
from tensorflow.keras.models import load_model


def main(model_saving_path, classes):
    """
        Sets up and Starts the final application.

        :param model_saving_path: path to the model's parameters. At this stage, the model is already trained and is able to recognize faces.
        :type model_saving_path: str

        :param classes: names of the persons the model should recognize.
        :type classes: list(str)

    """

    # Model
    model = load_model(model_saving_path)

    # Create a Video capture
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # previous_time = 0
    # current_time = time()

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    window_name = 'My Window'
    # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(
    #     window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Loop used to show all images of the video
    while True:
        try:
            # Read the video
            check, frame = video.read()

            # Convert the image in gray, just to make faster the detectMultiScale
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Search face in picture
            faces = face_cascade.detectMultiScale(
                gray_img, scaleFactor=1.10, minNeighbors=10)

            for x, y, w, h in faces:
                person_identified = model.predict(
                    input_preprocessing(frame[y:y+h, x:x+w]))

                frame = cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                confidence = str(
                    np.round(np.max(person_identified[0])*100, 2)) + '%'
                text = classes[np.argmax(
                    person_identified[0])] + ' ~ ' + confidence
                text_font = cv2.FONT_HERSHEY_COMPLEX
                text_font_size = 1
                text_thickness = 2
                (text_width, text_height) = cv2.getTextSize(
                    text, text_font, text_font_size, text_thickness)[0]

                cv2.putText(
                    img=frame,
                    text=text,
                    # org=(y, x + w//2 - text_width//2),
                    org=(x + w//2 - text_width//2, y),
                    fontFace=text_font,
                    fontScale=text_font_size,
                    color=(0, 0, 255),
                    thickness=text_thickness)

            # Show the actual frame
            cv2.imshow(window_name, frame)

            # Create an other frame after 1 ms
            key = cv2.waitKey(1)

            # Quit the video capturing
            if key == ord('q'):
                break

            # previous_time = current_time
            # current_time = time()
            # fps = 1 / (current_time - previous_time)
            # print('fps: ', fps)

        except Exception as e:
            print('Issue found: {}'.format(e))

    # Release the video
    video.release()

    # Destroy windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import os
    # print(os.getcwd())
    # print(os.path.isfile("haarcascade_frontalface_default.xml"))
    # print("!ls .")
    # 7/0

    from save import read
    current_working_directory = r'D:\\Users\\KARROUMI Nabil\\Desktop\\ApprendrePython\\PROJECTS\\FacesRecognition\\Restructure\\tests'

    names = read(os.path.join(current_working_directory,
                              os.path.join('default', 'names_list.txt')))
    # names.sort()
    model = os.path.join(current_working_directory,
                         os.path.join('default', 'my_model.h5'))

    main(model, names)
