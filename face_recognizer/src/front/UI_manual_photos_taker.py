"""
This module implementes an interface window that allows the user to create/generate photos (data) manually.
"""

import os
import sys
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from face_recognizer.src.back.process_data import process_data
from face_recognizer.raw_UIs.manual_photos_taker import Ui_Manual_photos_taker


class VideoThread(QtCore.QThread):
    """
        Video Thread Class.

        NOTE:
            This Class has been taken from: https://github.com/docPhil99/opencvQtdemo and cutomized.
    """
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cv_img = None

    def run(self):
        """
            Runs the video thread
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                self.cv_img = cv_img
        # shut down capture system
        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        """
            Stops the video thread
        """
        self._run_flag = False
        self.wait()


class CustomManualPhotosTakerWindow(Ui_Manual_photos_taker):
    """
        Custom Manual Photos Taker window Class.
    """

    def __init__(self, names, current_working_directory, model_name):
        """
            Custom Manual Photos Taker window Class constructor.

            :param names: list containing the different face labels
            :type: list(str)

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str

            :param model_name: name of the model.
            :type model_name: str
        """
        self.current_working_directory = current_working_directory
        self.initial_images_path = os.path.join(
            self.current_working_directory, 'initial_images')
        self.names = names
        self.model_name = model_name

    def setupUi(self, manual_photos_taker):
        """
            Sets up and displays the Custom Manual Photos Taker window.

            :param manual_photos_taker:
            :type manual_photos_taker: QtWidgets.QMainWindow() instance
        """
        self.manual_photos_taker = manual_photos_taker
        super().setupUi(self.manual_photos_taker)

        self.disply_width = 648
        self.display_height = 480

        self.process_data_btn.clicked.connect(self.launch_data_processing)

        for name in self.names:
            self.comboBox.addItem(name)

        self.take_phots_btn.clicked.connect(self.takePhotoManually)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        """
            Stops the thread linked to the Interface.

            NOTE:
                This function has been taken from: https://github.com/docPhil99/opencvQtdemo and cutomized.
        """
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        """
            Updates the image_container_label with a new opencv image

            :param cv_img: new image to display.
            :type cv_img: OpenCv2 instance

            NOTE:
                This function has been taken from: https://github.com/docPhil99/opencvQtdemo and cutomized.
        """
        qt_img = self.convert_cv_qt(cv_img)
        self.image_container_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """
            Converts an opencv image to QtGui.QPixmap

            :param cv_img: new image to display.
            :type cv_img: OpenCv2 instance

            NOTE:
                This function has been taken from: https://github.com/docPhil99/opencvQtdemo and cutomized.
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def takePhotoManually(self):
        """
            Takes (saves) a photo when the user clicks on the correct button. 
        """
        directory = os.path.join(
            self.initial_images_path, self.comboBox.currentText())

        # create directory if not existing
        if os.path.isdir(directory) is False:
            os.makedirs(directory, exist_ok=True)

        new_img_index = '{:04}'.format(len(os.listdir(directory)))
        photo_path = os.path.join(directory, self.comboBox.currentText().lower() + '_' + str(new_img_index) +
                                  '.jpg')

        cv2.imwrite(photo_path, self.thread.cv_img)

    # def launch_data_processing(self):
    #     process_data(self.current_working_directory)

    def popupWindow(self, window_title, text, icon, informative_text, buttons_number):
        """
            Pops up a window when the user wants to start the images processing phase.

            :param window_title: Title of the popup window.
            :type window_title: str

            :param text: text to display in the popup window.
            :type text: str

            :param icon: type of icon to display.
            :type icon: message.setIcon() instance

            :param informative_text: additional text to display.
            :type informative_text: str

            :param buttons_number: number of buttons. 
                NOTE: That is an argument used by the programmer that allows him to re-use this function several times with 1 or 2 buttons.
            :type buttons_number: int
        """
        message = QtWidgets.QMessageBox()
        message.setWindowTitle(window_title)
        message.setText(text)
        message.setStyleSheet(
            """
            QWidget {
                background-color: rgb(46, 46, 46); 
                color: #ffffff;
            }
            QPushButton {
                    color: #ffffff;
                    background-color: none;
                    border: 2px solid rgb(0, 170, 255);
                    border-radius: 10px;
                    width: 50px;
                    height: 20px;
            }
            QPushButton:hover {
                border: 2px solid rgb(255, 0, 0);
            }
            QPushButton:pressed {
                background-color: rgb(100, 100, 100);
            }
            """)

        message.setIcon(icon)

        if buttons_number == 2:
            message.setStandardButtons(
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Abort)
        elif buttons_number == 1:
            message.setStandardButtons(QtWidgets.QMessageBox.Ok)
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Excepted "buttons_number" values: 1 or 2, given: {}'.format(
                buttons_number))
            if error_dialog.exec() == 'QtWidgets.QMessageBox.Ok':
                sys.exit(1)

        message.setInformativeText(informative_text)

        buttonReply = message.exec()

        return buttonReply

    def launch_data_processing(self):
        """
            Launches the data processing phase.
        """
        buttonReply = self.popupWindow(
            "Data Processing",
            """You are about to process the data you have recorded.\n
            This step is mandatory before feeding and training the FacesRecognizer tool.\n
            However, the processing phase can take a while.\n""",
            QtWidgets.QMessageBox.Warning,
            "If you want to proceed, please push 'Ok'\nIf you want to add more data, push 'Abort'.",
            2)

        if buttonReply == QtWidgets.QMessageBox.Ok:
            self.thread.stop()  # stop webcam
            self.manual_photos_taker.hide()  # hide window
            # print('OK for data processing')
            from face_recognizer.src.back.process_data import process_data  # adjime !
            process_data(self.current_working_directory)
            model_saving_path = self.launch_model_training(
                self.current_working_directory, self.model_name)
            self.launch_FaceRecognizer(model_saving_path, self.names)
        elif buttonReply == QtWidgets.QMessageBox.Abort:
            pass
            # print('ABORT data processing')

    def launch_model_training(self, current_working_directory, model_name):
        """
            Launches the training phase.

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str

            :param model_name: name of the model.
            :type model_name: str            
        """
        buttonReply = self.popupWindow(
            "Model training",
            """CONGRATULATIONS, the data have been processed successfully !\n
            The application 'FaceRecognizer' will know be trained on these data!\n""",
            QtWidgets.QMessageBox.Information,
            "Please push 'Ok' to start the training phase\n",
            1)

        if buttonReply == QtWidgets.QMessageBox.Ok:
            from face_recognizer.src.back.build_model import train_model
            return train_model(current_working_directory, model_name)

    def launch_FaceRecognizer(self, model_saving_path, classes):
        """
            Launches the application.

            :param model_saving_path: path where to save the model trained.
            :type model_saving_path: str

            :param classes: classes (i.e. names) of people to recgnize.
            :type classes: list(str)
        """
        buttonReply = self.popupWindow(
            "FaceRecognizer launcher",
            "CONGRATULATIONS, the learning phase has been completed successfully !\n",
            QtWidgets.QMessageBox.Information,
            "Please push 'Ok' to start the training phase\n",
            1)

        if buttonReply == QtWidgets.QMessageBox.Ok:
            print('OK for starting the application')
            from face_recognizer.src.back.faceRecognizer import main
            main(model_saving_path, classes)
