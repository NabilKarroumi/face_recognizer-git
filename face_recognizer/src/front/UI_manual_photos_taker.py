"""
This module implementes an interface window that allows the user to create/generate photos (data) manually.
"""

import os
import sys
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from face_recognizer.raw_UIs.manual_photos_taker import Ui_Manual_photos_taker
from face_recognizer.src.front.UI_application_launcher import CustomApplicationLauncherWindow


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

        self.process_data_btn.clicked.connect(self.launch_data_processing_UI)

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

    def launch_data_processing_UI(self):
        """
            Launches the window allowing the user to process data.

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str
        """
        self.window = QtWidgets.QMainWindow()
        # self.ui = CustomApplicationLauncherWindow(current_working_directory)
        self.ui = CustomApplicationLauncherWindow(
            self.names, self.current_working_directory, self.model_name)
        self.ui.setupUi(self.window)
        self.window.show()
        self.thread.stop()  # stop webcam
        self.manual_photos_taker.hide()  # hide window


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = CustomManualPhotosTakerWindow(
        names=['titi', 'toto'],
        current_working_directory=r'D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\Restructure\new',
        model_name='mod'
    )
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
