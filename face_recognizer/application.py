# coding: utf-8

import sys
from PyQt5 import QtWidgets
from face_recognizer.src.front.UI_welcome_window import CustomWelcomeWindow


def launch_application():
    """
        Launches the full application. 
        This file is used in the face_recognizer_launcher.bat file. 
    """
    try:
        application = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        ui = CustomWelcomeWindow()
        ui.setupUi(window)
        window.show()
        sys.exit(application.exec_())

    except Exception as e:
        print('A problem has occured: {}'.format(e))
