# coding: utf-8

import sys
from PyQt5 import QtWidgets
from face_recognizer.src.front.UI_welcome_window import CustomWelcomeWindow


def launch_application():
    try:
        application = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        ui = CustomWelcomeWindow()
        ui.setupUi(window)
        window.show()
        sys.exit(application.exec_())

    except Exception as e:
        print('A problem has occured: {}'.format(e))


if __name__ == "__main__":
    launch_application()
