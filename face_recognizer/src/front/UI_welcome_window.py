"""
    This module implements an interface window that appear when the application is launched.
    Basically, it is a simple menu asking the user if he/she wants to start the application or not.
"""

from PyQt5 import QtWidgets
from face_recognizer.raw_UIs.welcome_window import Ui_MainWindow
from face_recognizer.src.front.UI_configuration_window import CustomConfigurationWindow


class CustomWelcomeWindow(Ui_MainWindow):
    """
        Welcome window Class.
    """

    def setupUi(self, main_window):
        """
            Sets up and displays the welcome window.

            :param main_window: 
            :type main_window: QtWidgets.QMainWindow() instance
        """
        self.main_window = main_window
        super().setupUi(self.main_window)

        self.application_start_btn.clicked.connect(
            self.launch_configuration_window_UI)

        self.application_cancel_btn.clicked.connect(
            self.quit_application)

    def launch_configuration_window_UI(self):
        """
            Launches the configuration window.
        """
        self.window = QtWidgets.QMainWindow()
        self.ui = CustomConfigurationWindow()
        self.ui.setupUi(self.window)
        self.main_window.hide()
        self.window.show()

    def quit_application(self):
        """
            Quits the application.
        """
        self.main_window.close()
