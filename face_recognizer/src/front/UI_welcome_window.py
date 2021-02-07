# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'welcome_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtWidgets
from face_recognizer.raw_UIs.welcome_window import Ui_MainWindow
from face_recognizer.src.front.UI_configuration_window import CustomConfigurationWindow


class CustomWelcomeWindow(Ui_MainWindow):

    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        super().setupUi(self.MainWindow)

        self.application_start_btn.clicked.connect(
            self.launch_configuration_window_UI)

    def launch_configuration_window_UI(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = CustomConfigurationWindow()
        self.ui.setupUi(self.window)
        self.MainWindow.hide()
        self.window.show()

    # def launch_faceRecognizer(self):
    #     buttonReply = self.popupWindow(
    #         "Caution",
    #         """You are about to start FaceRecognizer.\n
    #         Have you ever configured the application with your own data ?\n""",
    #         QtWidgets.QMessageBox.Question)

    #     if buttonReply == QtWidgets.QMessageBox.Yes:
    #         print('YES')
    #         # from faceRecognizer import main
    #         # self.MainWindow.hide()
    #         # main()
    #     elif buttonReply == QtWidgets.QMessageBox.No:
    #         print('NO')
    #     else:
    #         pass

    # def popupWindow(self, window_title, text, icon):
    #     message = QtWidgets.QMessageBox()
    #     message.setWindowTitle(window_title)
    #     message.setText(text)
    #     message.setStyleSheet(
    #         """
    #         QWidget {
    #             background-color: rgb(46, 46, 46);
    #             color: #ffffff;
    #         }
    #         QPushButton {
    #                 color: #ffffff;
    #                 background-color: none;
    #                 border: 2px solid rgb(0, 170, 255);
    #                 border-radius: 10px;
    #                 width: 50px;
    #                 height: 20px;
    #         }
    #         QPushButton:hover {
    #             border: 2px solid rgb(255, 0, 0);
    #         }
    #         QPushButton:pressed {
    #             background-color: rgb(100, 100, 100);
    #         }
    #         """)

    #     message.setIcon(icon)

    #     message.setStandardButtons(
    #         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    #     buttonReply = message.exec()

    #     return buttonReply


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = CustomWelcomeWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
