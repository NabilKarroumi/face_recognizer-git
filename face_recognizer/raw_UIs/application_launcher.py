# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\face_recognizer-git\face_recognizer\raw_UIs\application_launcher.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ApplicationLauncher_window(object):
    def setupUi(self, ApplicationLauncher_window):
        ApplicationLauncher_window.setObjectName("ApplicationLauncher_window")
        ApplicationLauncher_window.resize(800, 575)
        ApplicationLauncher_window.setMinimumSize(QtCore.QSize(800, 575))
        ApplicationLauncher_window.setMaximumSize(QtCore.QSize(800, 575))
        self.centralwidget = QtWidgets.QWidget(ApplicationLauncher_window)
        self.centralwidget.setMinimumSize(QtCore.QSize(800, 575))
        self.centralwidget.setMaximumSize(QtCore.QSize(800, 575))
        self.centralwidget.setStyleSheet("QWidget {\n"
                                         "    background-color: rgb(46, 46, 46);\n"
                                         "}")
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(300, 540, 200, 20))
        self.pushButton.setMinimumSize(QtCore.QSize(200, 20))
        self.pushButton.setMaximumSize(QtCore.QSize(200, 20))
        self.pushButton.setStyleSheet("QPushButton {\n"
                                      "    color: #ffffff;\n"
                                      "    background-color: none;\n"
                                      "    border: 2px solid rgb(0, 170, 255);\n"
                                      "    border-radius: 10px;\n"
                                      "}\n"
                                      "QPushButton:hover {\n"
                                      "    border: 2px solid rgb(255, 0, 0);\n"
                                      "}\n"
                                      "QPushButton:pressed {\n"
                                      "    background-color: rgb(100, 100, 100);\n"
                                      "}")
        self.pushButton.setObjectName("pushButton")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(11, 12, 781, 521))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMinimumSize(QtCore.QSize(200, 50))
        self.label.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(1)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel {\n"
                                 "    color: #ffffff;\n"
                                 "    background-color: none;\n"
                                 "    border: 2px solid rgb(11, 170, 22);\n"
                                 "    border-radius: 10px;\n"
                                 "    font-size: 13px;\n"
                                 "}")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
        self.edit_line = QtWidgets.QTextEdit(self.widget)
        self.edit_line.setEnabled(True)
        self.edit_line.setMaximumSize(QtCore.QSize(780, 400))
        self.edit_line.setStyleSheet("QTextEdit {\n"
                                     "    border: 2px solid rgb(255, 0, 255);\n"
                                     "    color: white;\n"
                                     "    border-radius: 20px;\n"
                                     "    font-family: FreeMono;\n"
                                     "    font-size: 13px;"
                                     "}")
        self.edit_line.setReadOnly(True)
        self.edit_line.setObjectName("edit_line")
        self.verticalLayout.addWidget(self.edit_line)
        # self.progressBar = QtWidgets.QProgressBar(self.widget)
        # self.progressBar.setMaximumSize(QtCore.QSize(780, 30))
        # self.progressBar.setStyleSheet("QProgressBar\n"
        #    "{\n"
        #    "    border: 2px solid rgb(65, 0, 0);\n"
        #    "    border-radius: 10px;\n"
        #    "    font: 75 10pt \"Arial\";\n"
        #    "    color: white;\n"
        #    "    text-align: center;\n"
        #    "}\n"
        #    "QProgressBar::chunk \n"
        #    "{\n"
        #    "    background-color: rgb(85, 0, 0);\n"
        #    "    border-radius :8px;\n"
        #    "}  ")
        # self.progressBar.setProperty("value", 50)
        # # self.progressBar.setObjectName("progressBar")
        # self.verticalLayout.addWidget(self.progressBar)
        ApplicationLauncher_window.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ApplicationLauncher_window)
        self.statusbar.setObjectName("statusbar")
        ApplicationLauncher_window.setStatusBar(self.statusbar)

        self.retranslateUi(ApplicationLauncher_window)
        QtCore.QMetaObject.connectSlotsByName(ApplicationLauncher_window)

    def retranslateUi(self, ApplicationLauncher_window):
        _translate = QtCore.QCoreApplication.translate
        ApplicationLauncher_window.setWindowTitle(_translate(
            "ApplicationLauncher_window", "FaceRecognizer-ApplicationLauncher"))
        self.pushButton.setText(_translate(
            "ApplicationLauncher_window", "Lauch Processing"))
        self.label.setText(_translate(
            "ApplicationLauncher_window", "Application Set Up"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ApplicationLauncher_window = QtWidgets.QMainWindow()
    ui = Ui_ApplicationLauncher_window()
    ui.setupUi(ApplicationLauncher_window)
    ApplicationLauncher_window.show()
    sys.exit(app.exec_())
