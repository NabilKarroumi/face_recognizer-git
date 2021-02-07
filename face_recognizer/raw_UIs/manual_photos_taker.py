# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'manual_photos_taker.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Manual_photos_taker(object):
    def setupUi(self, manual_photos_taker):
        manual_photos_taker.setObjectName("manual_photos_taker")
        manual_photos_taker.resize(650, 700)
        manual_photos_taker.setMinimumSize(QtCore.QSize(650, 700))
        manual_photos_taker.setMaximumSize(QtCore.QSize(650, 700))
        manual_photos_taker.setStyleSheet("QWidget {\n"
                                          "    background-color: rgb(46, 46, 46);\n"
                                          "}")
        self.Manual_photos_taker_label = QtWidgets.QLabel(manual_photos_taker)
        self.Manual_photos_taker_label.setGeometry(
            QtCore.QRect(225, 10, 200, 50))
        self.Manual_photos_taker_label.setMinimumSize(QtCore.QSize(200, 50))
        self.Manual_photos_taker_label.setMaximumSize(QtCore.QSize(200, 50))
        self.Manual_photos_taker_label.setStyleSheet("QLabel {\n"
                                                     "    color: #ffffff;\n"
                                                     "    background-color: none;\n"
                                                     "    border: 2px solid rgb(11, 170, 22);\n"
                                                     "    border-radius: 10px;\n"
                                                     "    font-size: 13px;\n"
                                                     #  "    justify-content: center;\n"
                                                     "}")
        self.Manual_photos_taker_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Manual_photos_taker_label.setObjectName(
            "Manual_photos_taker_label")
        self.process_data_btn = QtWidgets.QPushButton(manual_photos_taker)
        self.process_data_btn.setGeometry(QtCore.QRect(226, 660, 200, 20))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.process_data_btn.sizePolicy().hasHeightForWidth())
        self.process_data_btn.setSizePolicy(sizePolicy)
        self.process_data_btn.setMinimumSize(QtCore.QSize(200, 20))
        self.process_data_btn.setMaximumSize(QtCore.QSize(200, 20))
        self.process_data_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.process_data_btn.setStyleSheet("QPushButton {\n"
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
        self.process_data_btn.setCheckable(False)
        self.process_data_btn.setObjectName("process_data_btn")
        self.widget = QtWidgets.QWidget(manual_photos_taker)
        self.widget.setGeometry(QtCore.QRect(2, 81, 650, 563))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.image_container_label = QtWidgets.QLabel(self.widget)
        self.image_container_label.setMinimumSize(QtCore.QSize(648, 480))
        self.image_container_label.setMaximumSize(QtCore.QSize(648, 480))
        self.image_container_label.setStyleSheet("QLabel {\n"
                                                 "    color: #ffffff;\n"
                                                 "    background-color: none;\n"
                                                 "    border: 5px solid rgb(255, 170, 0);\n"
                                                 "    font-size: 13px;\n"
                                                 "}")
        self.image_container_label.setText("")
        self.image_container_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_container_label.setObjectName("image_container_label")
        self.verticalLayout.addWidget(self.image_container_label)
        self.comboBox = QtWidgets.QComboBox(self.widget)
        self.comboBox.setMinimumSize(QtCore.QSize(100, 20))
        self.comboBox.setMaximumSize(QtCore.QSize(100, 20))
        self.comboBox.setStyleSheet("QComboBox {\n"
                                    "    background-color: rgb(46, 46, 46);\n"
                                    "    color: #ffffff;\n"
                                    "    border: 2px solid rgb(170, 0, 255);\n"
                                    "    border-radius: 10px;\n"
                                    "}\n"
                                    "QComboBox:hover {\n"
                                    "    border: 2px solid rgb(255, 0, 0);\n"
                                    "}\n"
                                    "QComboBox::drop-down {\n"
                                    "    border-top-right-radius: 3px;\n"
                                    "    border-bottom-right-radius: 3px;\n"
                                    "}\n"
                                    "QComboBox QAbstractItemView {\n"
                                    "    color: #ffffff;\n"
                                    "    selection-background-color: rgb(100, 100, 100);\n"
                                    "}\n"
                                    "\n"
                                    "")
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox, 0, QtCore.Qt.AlignHCenter)
        self.take_phots_btn = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.take_phots_btn.sizePolicy().hasHeightForWidth())
        self.take_phots_btn.setSizePolicy(sizePolicy)
        self.take_phots_btn.setMinimumSize(QtCore.QSize(200, 20))
        self.take_phots_btn.setMaximumSize(QtCore.QSize(200, 20))
        self.take_phots_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.take_phots_btn.setStyleSheet("QPushButton {\n"
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
        self.take_phots_btn.setCheckable(False)
        self.take_phots_btn.setObjectName("take_phots_btn")
        self.verticalLayout.addWidget(
            self.take_phots_btn, 0, QtCore.Qt.AlignHCenter)

        self.retranslateUi(manual_photos_taker)
        QtCore.QMetaObject.connectSlotsByName(manual_photos_taker)

    def retranslateUi(self, manual_photos_taker):
        _translate = QtCore.QCoreApplication.translate
        manual_photos_taker.setWindowTitle(_translate(
            "manual_photos_taker", "Manual Photos taker"))
        self.Manual_photos_taker_label.setText(_translate(
            "manual_photos_taker", "Manual Photos Taker"))
        self.process_data_btn.setText(_translate(
            "manual_photos_taker", "Process Data"))
        self.take_phots_btn.setText(_translate(
            "manual_photos_taker", "Take a photo"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    manual_photos_taker = QtWidgets.QWidget()
    ui = Ui_Manual_photos_taker()
    ui.setupUi(manual_photos_taker)
    manual_photos_taker.show()
    sys.exit(app.exec_())
