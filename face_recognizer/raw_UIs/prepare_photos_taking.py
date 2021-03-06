# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'prepare_photos_taking.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_datasets_preparation(object):
    def setupUi(self, datasets_preparation):
        datasets_preparation.setObjectName("datasets_preparation")
        datasets_preparation.resize(500, 405)
        datasets_preparation.setMinimumSize(QtCore.QSize(500, 405))
        datasets_preparation.setMaximumSize(QtCore.QSize(500, 405))
        self.centralwidget = QtWidgets.QWidget(datasets_preparation)
        self.centralwidget.setMinimumSize(QtCore.QSize(500, 405))
        self.centralwidget.setMaximumSize(QtCore.QSize(500, 405))
        self.centralwidget.setStyleSheet("QWidget {\n"
                                         "    background-color: rgb(46, 46, 46);\n"
                                         "}")
        self.centralwidget.setObjectName("centralwidget")
        self.window_title_label = QtWidgets.QLabel(self.centralwidget)
        self.window_title_label.setGeometry(QtCore.QRect(150, 10, 200, 50))
        self.window_title_label.setMinimumSize(QtCore.QSize(200, 50))
        self.window_title_label.setMaximumSize(QtCore.QSize(200, 50))
        self.window_title_label.setStyleSheet("QLabel {\n"
                                              "    color: #ffffff;\n"
                                              "    background-color: none;\n"
                                              "    border: 2px solid rgb(11, 170, 22);\n"
                                              "    border-radius: 10px;\n"
                                              "    font-size: 13px;\n"
                                              "}")
        self.window_title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.window_title_label.setObjectName("window_title_label")
        self.confirmation_btn = QtWidgets.QPushButton(self.centralwidget)
        self.confirmation_btn.setGeometry(QtCore.QRect(150, 380, 200, 20))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.confirmation_btn.sizePolicy().hasHeightForWidth())
        self.confirmation_btn.setSizePolicy(sizePolicy)
        self.confirmation_btn.setMinimumSize(QtCore.QSize(200, 20))
        self.confirmation_btn.setMaximumSize(QtCore.QSize(200, 20))
        self.confirmation_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.confirmation_btn.setStyleSheet("QPushButton {\n"
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
        self.confirmation_btn.setCheckable(False)
        self.confirmation_btn.setObjectName("confirmation_btn")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 70, 483, 301))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ask_user_label = QtWidgets.QLabel(self.layoutWidget)
        self.ask_user_label.setMinimumSize(QtCore.QSize(0, 20))
        self.ask_user_label.setMaximumSize(QtCore.QSize(16777215, 20))
        self.ask_user_label.setStyleSheet("QLabel {\n"
                                          "    color: #ffffff;\n"
                                          "    background-color: none;\n"
                                          "    border: 2px solid rgb(255, 170, 0);\n"
                                          "    border-radius: 10px;\n"
                                          "    font-size: 13px;\n"
                                          "}")
        self.ask_user_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ask_user_label.setObjectName("ask_user_label")
        self.horizontalLayout.addWidget(self.ask_user_label)
        self.comboBox = QtWidgets.QComboBox(self.layoutWidget)
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
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.ask_user_label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.ask_user_label_2.setMinimumSize(QtCore.QSize(0, 50))
        self.ask_user_label_2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.ask_user_label_2.setStyleSheet("QLabel {\n"
                                            "    color: #ffffff;\n"
                                            "    background-color: none;\n"
                                            "    border: 2px solid rgb(255, 170, 0);\n"
                                            "    border-radius: 10px;\n"
                                            "    font-size: 13px;\n"
                                            "}")
        self.ask_user_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.ask_user_label_2.setObjectName("ask_user_label_2")
        self.verticalLayout.addWidget(self.ask_user_label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setContentsMargins(1, 0, 1, -1)
        self.horizontalLayout_2.setSpacing(10)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.enter_name_editline = QtWidgets.QLineEdit(self.layoutWidget)
        self.enter_name_editline.setStyleSheet("QLineEdit {\n"
                                               "    background-color: none;\n"
                                               "    color: #000000;\n"
                                               "    border: 2px solid rgb(170, 85, 255);\n"
                                               "}")
        self.enter_name_editline.setInputMask("")
        self.enter_name_editline.setText("")
        self.enter_name_editline.setMaxLength(32767)
        self.enter_name_editline.setObjectName("enter_name_editline")
        self.horizontalLayout_2.addWidget(self.enter_name_editline)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.add_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.add_btn.setMinimumSize(QtCore.QSize(50, 20))
        self.add_btn.setMaximumSize(QtCore.QSize(50, 20))
        self.add_btn.setStyleSheet("QPushButton {\n"
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
        self.add_btn.setObjectName("add_btn")
        self.verticalLayout_2.addWidget(
            self.add_btn, 0, QtCore.Qt.AlignHCenter)
        self.remove_btn = QtWidgets.QPushButton(self.layoutWidget)
        self.remove_btn.setMinimumSize(QtCore.QSize(70, 20))
        self.remove_btn.setMaximumSize(QtCore.QSize(70, 20))
        self.remove_btn.setStyleSheet("QPushButton {\n"
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
        self.remove_btn.setObjectName("remove_btn")
        self.verticalLayout_2.addWidget(self.remove_btn)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.ListWidget = QtWidgets.QListWidget(self.layoutWidget)
        self.ListWidget.setMinimumSize(QtCore.QSize(161, 100))
        self.ListWidget.setMaximumSize(QtCore.QSize(161, 100))
        self.ListWidget.setStyleSheet("QListWidget {\n"
                                      "    background-color: rgb(46, 46, 46);\n"
                                      "    color: #ffffff;\n"
                                      "    border: 2px solid rgb(170, 85, 0);\n"
                                      "    border-radius: 10px;\n"
                                      "}\n"
                                      "QListView::item:hover {\n"
                                      "    background-color: rgb(100 , 100, 100);\n"
                                      "    color: #ffffff;\n"
                                      "    border: none;\n"
                                      "    border-radius: 8px;\n"
                                      "}\n"
                                      "QListView::item:selected {\n"
                                      "    background-color: rgb(195, 195, 195);\n"
                                      "    color: #ffffff;\n"
                                      "    border: none;\n"
                                      "    border-radius: 8px;\n"
                                      "}\n"
                                      "QListView::item {\n"
                                      "    margin-left: auto;\n"
                                      "    margin-right: auto;\n"
                                      "}")
        self.ListWidget.setMovement(QtWidgets.QListView.Static)
        self.ListWidget.setObjectName("ListWidget")
        self.horizontalLayout_2.addWidget(self.ListWidget)
        self.horizontalLayout_2.setStretch(0, 5)
        self.horizontalLayout_2.setStretch(2, 3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ask_user_model_name_label = QtWidgets.QLabel(self.layoutWidget)
        self.ask_user_model_name_label.setMinimumSize(QtCore.QSize(0, 20))
        self.ask_user_model_name_label.setMaximumSize(
            QtCore.QSize(16777215, 20))
        self.ask_user_model_name_label.setStyleSheet("QLabel {\n"
                                                     "    color: #ffffff;\n"
                                                     "    background-color: none;\n"
                                                     "    border: 2px solid rgb(255, 170, 0);\n"
                                                     "    border-radius: 10px;\n"
                                                     "    font-size: 13px;\n"
                                                     "}")
        self.ask_user_model_name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ask_user_model_name_label.setObjectName(
            "ask_user_model_name_label")
        self.horizontalLayout_3.addWidget(self.ask_user_model_name_label)
        self.model_name_editline = QtWidgets.QLineEdit(self.layoutWidget)
        self.model_name_editline.setStyleSheet("QLineEdit {\n"
                                               "    background-color: none;\n"
                                               "    color: #000000;\n"
                                               "    border: 2px solid rgb(170, 85, 255);\n"
                                               "}")
        self.model_name_editline.setInputMask("")
        self.model_name_editline.setText("")
        self.model_name_editline.setMaxLength(32767)
        self.model_name_editline.setObjectName("model_name_editline")
        self.horizontalLayout_3.addWidget(self.model_name_editline)
        self.horizontalLayout_3.setStretch(0, 4)
        self.horizontalLayout_3.setStretch(1, 3)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        datasets_preparation.setCentralWidget(self.centralwidget)

        self.retranslateUi(datasets_preparation)
        QtCore.QMetaObject.connectSlotsByName(datasets_preparation)

    def retranslateUi(self, datasets_preparation):
        _translate = QtCore.QCoreApplication.translate
        datasets_preparation.setWindowTitle(_translate(
            "datasets_preparation", "FaceRecognizer-DatasetsPreparation"))
        self.window_title_label.setText(_translate(
            "datasets_preparation", "Preparation of the Datasets ..."))
        self.confirmation_btn.setText(
            _translate("datasets_preparation", "Confirm"))
        self.ask_user_label.setText(_translate(
            "datasets_preparation", "How would you like to take photos ?"))
        self.comboBox.setItemText(0, _translate(
            "datasets_preparation", "Manually"))
        self.comboBox.setItemText(1, _translate(
            "datasets_preparation", "Automatically"))
        self.ask_user_label_2.setText(_translate(
            "datasets_preparation", "Please enter the names of all the people you want the tool to be able to recognize !"))
        self.enter_name_editline.setPlaceholderText(
            _translate("datasets_preparation", "Enter a name"))
        self.add_btn.setText(_translate("datasets_preparation", "Add"))
        self.remove_btn.setText(_translate("datasets_preparation", "Remove"))
        self.ask_user_model_name_label.setText(_translate(
            "datasets_preparation", "How would you like to name your model ?"))
        self.model_name_editline.setPlaceholderText(
            _translate("datasets_preparation", "Enter a model name"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    datasets_preparation = QtWidgets.QMainWindow()
    ui = Ui_datasets_preparation()
    ui.setupUi(datasets_preparation)
    datasets_preparation.show()
    sys.exit(app.exec_())
