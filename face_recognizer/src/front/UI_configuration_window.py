"""
This module implementes an interface window that allows the user to:
* provide the location of the main directory storing all the data, the model and other information. 
* specify whether he/she wants to add new data to the old one (and hence train a new model) or not (launch the application directly).
"""

import os
from PyQt5 import QtWidgets
from face_recognizer.src.front.UI_prepare_photos_taking import CustomDatasetPreparationWindow
from face_recognizer.raw_UIs.configuration_window import Ui_Configuration_window
from face_recognizer.src.back.utils import files_in_dir, find_specific_file_extension_in_dir, read


def getOpenFilesAndDirs(parent=None, caption='', directory='',
                        filter='', initialFilter='', options=None):
    """
        Custom dialog window allowing the user to open a file or a folder. 

        NOTE:
            This function as been taken from: https://stackoverflow.com/questions/64336575/select-a-file-or-a-folder-in-qfiledialog-pyqt5
    """
    def updateText():
        # update the contents of the line edit widget with the selected files
        selected = []
        for index in view.selectionModel().selectedRows():
            selected.append('"{}"'.format(index.data()))
        lineEdit.setText(' '.join(selected))

    dialog = QtWidgets.QFileDialog(parent, windowTitle=caption)
    dialog.setFileMode(dialog.ExistingFiles)
    if options:
        dialog.setOptions(options)
    dialog.setOption(dialog.DontUseNativeDialog, True)
    if directory:
        dialog.setDirectory(directory)
    if filter:
        dialog.setNameFilter(filter)
        if initialFilter:
            dialog.selectNameFilter(initialFilter)

    # by default, if a directory is opened in file listing mode,
    # QFileDialog.accept() shows the contents of that directory, but we
    # need to be able to "open" directories as we can do with files, so we
    # just override accept() with the default QDialog implementation which
    # will just return exec_()
    dialog.accept = lambda: QtWidgets.QDialog.accept(dialog)

    # there are many item views in a non-native dialog, but the ones displaying
    # the actual contents are created inside a QStackedWidget; they are a
    # QTreeView and a QListView, and the tree is only used when the
    # viewMode is set to QFileDialog.Details, which is not this case
    stackedWidget = dialog.findChild(QtWidgets.QStackedWidget)
    view = stackedWidget.findChild(QtWidgets.QListView)
    view.selectionModel().selectionChanged.connect(updateText)

    lineEdit = dialog.findChild(QtWidgets.QLineEdit)
    # clear the line edit contents whenever the current directory changes
    dialog.directoryEntered.connect(lambda: lineEdit.setText(''))

    dialog.exec_()
    return dialog.selectedFiles()


class CustomConfigurationWindow(Ui_Configuration_window):
    """
        Configuration window Class.
    """

    def setupUi(self, configuration_window):
        """
            Sets up and displays the configuration window.

            :param configuration_window:
            :type configuration_window: QtWidgets.QMainWindow() instance
        """
        self.configuration_window = configuration_window
        super().setupUi(self.configuration_window)

        self.browse_btn.clicked.connect(self.get_working_directory)
        self.datasets_generation_label = QtWidgets.QLabel(
            self.layoutWidget)
        self.confirmation_btn.clicked.connect(self.configuration_confirmed)

    def launch_datasets_preparation_UI(self, current_working_directory):
        """
            Launches the window allowing the user to take photos.

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str
        """
        self.window = QtWidgets.QMainWindow()
        self.ui = CustomDatasetPreparationWindow(current_working_directory)
        self.ui.setupUi(self.window)
        self.window.show()
        self.configuration_window.hide()

    def configuration_confirmed(self):
        """
            Processes the settings set by the user.
        """
        cwd = self.select_working_directory_path_lineEdit.text()

        # First, we chech that a cwd is selected
        if os.path.isdir(cwd):
            if self.comboBox.currentText() == 'No':
                # The user does not want to build a new dataset, we check if the cwd selected contains a default folder
                # containing the classes and the DL model
                default = os.path.join(cwd, 'default')
                if os.path.isdir(default):
                    model = find_specific_file_extension_in_dir(default, '.h5')

                    if model and 'names_list.txt' in files_in_dir(default):
                        classes = read(os.path.join(default, 'names_list.txt'))

                        from face_recognizer.src.back.faceRecognizer import main
                        main(os.path.join(default, model), classes)
                else:
                    buttonReply = self.popupWindow(
                        'Error',
                        """The directory does not contain any trained model.\n
                        You have to generate a dataset and train a model first!""",
                        "Please answer 'Yes' to the question and press 'Confirm'!",
                        QtWidgets.QMessageBox.Critical)

                    if buttonReply == QtWidgets.QMessageBox.Ok:
                        pass

            elif self.comboBox.currentText() == 'Yes':
                self.launch_datasets_preparation_UI(
                    self.select_working_directory_path_lineEdit.text())

    def get_working_directory(self):
        """
            Allow the user to select the CWD he/she wants.
        """
        working_directory = []

        # check if the user selected something
        while not bool(working_directory):
            working_directory = getOpenFilesAndDirs(
                caption='Choose a working directory', directory=r'D:\Users\KARROUMI Nabil\Desktop\ApprendrePython\PROJECTS\FacesRecognition\Restructure')

        # make sure that object is a directory
        if os.path.isdir(working_directory[0]) is False:
            self.get_working_directory()
        else:
            # put the path in the LineEdit widget
            self.select_working_directory_path_lineEdit.setText(
                working_directory[0])

    def popupWindow(self, window_title, text, informative_text, icon):
        """
            Pops up a window if an issue is detected with the user's settings.

            :param window_title: Title of the popup window.
            :type window_title: str

            :param text: text to display in the popup window.
            :type text: str

            :param informative_text: additional text to display.
            :type informative_text: str

            :param icon: type of icon to display.
            :type icon: message.setIcon() instance
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

        message.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message.setInformativeText(informative_text)

        buttonReply = message.exec()

        return buttonReply
