import io
import os
import time
from contextlib import redirect_stdout

from PyQt5 import QtCore, QtGui, QtWidgets
from face_recognizer.raw_UIs.data_processing import Ui_data_processing


# class OutputThread(QtCore.QThread):

#     def __init__(self, out):
#         super().__init__()
#         self.out = out

#     # This signal is sent when stdout captures a message
#     # and triggers the update of the text box in the main thread.
#     output_changed = QtCore.pyqtSignal(object)

#     def run(self):
#         '''listener of changes to global `out`'''
#         while True:
#             self.out.flush()
#             text = self.out.getvalue()
#             if text:
#                 self.output_changed.emit(text)
#                 # clear the buffer
#                 self.out.truncate(0)
#                 self.out.seek(0)
#             time.sleep(0.1)


# def update_stdout_to_show_in_gui(func):
#     def wrapper(out, *args, **kwargs):
#         out.write('Test \n')

#         # This context manager will redirect the output from
#         # `sys.stdout` to `out`
#         with redirect_stdout(out):
#             result = func(out, *args, **kwargs)

#         out.write('='*80 + '\n')
#         return result
#     return wrapper

class OutputThread(QtCore.QThread):

    out = io.StringIO()

    def __init__(self):
        super().__init__()
        self._run_flag = True

    # This signal is sent when stdout captures a message
    # and triggers the update of the text box in the main thread.
    output_changed = QtCore.pyqtSignal(object)

    def run(self):
        '''listener of changes to global `out`'''
        while self._run_flag:
            OutputThread.out.flush()
            text = OutputThread.out.getvalue()
            if text:
                self.output_changed.emit(text)
                # clear the buffer
                OutputThread.out.truncate(0)
                OutputThread.out.seek(0)
            time.sleep(0.5)

    def update_stdout_to_show_in_gui(func):
        # out = io.StringIO()

        def wrapper(self):
            OutputThread.out.write('Test \n')

            # This context manager will redirect the output from
            # `sys.stdout` to `out`
            with redirect_stdout(OutputThread.out):
                # result = func(self)
                func(self)

            OutputThread.out.write('='*80 + '\n')
            # return result
        return wrapper

    def stop_running(self):
        self._run_flag = False


class CustomDataProcessing(Ui_data_processing):

    def __init__(self, names, current_working_directory, model_name):
        """
            Custom Data Processing window Class constructor.

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

    def setupUi(self, data_processing):
        """
            Sets up and displays the Custom Manual Photos Taker window.

            :param data_processing:
            :type data_processing: QtWidgets.QMainWindow() instance
        """
        self.data_processing = data_processing
        super().setupUi(self.data_processing)

        # Thread to update text of output tab
        self.output_thread = OutputThread()
        self.output_thread.output_changed.connect(self.on_output_changed)
        # Start the listener
        self.output_thread.start()

        # launch processing
        self.pushButton.clicked.connect(self.launch_data_processing)

    def on_output_changed(self, text):
        self.output_text.append(text.strip())

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

    @OutputThread.update_stdout_to_show_in_gui
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
            from face_recognizer.src.back.process_data import process_data  # adjime !

            process_data(self.current_working_directory)

            model_saving_path = self.launch_model_training(
                self.current_working_directory, self.model_name)

            self.output_thread.stop_running()
            self.data_processing.hide()

            self.launch_FaceRecognizer(model_saving_path, self.names)
        elif buttonReply == QtWidgets.QMessageBox.Abort:
            pass

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
            from face_recognizer.src.back.faceRecognizer import main
            main(model_saving_path, classes)
            exit()
