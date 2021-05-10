"""
This module implementes an interface window that processes the data and trains the classifier.
"""

import os
import sys
import queue
import time
import traceback

from PyQt5 import QtCore, QtGui, QtWidgets
from face_recognizer.raw_UIs.application_launcher import Ui_ApplicationLauncher_window
from face_recognizer.src.back.utils import print_header_with_specific_layout
from face_recognizer.src.back.pyqt_multithreadings_management import CustomStreamWriter, StdoutListener, LongRunningFunction


class CustomApplicationLauncherWindow(Ui_ApplicationLauncher_window):

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

    def setupUi(self, ApplicationLauncher_window):
        """
                Sets up and displays the Custom Manual Photos Taker window.

                :param ApplicationLauncher_window:
                :type ApplicationLauncher_window: QtWidgets.QMainWindow() instance
        """
        self.ApplicationLauncher_window = ApplicationLauncher_window
        super().setupUi(self.ApplicationLauncher_window)

        # Create a QtCore.QThreadPoolobject. It creates and manages different threads of class QtCore.QRunnable
        self.threadpool = QtCore.QThreadPool()
        # Show how many threads can be handled depending on computer
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())

        # launch processing
        self.pushButton.clicked.connect(self.trigger_popup)

    def trigger_popup(self):
        buttonReply = self.popupWindow(
            window_title="Start Application Setup",
            text="""You are about to launch the Application.\n
                Before it starts though, the data you provided will be processed and a custom Deep Learning Model will be trained.\n
                These steps may take a while.\n""",
            icon=QtWidgets.QMessageBox.Warning,
            informative_text="If you want to proceed, please push 'Ok'\nOtherwise, push 'Abort'.",
            buttons_number=2
        )

        if buttonReply == QtWidgets.QMessageBox.Ok:
            self.threads_launcher()
        elif buttonReply == QtWidgets.QMessageBox.Abort:
            pass

    def threads_launcher(self):
        # Create a queue
        q = queue.Queue()
        # Redirect sys.stdout to the queue through CustomStreamWriter
        sys.stdout = CustomStreamWriter(q, 'white')
        sys.stderr = CustomStreamWriter(q, 'red')

        # Instanciating the long-running function (the function to be run in a separate thread is self.launch_application)
        self.long_running_function = LongRunningFunction(
            self.launch_application)

        # Create a new thread that will deal execute this function
        self.threadpool.start(self.long_running_function)

        # Instanciating the Listener for output
        self.stdout_listener = StdoutListener(q)
        # Connect the emmited signal to a function that will update the edit_line (this function is self.update_edit_line)
        self.stdout_listener.signals.message.connect(self.update_edit_line)
        # Create a new thread that will handle the listening phase
        self.threadpool.start(self.stdout_listener)

    def update_edit_line(self, text):
        self.edit_line.moveCursor(QtGui.QTextCursor.End)
        self.edit_line.insertHtml(text)

    def launch_application(self):
        """
            Launches the application.
        """
        try:
            print_header_with_specific_layout("DATA IS BEING PROCESSED ...")
            time.sleep(1)
            self.launch_data_processing(self.current_working_directory)

            print_header_with_specific_layout("MODEL IS BEING TRAINED ...")
            time.sleep(1)
            model_saving_path = self.launch_model_training(
                self.current_working_directory, self.model_name)

            print_header_with_specific_layout("STARTING APPLICATION ...")
            time.sleep(1)
            # self.launch_FaceRecognizer(model_saving_path, self.names)

            self.close_application()

        except:
            traceback.print_exc()
            self.close_application('fail')

    def close_application(self, run_status='success'):
        """
            Closes the application efficiently (kills all running threads and closes the window).

            :param run_status: flag indicating whether the application runs successfully or not.
            :type run_status: str
        """
        if run_status != 'success':
            print("THE APPLICATION CRASHED !")
        else:
            print("THE APPLICATION RAN SUCCESSFULLY !")

        time.sleep(1)
        print_header_with_specific_layout("KILLING THREADS ...")
        time.sleep(1)
        print("Done.")

        print_header_with_specific_layout("CLOSING APPLICATION ...")
        time.sleep(1)
        print("Done.")

        time.sleep(5)
        self.stdout_listener.stop()
        self.ApplicationLauncher_window.close()
        sys.exit(0)

    def launch_data_processing(self, current_working_directory):
        """
            Launches the data processing phase.

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str
        """
        from face_recognizer.src.back.process_data import process_data
        process_data(self.current_working_directory)

    def launch_model_training(self, current_working_directory, model_name):
        """
            Launches the training phase.

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str

            :param model_name: name of the model.
            :type model_name: str
        """

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

        from face_recognizer.src.back.faceRecognizer import main
        main(model_saving_path, classes)

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
