"""
This module contains all the classes used to setup and manage the GUI multithreadings.
"""

import sys
import traceback
from PyQt5 import QtCore


class CustomStreamWriter(object):
    """
        This class defines a custom stream writer that will stack the data fetched by sys.stdout into a queue.
        StdoutListener will then fetch this queue content and emit it as a QtCore.pyqtSignal(str) signal.
    """

    def __init__(self, queue, color='white'):
        """
            :param queue: Queue data structure.
            :type queue: queue.Queue() instance
        """
        self.queue = queue
        self.color = color

    def write(self, text):
        """
            This function is mandatory. It overwrites the write function usually used by sys.stdout.
            Instead of displaying the text, it stacks it into the queue.

            :param text: Text to be stacked into the queue.
            :type text: str
        """
        # self.queue.put(f"<font-color=\"{self.color}\" font-family: Consolas>" + text + "</font-color=><br>")
        self.queue.put(
            f"<center><font color=\"{self.color}\">" + text + "</font><br></center>")

    def flush(self):
        self.queue.queue.clear()


class StdoutListenerSignals(QtCore.QObject):
    """
        This class simply creates QtCore.pyqtSignal(s) to be emitted by a :func:`~face_recognizer.src.back.pyqt_multithreadings_management.StdoutListener` object.
    """
    message = QtCore.pyqtSignal(str)


class StdoutListener(QtCore.QRunnable):
    """
        This class is a "worker". It defines and sets up all the parameters and behaviour we want our future thread to accomplish.
        Will be added to the QThreadPool instance defined in the MainApplication.

        This worker has to continuously (run is overwritten) listen for data stacked in a queue and emit the related signal defined in StdoutListenerSignals.
    """

    def __init__(self, queue):
        """
            :param queue: Queue data structure.
            :type queue: queue.Queue() instance
        """
        super(StdoutListener, self).__init__()
        self.queue = queue
        self.signals = StdoutListenerSignals()
        self.is_killed = False

    @QtCore.pyqtSlot()
    def run(self):
        # While the thread is running
        while not(self.is_killed):
            # Get data from the queue
            text = self.queue.get()
            # Emit it as a signal
            self.signals.message.emit(text)

    def stop(self):
        self.is_killed = True


class LongRunningFunctionSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(tuple)
    finished = QtCore.pyqtSignal()


class LongRunningFunction(QtCore.QRunnable):
    """
        This class is a "worker". It defines and sets up all the parameters and behaviour we want our future thread to accomplish.
        Will be added to the QThreadPool instance defined in the MainApplication.

        This worker has to continuously execute a long-running function.
    """

    def __init__(self, fn, *args, **kwargs):
        """
            :param fn: Function to be run.
            :type fn: callback

            :param *args: fn function arguments.
            :type *args: list()

            :param **kwargs: fn function arguments.
            :type **kwargs: dict()
        """
        super(LongRunningFunction, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = LongRunningFunctionSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            fn_output = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(fn_output)
        finally:
            self.signals.finished.emit()
