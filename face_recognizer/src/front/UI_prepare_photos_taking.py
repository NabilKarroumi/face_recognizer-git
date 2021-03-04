"""
This module implements an interface window that appear immediately after the configuration window (only if the user decides to generate new data).
Basically, it is a second configuration window that asks the user to:
* specify how he/she wants to take photos (either manually or automatically).
* specify the names of all people/classes he/she wants the application to be able to recognize.
* provide the name of the model to be trained.
"""

from PyQt5 import QtWidgets
from face_recognizer.raw_UIs.prepare_photos_taking import Ui_datasets_preparation
from face_recognizer.src.front.UI_automatic_photos_taker import CustomAutomaticPhotosTakerWindow
from face_recognizer.src.front.UI_manual_photos_taker import CustomManualPhotosTakerWindow


def get_all_item_in_QListWidget(list_widget):
    """
        Lists all items listed in a QListWidget.

        :param list_widget: list containing items.
        :type list_widget: QtWidgets.QListWidget() instance
    """
    assert type(
        list_widget) == QtWidgets.QListWidget, 'Make sure the argument is a QListWidget object'

    items = []
    for x in range(list_widget.count()):
        items.append(list_widget.item(x).text())
    return items


class CustomDatasetPreparationWindow(Ui_datasets_preparation):
    """
        Dataset Preparation window Class.
    """

    def __init__(self, current_working_directory):
        """
            Dataset Preparation window Class Constructor.

            :param current_working_directory: path the to the Current Working Directory (CWD).
            :type current_working_directory: str
        """
        self.current_working_directory = current_working_directory

    def setupUi(self, datasets_preparation):
        """
            Sets up and displays the Dataset Preparation window.

            :param datasets_preparation:
            :type datasets_preparation: QtWidgets.QMainWindow() instance
        """
        self.datasets_preparation = datasets_preparation
        super().setupUi(self.datasets_preparation)

        self.confirmation_btn.clicked.connect(self.settings_confirmed)
        self.add_btn.clicked.connect(self.add_name_to_list)
        self.remove_btn.clicked.connect(self.remove_name_from_list)

    def add_name_to_list(self):
        """
            Adds an item to a QListWidget object.
        """
        items = get_all_item_in_QListWidget(self.ListWidget)

        if self.enter_name_editline.text() != '' and self.enter_name_editline.text() not in items:
            self.ListWidget.addItem(self.enter_name_editline.text())
            self.enter_name_editline.clear()

        self.ListWidget.sortItems()

    def remove_name_from_list(self):
        """
            Removes an item from a QListWidget object.
        """
        # only a selected item in the list can be removed
        if self.ListWidget.currentItem():
            self.ListWidget.takeItem(self.ListWidget.currentRow())
            # print(self.ListWidget.count())
            # print(self.ListWidget.currentItem())
            # print(type(self.ListWidget.currentItem()))
        # self.ListWidget.addItem(self.enter_name_editline.text())
        # self.enter_name_editline.clear()

    def settings_confirmed(self):
        """
            Processes the settings set by the user.
        """
        items = get_all_item_in_QListWidget(self.ListWidget)
        if bool(items) and self.model_name_editline.text() != '':  # At leat one name in the list
            # get the combobox content
            if self.comboBox.currentText() == 'Manually':
                self.manual_photos_taker = QtWidgets.QWidget()
                self.ui = CustomManualPhotosTakerWindow(
                    items, self.current_working_directory, self.model_name_editline.text())
                self.ui.setupUi(self.manual_photos_taker)
                self.manual_photos_taker.show()
                self.datasets_preparation.hide()
            else:
                self.automatic_photos_taker = QtWidgets.QWidget()
                self.ui = CustomAutomaticPhotosTakerWindow(
                    items, self.current_working_directory, self.model_name_editline.text())
                self.ui.setupUi(self.automatic_photos_taker)
                self.automatic_photos_taker.show()
                self.datasets_preparation.hide()
