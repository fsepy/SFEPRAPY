import sys

from PySide2 import QtWidgets

from sfeprapy.guilayout.main import Ui_MainWindow
from sfeprapy.mcs0 import EXAMPLE_INPUT_CSV
from sfeprapy.mcs0.__main__ import main as mcs0


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()  # inherit base class

        self.ui = Ui_MainWindow()  # instantiate ui
        self.ui.setupUi(self)  # set up ui

        # set ui elements initial status
        self.ui.lineEdit_input_file.setReadOnly(True)

        # set ui elements connections
        self.ui.pushButton_input_file.clicked.connect(self.select_input_file)
        self.ui.pushButton_test.clicked.connect(self.make_input_template)
        self.ui.pushButton_run.clicked.connect(self.run)

    def select_input_file(self):
        """select input file and copy its path to ui object"""

        # dialog to select file
        path_to_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            "~/",
            "Spreadsheet (*.csv *.xls *.xlsx)")

        # paste the select file path to the ui object
        self.ui.lineEdit_input_file.setText(path_to_file)

    def make_input_template(self):
        """save an input template file to a user specified location"""

        # get the user desired file path
        path_to_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            caption='Save Template Input File',
            dir='mcs0.csv'
        )

        # save file
        with open(path_to_file, "w+") as f:
            f.write(EXAMPLE_INPUT_CSV)

    def run(self):
        """to fetch input and run simulation"""

        # get input file path
        try:
            fp_mcs_in = self.ui.lineEdit_input_file.text()
            if len(fp_mcs_in) == 0:  # raise error if path length is zero.
                raise
        except:
            self.ui.statusbar.showMessage('Run failed, unable to fetch input file path.')
            self.repaint()
            return -1

        # get number of processes
        try:
            n_threads = int(self.ui.lineEdit_n_proc.text())
        except ValueError:
            self.ui.statusbar.showMessage('Run failed, unable to fetch number of processes.')
            self.repaint()
            return -1

        # run simulation
        mcs0(fp_mcs_in=fp_mcs_in, n_threads=int(n_threads))

    def update_status_bar_text(self, text: str):
        self.ui.statusbar.showMessage(text=text)


def main(app_window=None):
    """main function to fire up the application cycle"""

    app = QtWidgets.QApplication(sys.argv)
    if app_window is None:
        window = MainWindow()
    else:
        window = app_window()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    """update ui files and run the application. this is used only for testing"""
    from sfeprapy.guilayout.ui2py import ui2py
    ui2py()
    main()
