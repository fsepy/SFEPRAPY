
import sys

from PySide2 import QtWidgets

from sfeprapy.gui.logic.main import MainWindow


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
    main()
