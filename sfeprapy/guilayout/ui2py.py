

def ui2py():
    import os
    import subprocess

    list_ui_file_names = [
        'main.ui',
    ]

    cwd = os.path.dirname(os.path.realpath(__file__))
    destination_dir = os.path.dirname(os.path.realpath(__file__))

    for ui_file_name in list_ui_file_names:
        cmd = f'pyside2-uic {os.path.join(cwd, ui_file_name)} > {os.path.join(destination_dir, ui_file_name.replace(".ui", ".py"))}'
        print(cmd)
        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    ui2py()
