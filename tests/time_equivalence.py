# coding: utf-8

if __name__ == '__main__':
    import os
    from sfeprapy import time_equivalence as app

    work_directory = os.path.dirname(os.path.realpath(__file__))

    app.run([work_directory])
