"""SfePrapy CLI Help.
Usage:
    sfeprapy mcs0 [-p=<int>] [-f] [--template] <file_name>

Options:
    -f              to produce figure from the output file <file_name>.
    -p=<int>        to define number of processes for MCS, positive integer only.
    --template      to save example input file to <file_name>.
    -h --help       to show this message.

Commands:
    mcs0            Monte Carlo Simulation to solve equivalent time exposure in ISO 834 fire, method 0.
"""

from docopt import docopt


def main():
    import os

    arguments = docopt(__doc__)

    if 'mcs0' in arguments:
        import sfeprapy
        from sfeprapy.mcs0.__main__ import main as mcs0
        from sfeprapy.mcs0.__main__ import save_figure as mcs0_figure

        arguments['<file_name>'] = os.path.realpath(arguments['<file_name>'])

        if arguments['-f']:
            mcs0_figure(
                fp_mcs0_out=arguments['<file_name>']
            )
        elif arguments['--template']:
            with open(arguments['<file_name>'], 'w+') as f:
                f.write(sfeprapy.mcs0.EXAMPLE_INPUT_CSV)
            # DO NOT NEED CONFIGURATION FILE FOR CLI
            # import json
            # with open(os.path.join(os.path.dirname(arguments['<file_name>']), 'config.json'), 'w+') as f:
            #     json.dump(sfeprapy.mcs0.EXAMPLE_CONFIG_DICT, f)
        else:
            arguments['-p'] = arguments['-p'] or 2
            mcs0(
                fp_mcs_in=os.path.realpath(arguments['<file_name>']),
                n_threads=int(arguments['-p'])
            )
    else:
        print('instruction unclear.')
