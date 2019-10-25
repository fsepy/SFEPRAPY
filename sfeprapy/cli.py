"""SfePrapy CLI Help.
Usage:
    sfeprapy mcs0 [-p=<int>] [-f] [--template] <file_name>
    sfeprapy distfit [--data_t=<int>] [--dist_g=<int>] <file_name>

Options:
    --data_t=<int>  an integer indicating data type:
                    0   Samples only, a single column data.
                    1   PDF, two columns data containing x, y.
                    2   CDF, two columns data containing x, y.
    --dist_g=<int>  an integer indicating what distribution group to be used for fitting the data:
                    0   fit to all available distributions
                    1   fit to common distribution types
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
    import sfeprapy
    sfeprapy.check_pip_upgrade()

    arguments = docopt(__doc__)

    if arguments['mcs0']:
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
        else:
            arguments['-p'] = arguments['-p'] or 2
            mcs0(
                fp_mcs_in=os.path.realpath(arguments['<file_name>']),
                n_threads=int(arguments['-p'])
            )
    elif arguments['distfit']:
        # Prerequisites
        from sfeprapy.func.stats_dist_fit import auto_fit_2

        # Default values
        data_type = arguments['--data_t'] or 2
        distribution_list = arguments['--dist_g'] or 1

        # Main
        auto_fit_2(
            data_type=int(data_type),
            distribution_list=int(distribution_list),
            data=arguments['<file_name>'])
    else:
        print('instruction unclear.')
