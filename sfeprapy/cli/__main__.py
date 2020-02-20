"""SfePrapy CLI Help.
Usage:
    sfeprapy
    sfeprapy mcs0 gui
    sfeprapy mcs0 run [-p=<int>] <file_name>
    sfeprapy mcs0 figure <file_name>
    sfeprapy mcs0 template <file_name>
    sfeprapy distfit [--data_t=<int>] [--dist_g=<int>] <file_name>

Options:
    --data_t=<int>  an integer indicating data type:
                    0   (default) samples only, a single column data.
                    1   PDF, two columns data containing x, y, without heading.
                    2   CDF, two columns data containing x, y, without heading.
    --dist_g=<int>  an integer indicating what distribution group to be used for fitting the data:
                    0   fit to all available distributions.
                    1   (default) fit to common distribution types.
    -p=<int>        to define number of processes for MCS, positive integer only.
    -h --help       to show this message.

Commands:
                    Will run `mcs0 gui` if no command is supplied.
    mcs0 gui        Experimental GUI version of sfeprapy.
    mcs0 run        Monte Carlo Simulation to solve equivalent time exposure in ISO 834 fire, method 0.
    mcs0 figure     produce figure from the output file <file_name>.
    mcs0 template   save example input file to <file_name>.
"""

from docopt import docopt

from sfeprapy.func.stats_dist_fit import auto_fit_2
from sfeprapy.gui.__main__ import main as main_gui
from sfeprapy.mcs0 import EXAMPLE_INPUT_CSV
from sfeprapy.mcs0.__main__ import main as mcs0
from sfeprapy.mcs0.__main__ import save_figure as mcs0_figure


def main():
    import os

    arguments = docopt(__doc__)

    if arguments["<file_name>"]:
        arguments["<file_name>"] = os.path.realpath(arguments["<file_name>"])

    if arguments["mcs0"]:

        if arguments["gui"]:
            main_gui()
            return 0

        elif arguments["figure"]:

            mcs0_figure(fp_mcs0_out=arguments["<file_name>"])

        elif arguments["template"]:

            with open(arguments["<file_name>"], "w+") as f:
                f.write(EXAMPLE_INPUT_CSV)

        else:

            fp_mcs_in = arguments["<file_name>"]
            n_threads = arguments["-p"] or 2
            mcs0(fp_mcs_in=fp_mcs_in, n_threads=int(n_threads))

    elif arguments["distfit"]:

        # Default values
        data_type = arguments["--data_t"] or 2
        distribution_list = arguments["--dist_g"] or 1

        # Main
        auto_fit_2(
            data_type=int(data_type),
            distribution_list=int(distribution_list),
            data=arguments["<file_name>"],
        )

    else:
        main_gui()