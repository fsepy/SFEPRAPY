"""SfePrapy CLI Help.
Usage:
    sfeprapy
    sfeprapy mcs0 run [-p=<int>] <file_name>
    sfeprapy mcs0 template <file_name>
    sfeprapy mcs2 run [-p=<int>] <file_name>
    sfeprapy mcs2 template <file_name>
    sfeprapy mcs3 run [-p=<int>] <file_name>
    sfeprapy mcs3 template <file_name>
    sfeprapy distfit [--data_t=<int>] [--dist_g=<int>] <file_name>

Examples:
    sfeprapy mcs0 template inputs.csv
    sfeprapy mcs0 template inputs.xlsx
    sfeprapy mcs0 run -p 2 inputs.csv
    sfeprapy mcs0 figure mcs.out.csv

Options:
    -p=<int>        to define number of processes for MCS, positive integer only. 2 by default.
    -h --help       to show this message.
    --data_t=<int>  an integer indicating data type:
                    0   (default) samples only, a single column data.
                    1   PDF, two columns data containing x, y, without heading.
                    2   CDF, two columns data containing x, y, without heading.
    --dist_g=<int>  an integer indicating what distribution group to be used for fitting the data:
                    0   fit to all available distributions.
                    1   (default) fit to common distribution types.

Commands:
    mcs0 run        Monte Carlo Simulation to solve equivalent time exposure in ISO 834 fire, method 0.
    mcs0 figure     produce figure from the output file <file_name>.
    mcs0 template   save example input file to <file_name>.
    mcs2 run        Monte Carlo Simulation to solve equivalent time exposure in ISO 834 fire, method 0.
    mcs2 figure     produce figure from the output file <file_name>.
    mcs2 template   save example input file to <file_name>.
    mcs3 run        Monte Carlo Simulation to solve equivalent time exposure in ISO 834 fire, method 0.
    mcs3 figure     produce figure from the output file <file_name>.
    mcs3 template   save example input file to <file_name>.
"""

from docopt import docopt

from sfeprapy.mcs0 import EXAMPLE_INPUT_CSV as EXAMPLE_INPUT_CSV_MCS0
from sfeprapy.mcs0 import EXAMPLE_INPUT_DF as EXAMPLE_INPUT_DF_MCS0
from sfeprapy.mcs0.__main__ import main as mcs0
from sfeprapy.mcs2 import EXAMPLE_INPUT_CSV as EXAMPLE_INPUT_CSV_MCS2
from sfeprapy.mcs2 import EXAMPLE_INPUT_DF as EXAMPLE_INPUT_DF_MCS2
from sfeprapy.mcs2.__main__ import main as mcs2
from sfeprapy.mcs3 import EXAMPLE_INPUT_CSV as EXAMPLE_INPUT_CSV_MCS3
from sfeprapy.mcs3 import EXAMPLE_INPUT_DF as EXAMPLE_INPUT_DF_MCS3
from sfeprapy.mcs3.__main__ import main as mcs3
from sfeprapy.func.stats_dist_fit import auto_fit_2


def main():
    import os

    arguments = docopt(__doc__)

    if arguments["<file_name>"]:
        arguments["<file_name>"] = os.path.realpath(arguments["<file_name>"])

    if arguments["mcs0"]:
        if arguments["template"]:
            if arguments["<file_name>"].endswith('.xlsx'):
                EXAMPLE_INPUT_DF_MCS0.to_excel(arguments["<file_name>"])
            else:
                with open(arguments["<file_name>"], "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV_MCS0)

        else:
            fp_mcs_in = arguments["<file_name>"]
            n_threads = arguments["-p"] or 1
            mcs0(fp_mcs_in=fp_mcs_in, n_threads=int(n_threads))

    elif arguments['mcs2']:
        if arguments['template']:
            if arguments["<file_name>"].endswith('.xlsx'):
                EXAMPLE_INPUT_DF_MCS2.to_excel(arguments["<file_name>"])
            else:
                with open(arguments["<file_name>"], "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV_MCS2)
        else:
            fp_mcs_in = arguments["<file_name>"]
            n_threads = arguments["-p"] or 1
            mcs2(fp_mcs_in=fp_mcs_in, n_threads=int(n_threads))

    elif arguments['mcs3']:
        if arguments['template']:
            if arguments["<file_name>"].endswith('.xlsx'):
                EXAMPLE_INPUT_DF_MCS3.to_excel(arguments["<file_name>"])
            else:
                with open(arguments["<file_name>"], "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV_MCS3)
        else:
            fp_mcs_in = arguments["<file_name>"]
            n_threads = arguments["-p"] or 1
            mcs3(fp_mcs_in=fp_mcs_in, n_threads=int(n_threads))

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
        raise ValueError(f'sfeprapy -h for help')
