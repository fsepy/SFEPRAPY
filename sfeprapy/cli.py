# -*- coding: utf-8 -*-
"""Greeter.

Usage:
    sfeprapy mcs0 <name> [--caps] [--greeting=<str>]
    sfeprapy (-h | --help)

Options:
    -h --help         Show this screen.
    --caps            Uppercase the output.
    --greeting=<str>  Greeting to use [default: Hello].

Commands:
    mcs0        Time equivalence Monte Carlo Simulation method 0
    mcs1        Time equivalence Monte Carlo Simulation method 1
"""


def main():
    from docopt import docopt
    arguments = docopt(__doc__, options_first=True)
    if arguments['<command>']:
        print('test successful, msc0 is selected.')
    elif arguments['msc1']:
        print('test successful, msc1 is selected.')


if __name__ == '__main__':
    main()
