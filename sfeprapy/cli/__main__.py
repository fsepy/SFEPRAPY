def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='SFEPRAPY - Structural Fire Engineering Probabilistic Reliability Analysis'
    )
    subparsers = parser.add_subparsers(dest='sub_parser')

    p_mcs0 = subparsers.add_parser('mcs0', help='Monte Carlo simulation Type 0')
    p_mcs0.add_argument('-r', '--run',
                        help='run Monte Carlo simulation from input file filepath',
                        action='store_true', )
    p_mcs0.add_argument('-e', '--template',
                        help='save example input file to filepath',
                        action='store_true', )
    p_mcs0.add_argument('-t', '--threads',
                        help='number of threads to run the simulation, default 1',
                        default=1,
                        type=int)
    p_mcs0.add_argument('filepath',
                        help=f'Input file name (including extension).',
                        type=str)

    p_mcs2 = subparsers.add_parser('mcs2', help='Monte Carlo simulation Type 2')
    p_mcs2.add_argument('-r', '--run',
                        help='run Monte Carlo simulation from input file filepath',
                        action='store_true', )
    p_mcs2.add_argument('-e', '--template',
                        help='save example input file to filepath',
                        action='store_true', )
    p_mcs2.add_argument('-t', '--threads',
                        help='number of threads to run the simulation, default 1',
                        default=1,
                        type=int)
    p_mcs2.add_argument('filepath',
                        help=f'Input file name (including extension).',
                        type=str)

    p_distfit = subparsers.add_parser('distfit', help='distribution fit')
    p_distfit.add_argument('-t', '--type',
                           help='an integer indicating data type\n'
                                '0   (default) samples only, a single column data.\n'
                                '1   PDF, two columns data containing x, y, without heading.\n'
                                '2   CDF, two columns data containing x, y, without heading.',
                           default=0,
                           type=int, )
    p_distfit.add_argument('-g', '--group',
                           help='an integer indicating what distribution group to be used for fitting the data:\n'
                                '0   fit to all available distributions.\n'
                                '1   (default) fit to common distribution types.\n',
                           default=1,
                           type=int, )
    p_distfit.add_argument('filepath',
                           help=f'Input file name (including extension).',
                           type=str)

    args = parser.parse_args()

    if args.sub_parser == 'mcs0':
        from sfeprapy.mcs0 import cli_main as mcs0
        from sfeprapy.mcs0 import EXAMPLE_INPUT_CSV
        from sfeprapy.mcs0 import EXAMPLE_INPUT_DF

        if args.template:
            if args.filepath.endswith('.xlsx'):
                EXAMPLE_INPUT_DF.to_excel(args.filepath)
            else:
                with open(args.filepath, "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV)

        if args.run:
            mcs0(fp_mcs_in=os.path.realpath(args.filepath), n_threads=int(args.threads))
        return

    if args.sub_parser == 'mcs2':
        from sfeprapy.mcs2 import cli_main as mcs2
        from sfeprapy.mcs2 import EXAMPLE_INPUT_CSV
        from sfeprapy.mcs2 import EXAMPLE_INPUT_DF

        if args.template:
            if args.filepath.endswith('.xlsx'):
                EXAMPLE_INPUT_DF.to_excel(args.filepath)
            else:
                with open(args.filepath, "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV)

        if args.run:
            mcs2(fp_mcs_in=os.path.realpath(args.filepath), n_threads=int(args.threads))
        return

    if args.sub_parser == 'distfit':
        from sfeprapy.func.stats_dist_fit import auto_fit
        auto_fit(
            data_type=int(args.type),
            distribution_list=int(args.group),
            data=args.filepath,
        )
        return


if __name__ == '__main__':
    main()
