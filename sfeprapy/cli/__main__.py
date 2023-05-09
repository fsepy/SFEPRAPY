def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='SFEPRAPY - Structural Fire Engineering Probabilistic Reliability Analysis'
    )
    subparsers = parser.add_subparsers(dest='sub_parser')

    p_mcs0 = subparsers.add_parser('mcs0', help='Monte Carlo simulation Type 0')
    p_mcs1 = subparsers.add_parser('mcs1', help='Monte Carlo simulation Type 1')
    p_mcs2 = subparsers.add_parser('mcs2', help='Monte Carlo simulation Type 2')
    for p_mcs in (p_mcs0, p_mcs1, p_mcs2):
        p_mcs.add_argument('-r', '--run',
                           help='run Monte Carlo simulation from input file filepath',
                           action='store_true', )
        p_mcs.add_argument('-e', '--template',
                           help='save example input file to filepath',
                           action='store_true', )
        p_mcs.add_argument('-p', '--processor',
                           help='number of processors to run the simulation, use no more than available logical '
                                'processors, default 1',
                           default=1,
                           type=int,
                           metavar='Integer')
        p_mcs.add_argument('filepath',
                           help=f'input file name (including extension).',
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
            mcs0(fp_mcs_in=os.path.realpath(args.filepath), n_threads=int(args.processor))
        return

    if args.sub_parser == 'mcs1':
        from sfeprapy.mcs1 import cli_main as mcs1
        from sfeprapy.mcs1 import EXAMPLE_INPUT_CSV, EXAMPLE_INPUT_DF

        if args.template:
            if args.filepath.endswith('.xlsx'):
                EXAMPLE_INPUT_DF.to_excel(args.filepath)
            else:
                with open(args.filepath, "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV)

        if args.run:
            mcs1(fp_mcs_in=os.path.realpath(args.filepath), n_threads=int(args.processor))
        return

    if args.sub_parser == 'mcs2':
        from sfeprapy.mcs2 import cli_main as mcs2
        from sfeprapy.mcs2 import EXAMPLE_INPUT_CSV, EXAMPLE_INPUT_DF

        if args.template:
            if args.filepath.endswith('.xlsx'):
                EXAMPLE_INPUT_DF.to_excel(args.filepath)
            else:
                with open(args.filepath, "w+", encoding='utf-8') as f:
                    f.write(EXAMPLE_INPUT_CSV)

        if args.run:
            mcs2(fp_mcs_in=os.path.realpath(args.filepath), n_threads=int(args.processor))
        return

    # DEPRECIATED 9th May 2023
    # if args.sub_parser == 'distfit':
    #     from sfeprapy.func.stats_dist_fit import auto_fit
    #     auto_fit(
    #         data_type=int(args.type),
    #         distribution_list=int(args.group),
    #         data=args.filepath,
    #     )
    #     return


if __name__ == '__main__':
    main()
