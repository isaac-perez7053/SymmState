import argparse


def main():
    """
    Allows the flpz program to be run from the terminal as python -m flpz
    """
    parser = argparse.ArgumentParser(
        description="FLPZ: A tool to study flexo and piezoelectricity"
    )
    parser.add_argument(
        "program",
        choices=["energy", "perturbations", "coupling"],
        help="Program type to run",
    )
    parser.add_argument("args", nargs="*", help="Arguments for the selected program")

    # args = parser.parse_args()

    # if args.program == 'energy':
    #     flpz.energy(*args.args)
    # elif args.program == 'perturbations':
    #     flpz.perturbations(*args.args)
    # elif args.program == 'coupling':
    #     flpz.coupling(*args.args)


if __name__ == "__main__":
    main()
