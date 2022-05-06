import argparse
from argparse import Namespace


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns:
      parsed_args [Namespace]: command line arguments
    """

    # main parser
    parser = argparse.ArgumentParser(
        prog="Feature-detection-and-extraction",
        description="""
        Feature detection and extraction:
        Second assignment of the Computer Vision course at University of Trento
        """,
    )

    # subparsers: so as to have command line options for sub-functions
    subparsers = parser.add_subparsers(help="sub-commands help")

    # parse the command line arguments
    parsed_args = parser.parse_args()

    # if function not passed, then print the usage and exit the program
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)

    return parsed_args


def main(args: Namespace) -> None:
    """
    Main function
    It runs the `func` function passed to the parsers with the respective
    parameters

    Args:
      args [Namespace]: command line arguments
    """
    # execute the function `func` with args as arguments
    args.func(
        args,
    )


if __name__ == "__main__":
    """
    Main
    Calls the main function with the command line arguments passed as parameters
    """
    main(get_args())
