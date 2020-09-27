#!/usr/bin/env python
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config files."
        + " Multiple config files is supported with the last one having the highest priority."
        + " It can be usefull by reducing the dupplication in settings between experiemnts."
        + " E.g --config specific_training.yml specific_dataset.yml specific_model.yml",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output directory.", type=str, required=True,
    )
    parser.add_argument(
        "-l",
        "--logging",
        help="Set the location of the logs. Possible values are 'std' and 'file'",
        type=str,
        nargs="+",
        default=["std", "file"],
    )
    parser.add_argument(
        "--debug", action="store_true", help="Activate more logging to help debugging."
    )
    return parser.parse_args()


def run(args):
    from mcp.config.loader import load, save, to_dict
    from mcp.config.parser import parse

    configs = [load(c) for c in args.config]
    config_experiment = parse(configs)

    config_full = to_dict(config_experiment)
    save(config_full, os.path.join(args.output, "config_full.yml"))

    print(config_experiment)


def _initialize_logging(args):
    from mcp.utils import logging

    logging_file = "file" in args.logging
    logging_std = "std" in args.logging

    if logging_file and args.config is not None:
        logging_file_path = os.path.join(args.output, "experiment.log")
        logging.initialize(
            file_name=logging_file_path, std=logging_std, debug=args.debug
        )
    else:
        logging.initialize(std=logging_std, debug=args.debug)


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    _initialize_logging(args)
    run(args)
