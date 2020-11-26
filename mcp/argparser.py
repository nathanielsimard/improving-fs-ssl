import argparse
import os


def parse_train_arguments():
    parser = _default_parser()
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
        "-e", "--epoch", help="Checkpoint epoch to load.", type=int, required=False,
    )
    return parser.parse_args()


def parse_eval_arguments():
    parser = _default_parser()
    parser.add_argument(
        "-r", "--result", help="Path to the result directory.", type=str, required=True,
    )
    return parser.parse_args()


# For now both are the same
parse_viz_arguments = parse_eval_arguments


def _default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        help="Device to run on.",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
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
    return parser


def initialize_logging(logging_tags, output, debug):
    from mcp.utils import logging

    logging_file = "file" in logging_tags
    logging_std = "std" in logging_tags

    if logging_file:
        logging_file_path = os.path.join(output, "experiment.log")
        logging.initialize(file_name=logging_file_path, std=logging_std, debug=debug)
    else:
        logging.initialize(std=logging_std, debug=debug)
