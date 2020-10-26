#!/usr/bin/env python
from mcp.argparser import parse_eval_arguments, initialize_logging
import os


def run(args):
    from mcp import main
    from mcp.config.loader import load
    from mcp.config.parser import parse

    config_path = os.path.join(args.result, "config_full.yml")
    configs = load(config_path)
    config_experiment = parse([configs])

    main.run_eval(config_experiment, args.result, args.device)  # type: ignore


if __name__ == "__main__":
    args = parse_eval_arguments()
    os.makedirs(args.output, exist_ok=True)
    initialize_logging(args)
    run(args)
