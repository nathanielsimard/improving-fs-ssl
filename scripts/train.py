#!/usr/bin/env python
import os

from mcp.argparser import initialize_logging, parse_train_arguments


def run(args):
    from mcp import main
    from mcp.config.loader import load, save, to_dict
    from mcp.config.parser import parse

    configs = [load(c) for c in args.config]
    config_experiment = parse(configs)

    config_full = to_dict(config_experiment)
    save(config_full, os.path.join(args.output, "config_full.yml"))

    main.run_train(config_experiment, args.output, args.device)  # type: ignore


if __name__ == "__main__":
    args = parse_train_arguments()
    os.makedirs(args.output, exist_ok=True)
    initialize_logging(args.logging, args.output, args.debug)
    run(args)
