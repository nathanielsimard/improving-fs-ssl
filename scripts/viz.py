#!/usr/bin/env python
import os

from mcp.argparser import initialize_logging, parse_viz_arguments


def run(args):
    from mcp import main
    from mcp.config.loader import load
    from mcp.config.parser import parse

    config_path = os.path.join(args.result, "config_full.yml")
    configs = load(config_path)
    config_experiment = parse([configs])

    main.run_viz(config_experiment, args.result, args.device)  # type: ignore


if __name__ == "__main__":
    args = parse_viz_arguments()
    initialize_logging(args.logging, args.result, args.debug)
    run(args)
