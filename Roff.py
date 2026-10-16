#!/usr/bin/env python3
"""Command-line entry point for roof-material segmentation."""

from roofseg import *
from roofseg.cli import cmd_predict, main
from roofseg.training import cmd_train, confusion, evaluate_tiles, make_cfg

if __name__ == "__main__":
    main()
