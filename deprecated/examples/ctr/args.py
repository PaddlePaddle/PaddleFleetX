import subprocess
import os
import sys
from argparse import ArgumentParser, REMAINDER
import argparse as argparse

def parse_args():
        parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
        parser.add_argument(
            '--is_local',
            type=int,
            default=0,
            help='Local train or distributed train (default: 1)')
        return parser.parse_args()
