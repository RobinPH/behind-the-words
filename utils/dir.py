import glob
import os
from pathlib import Path


def make_dir(_dir):
    os.makedirs(os.path.dirname(_dir), exist_ok=True)

    return _dir


def get_latest_file_in_dir(_dir):
    make_dir(_dir)

    files = glob.glob(os.path.join(_dir, "*"))

    return max(files, key=os.path.getctime)


def get_filename(_dir):
    return Path(_dir).stem
