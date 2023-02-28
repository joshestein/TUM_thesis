import os
from pathlib import Path


def setup_dirs(root_dir=Path(os.getcwd())):
    data_dir = root_dir / "data"
    log_dir = root_dir / "logs"
    out_dir = root_dir / "out"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    return data_dir, log_dir, out_dir
