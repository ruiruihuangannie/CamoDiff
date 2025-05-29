from pathlib import Path
import os
import sys

# set env variable to use the correct GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# get current file path
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = Path(current_file_path).parent

# add source folder to path
sys.path.insert(0, str(project_root_path / "denoising-diffusion-pytorch"))

def resolve_path(path_dir: str) -> str:
    """
    Resolves a path string to an absolute path, handling both relative and absolute paths.
    """
    abs_path = Path(path_dir).resolve()
    return str(abs_path)


if __name__ == "__main__":
    print(sys.path)