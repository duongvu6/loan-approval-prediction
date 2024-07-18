import subprocess
from pathlib import Path
import os


def test_preprocess_data():
    files = [
        "transformer.pkl",
        "train.pkl",
        "val.pkl"
    ]
    
    main_path = Path(__file__).resolve().parents[1]

    clear_data = []

    
    for file in files:
        clear_data.append(os.path.join(os.path.join(main_path, "data/clear_data"), file))

    for file_name in clear_data:
        assert os.path.isfile(file_name) is True, f"{file_name} does not exist"
    
