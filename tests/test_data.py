import os
from pathlib import Path


def test_folder_existing():
    current_file_path = Path(__file__).resolve().parents[1]

    directories = [
        "data/raw_data",
        "config"
    ]
    
    results = [os.path.isdir(os.path.join(current_file_path, directory)) for directory in directories]
    
    for res in results:
        print(results)
        assert res is True, "Folder does not exist"


def test_data_existing():
    current_file_path = Path(__file__).resolve().parents[1]
    directory = os.path.join(current_file_path, "data/raw_data")

    assert os.path.isdir(directory) is True, ValueError(f"{directory} is not a valid directory")
    
    for filename in os.listdir(directory):
        assert filename.endswith('.csv') is True, "There are not CSV files"
