import subprocess

def main():
    models_location = ["./hpo/rfc.py", "./hpo/xgb.py"]
    for id, path in enumerate(models_location):
        print(f"Model {id} optimization... ", end="")
        subprocess.run(['python', path], capture_output=True, text=True)
        print("Done.")


if __name__ == "__main__":
    main()