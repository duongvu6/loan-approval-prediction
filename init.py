import subprocess

def main():
    models_location = [
        "db.py",
        "preprocess_data.py",
        "./hpo/models_hpo.py",
        "reg_model.py",
        "check_accuracy.py"
    ]
    for path in models_location:
        print(f"File {path}... ", end="")
        subprocess.run(['python', path], capture_output=True, text=True)
        print("Done.")

    print("Run pytest")
    subprocess.run(['pytest'])

    print("File app.py run on 8000 port...")
    subprocess.run(['python', "app.py"], capture_output=True, text=True)


if __name__ == "__main__":
    main()