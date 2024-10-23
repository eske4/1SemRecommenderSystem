import os
import subprocess


def setup_remappings():
    print("Setting up remappings, wait a while")

    remappings_directory = "remappings"  # Change this to your desired directory
    # Change the working directory
    os.chdir(remappings_directory)

    scripts = ["run.py"]

    for script in scripts:
        # Call the script and wait for it to complete
        result = subprocess.run(
            ["python", script], check=True, text=True, capture_output=True
        )
        print(f"Output of {script}:\n{result.stdout}")

    os.chdir("..")


if __name__ == "__main__":
    setup_remappings()
