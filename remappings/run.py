import os
import subprocess

# Create data folder
if not os.path.exists("data"):
    os.makedirs("data")


# List of scripts to call
scripts = ["src/data_to_dict.py", "src/data_mapping.py"]

for script in scripts:
    try:
        # Call the script and wait for it to complete
        result = subprocess.run(
            ["python", script], check=True, text=True, capture_output=True
        )
        print(f"Output of {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error calling {script}: {e.stderr}")
        print("Probably missing the data set in root folder /data")
