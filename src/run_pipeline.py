import subprocess
import os
import sys

def run_script(script):
    print(f"Running {script}...")
    script_path = os.path.join(os.path.dirname(__file__), script)
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}:")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.output}")
        print("\nContinuing with the next script...\n")

def main():
    scripts = [
       # "data_exploration.py",
        "customer_segmentation.py",
        "customer_classification.py",
        "credit_recommendation.py"
    ]

    for script in scripts:
        run_script(script)

    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()