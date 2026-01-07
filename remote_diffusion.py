# import subprocess
# import time
# import os

# def run_kaggle_diffusion(prompt):
#     # Upload snapshot
#     subprocess.run([
#         "kaggle", "datasets", "version",
#         "-p", "snapshot",
#         "-m", "new snapshot"
#     ])

#     # Trigger notebook (manual run)
#     print("⚠️ Run the Kaggle notebook manually with prompt:")
#     print(prompt)

#     input("Press ENTER after Kaggle finishes...")

#     # Download result
#     subprocess.run([
#         "kaggle", "datasets", "download",
#         "your-username/your-dataset",
#         "-p", "results",
#         "--unzip"
#     ])
def run_kaggle_diffusion(prompt):
    print("\n==============================")
    print("KAGGLE DIFFUSION STEP")
    print("==============================")
    print("1. Open your Kaggle notebook")
    print("2. Ensure snapshot/input.png is uploaded as dataset version")
    print("3. Run the notebook")
    print("4. Download output.png into results/")
    print("\nPrompt used:")
    print(prompt)
    print("\nPress ENTER when output.png is ready...")
    input()
