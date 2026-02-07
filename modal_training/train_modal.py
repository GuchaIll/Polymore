import modal

import sys
import subprocess
import os

app = modal.App("polymer-prediction-training")

# Define the volume to persist data and models
volume = modal.Volume.from_name("polymer-volume", create_if_missing=True)

# Define the image with all necessary dependencies
# Define the image with all necessary dependencies and local code
# 1. Main Image for BERT, Tabular (Autogluon), etc.
image_main = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "wget")
    .pip_install(
        "torch",
        "pandas", # Will install latest compatible (likely >= 2.0)
        "numpy",
        "scikit-learn",
        "transformers",
        "accelerate",
        "optuna",
        "polars",
        "rdkit",
        "xgboost",
        "catboost",
        "lightgbm",
        "kaggle",
        "openpyxl",
        "sentence-transformers",
        "torch-geometric",
        "autogluon", # Requires pandas >= 2.0
        # "calm-model", # Not found on PyPI, assumes user has it or custom
    )
    .env({"MODAL_IS_RUNNING": "1"})
    .add_local_dir("../tolga-workspace/NeurIPS-polymer-prediction", remote_path="/root", ignore=[".git", "__pycache__", "*.pyc", "modal_training"])
    .add_local_file("bert_pretrain.py", remote_path="/root/bert/pretrain.py")
    .add_local_file("bert_supervised_finetune.py", remote_path="/root/bert/supervised_finetune.py")
    .add_local_file("train_metric_predictor.py", remote_path="/root/simulations/train_metric_predictor.py")
    .add_local_file("tabular_train.py", remote_path="/root/tabular/train.py")
)

# 2. Uni-Mol Image (requires older pandas)
image_unimol = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "wget")
    .pip_install(
        "torch",
        "pandas<2.0.0", # Downgrade pandas for Uni-Mol
        "numpy",
        "scikit-learn",
        "optuna",
        "rdkit",
        "kaggle",
        # "unimol_tools", 
        "git+https://github.com/dptech-corp/Uni-Mol.git#subdirectory=unimol_tools",
    )
    .env({"MODAL_IS_RUNNING": "1"})
    .add_local_dir("../tolga-workspace/NeurIPS-polymer-prediction", remote_path="/root", ignore=[".git", "__pycache__", "*.pyc", "modal_training"])
    .add_local_file("unimol_train.py", remote_path="/root/uni_mol/train.py")
)

# Constants
REMOTE_PROJECT_ROOT = "/root"
DATA_DIR = "/data"
MODEL_DIR = "/models"

def setup_persistence():
    os.chdir(REMOTE_PROJECT_ROOT)
    os.makedirs("/persistent/data", exist_ok=True)
    os.makedirs("/persistent/models", exist_ok=True)
    
    if not os.path.exists("data"):
        os.symlink("/persistent/data", "data")
    if not os.path.exists("models"):
        os.symlink("/persistent/models", "models")
        
    # Also ensure subdirectories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("simulations/models", exist_ok=True)
    os.makedirs(f"models/UniMol2_2025_09_07_TabM", exist_ok=True)


@app.function(
    image=image_main,
    volumes={"/persistent": volume},
    timeout=86400,
)
def download_data():
    setup_persistence()

    if os.path.exists("data/from_host/train.csv"):
        print("Data appears to be present. Skipping download.")
        return

    print("Starting data download...")
    # ... (rest of download logic same as before, user provides data manually or via kaggle)
    # Since kaggle might not be configured, we assume manual upload or existing data for now.
    pass

@app.function(
    image=image_main,
    volumes={"/persistent": volume},
    gpu="A10G",
    timeout=86400,
)
def train_main_pipeline():
    setup_persistence()
    print("Starting Main Training Pipeline (BERT, Tabular, etc.)...")

    # 1. BERT Pretraining
    print("\n>>> Running BERT Pretraining...")
    subprocess.run(["python", "bert/pretrain.py"], check=True)

    # 2. BERT Supervised Finetuning
    print("\n>>> Running BERT Supervised Finetuning...")
    subprocess.run(["python", "bert/supervised_finetune.py"], check=True)

    # 3. Simulation Metric Predictor
    print("\n>>> Running Simulation Metric Predictor...")
    subprocess.run(["python", "simulations/train_metric_predictor.py"], check=True)

    # 4. Tabular Training
    print("\n>>> Running Tabular Training...")
    subprocess.run(["python", "tabular/train.py"], check=True)


@app.function(
    image=image_unimol,
    volumes={"/persistent": volume},
    gpu="A10G",
    timeout=86400,
)
def train_unimol_pipeline():
    setup_persistence()
    print("Starting Uni-Mol Training Pipeline...")

    # 5. Uni-Mol Training
    print("\n>>> Running Uni-Mol Training...")
    subprocess.run(["python", "uni_mol/train.py"], check=True)


@app.local_entrypoint()
def main():
    # Trigger functions in order
    # download_data.remote() 
    
    print("Launching Main Pipeline...")
    train_main_pipeline.remote()
    
    print("Launching Uni-Mol Pipeline...")
    train_unimol_pipeline.remote()

