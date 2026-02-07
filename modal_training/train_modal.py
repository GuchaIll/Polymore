import modal
import sys
import subprocess
import os
import json
from pathlib import Path

app = modal.App("polymer-prediction-training")

# Define the volume to persist data and models
volume = modal.Volume.from_name("polymer-volume", create_if_missing=True)

# 1. Main Image for BERT, Tabular (Autogluon), etc.
image_main = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "wget")
    .pip_install(
        "torch",
        "pandas", 
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
        "autogluon", 
    )
    .env({"MODAL_IS_RUNNING": "1", "CACHE_BUST": "20260207"})
    .add_local_dir("../tolga-workspace/NeurIPS-polymer-prediction", remote_path="/root", ignore=[".git", "__pycache__", "*.pyc", "modal_training"])
)

# 2. Uni-Mol Image (requires older pandas)
image_unimol = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "wget", "libxrender1", "libxext6", "libsm6", "libgl1")
    .pip_install(
        "torch",
        "pandas<2.0.0", 
        "numpy",
        "scikit-learn",
        "optuna",
        "polars",
        "transformers",
        "accelerate",
        "sentence-transformers",
        "torch-geometric",
        "openpyxl",
        "xgboost",
        "catboost",
        "lightgbm",
        "rdkit",
        "kaggle",
        "git+https://github.com/dptech-corp/Uni-Mol.git#subdirectory=unimol_tools",
    )
    .env({"MODAL_IS_RUNNING": "1", "CACHE_BUST": "20260207"})
    .add_local_dir("../tolga-workspace/NeurIPS-polymer-prediction", remote_path="/root", ignore=[".git", "__pycache__", "*.pyc", "modal_training"])
)

# Constants
REMOTE_PROJECT_ROOT = "/root"
TARGET_NAMES = ["Tg", "FFV", "Tc", "Density", "Rg"]
UNIMOL_TARGETS = ["Rg", "Tc", "Tg", "Density"]

def setup_persistence():
    os.chdir(REMOTE_PROJECT_ROOT)
    os.makedirs("/persistent/data", exist_ok=True)
    os.makedirs("/persistent/models", exist_ok=True)
    
    if not os.path.exists("data"):
        os.symlink("/persistent/data", "data")
    if not os.path.exists("models"):
        os.symlink("/persistent/models", "models")
        
    os.makedirs("models", exist_ok=True)
    os.makedirs("simulations/models", exist_ok=True)
    os.makedirs(f"models/UniMol2_2025_09_07_TabM", exist_ok=True)
    # Ensure intermediate directories exist
    os.makedirs(f"models/UniMol2_2025_09_07_TabM", exist_ok=True)
    
    # Persist relabeling results
    os.makedirs("/persistent/results", exist_ok=True)
    results_dir = "data_preprocessing/results"
    if os.path.exists(results_dir):
        if not os.path.islink(results_dir):
            import shutil
            shutil.rmtree(results_dir)
            os.symlink("/persistent/results", results_dir)
    else:
        # Parent dir might not exist if running minimal image?
        os.makedirs(os.path.dirname(results_dir), exist_ok=True)
        os.symlink("/persistent/results", results_dir)

# --- Data Download ---

@app.function(
    image=image_main,
    volumes={"/persistent": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("my-kaggle-secret")],
)
def download_data():
    setup_persistence()
    if os.path.exists("data/PI1M_pseudolabels/PI1M_50000_v2.1.csv"):
        print("Data appears to be present. Skipping download.")
        return

    print("Starting data download...")
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
         print("WARNING: KAGGLE_USERNAME/KEY not found.")
    else:
        raw_key = os.environ["KAGGLE_KEY"]
        if raw_key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = raw_key
            if "KAGGLE_KEY" in os.environ: del os.environ["KAGGLE_KEY"]

    commands = [
        "mkdir -p data",
        "mkdir -p models",
        "cd data && kaggle competitions download -c neurips-open-polymer-prediction-2025",
        "cd data && kaggle datasets download -d jsday96/pi1m-pseudolabels",
        "cd data && kaggle datasets download -d dmitryuarov/smiles-extra-data",
        "cd data && kaggle datasets download -d jsday96/polymer-merged-extra-host-data",
        "cd data && kaggle datasets download -d jsday96/md-simulation-results",
        "cd data && wget https://zenodo.org/records/15210035/files/LAMALAB_CURATED_Tg_structured_polymerclass.csv",
        "cd data && wget -O PI1070.csv https://raw.githubusercontent.com/RadonPy/RadonPy/develop/data/PI1070.csv",
        "cd data && unzip -o neurips-open-polymer-prediction-2025.zip -d from_host",
        "cd data && unzip -o pi1m-pseudolabels.zip -d PI1M_pseudolabels",
        "cd data && unzip -o smiles-extra-data.zip -d smiles_extra_data",
        "cd data && unzip -o polymer-merged-extra-host-data.zip -d from_host",
        "cd data && unzip -o md-simulation-results.zip -d md_simulation_results",
        "cd models && kaggle datasets download -d jsday96/polymer-relabeling-models",
        "cd models && unzip -o polymer-relabeling-models.zip",
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)

# --- Relabeling Tasks ---

@app.function(image=image_main, volumes={"/persistent": volume}, timeout=3600)
def relabel_prepare_smiles():
    setup_persistence()
    print("Running Relabeling: Prepare SMILES")
    subprocess.run(["python", "data_preprocessing/relabeling.py", "--task", "prepare_smiles"], check=True)

@app.function(image=image_main, volumes={"/persistent": volume}, gpu="A10G", timeout=3600)
def relabel_bert(target, fold):
    setup_persistence()
    output_path = f"data_preprocessing/results/{target}/bert_fold_{fold}.csv"
    if os.path.exists(output_path):
        print(f"Skipping BERT {target} Fold {fold} - already exists")
        return
    print(f"Running Relabeling: BERT {target} Fold {fold}")
    cmd = [
        "python", "data_preprocessing/relabeling.py", 
        "--task", "bert", 
        "--target", target, 
        "--fold", str(fold),
        "--input", "data_preprocessing/results/extra_smiles.csv",
        "--output", output_path
    ]
    print(f"DEBUG COMMAND: {cmd}")
    subprocess.run(cmd, check=True)

@app.function(image=image_main, volumes={"/persistent": volume}, gpu="A10G", timeout=3600)
def relabel_dmpnn(target):
    setup_persistence()
    output_path = f"data_preprocessing/results/{target}/dmpnn.csv"
    if os.path.exists(output_path):
        print(f"Skipping DMPNN {target} - already exists")
        return
    print(f"Running Relabeling: DMPNN {target}")
    cmd = [
        "python", "data_preprocessing/relabeling.py", 
        "--task", "dmpnn", 
        "--target", target, 
        "--input", "data_preprocessing/results/extra_smiles.csv",
        "--output", output_path
    ]
    print(f"DEBUG COMMAND: {cmd}")
    subprocess.run(cmd, check=True)

@app.function(image=image_main, volumes={"/persistent": volume}, gpu="A10G", timeout=3600)
def relabel_tabular(target):
    setup_persistence()
    output_path = f"data_preprocessing/results/{target}/tabular.csv"
    if os.path.exists(output_path):
        print(f"Skipping Tabular {target} - already exists")
        return
    print(f"Running Relabeling: Tabular {target}")
    cmd = [
        "python", "data_preprocessing/relabeling.py", 
        "--task", "tabular", 
        "--target", target, 
        "--input", "data_preprocessing/results/extra_smiles.csv",
        "--output", output_path
    ]
    print(f"DEBUG COMMAND: {cmd}")
    subprocess.run(cmd, check=True)

@app.function(image=image_unimol, volumes={"/persistent": volume}, gpu="A100", timeout=7200)
def relabel_unimol(target):
    setup_persistence()
    output_path = f"data_preprocessing/results/{target}/unimol.csv"
    if os.path.exists(output_path):
        print(f"Skipping Uni-Mol {target} - already exists")
        return
    print(f"Running Relabeling: Uni-Mol {target}")
    cmd = [
        "python", "data_preprocessing/relabeling.py", 
        "--task", "unimol", 
        "--target", target, 
        "--input", "data_preprocessing/results/extra_smiles.csv",
        "--output", output_path
    ]
    print(f"DEBUG COMMAND: {cmd}")
    subprocess.run(cmd, check=True)

@app.function(image=image_main, volumes={"/persistent": volume}, timeout=3600)
def relabel_merge():
    setup_persistence()
    print("Merging Relabeling Results...")
    subprocess.run(["python", "data_preprocessing/relabeling.py", "--task", "merge"], check=True)
    subprocess.run(["python", "data_preprocessing/merge_updated_labels.py"], check=True)
    subprocess.run(["python", "uni_mol/unimol_datasets/get_extra_data.py"], check=True)

# --- Training Tasks ---

@app.function(image=image_main, volumes={"/persistent": volume}, gpu="A100", timeout=86400)
def train_bert(target, fold):
    setup_persistence()
    print(f"Training BERT: {target} Fold {fold}")
    # Note: supervised_finetune writes to models/ so persistence is handled by setup_persistence mapping
    subprocess.run([
        "python", "bert/supervised_finetune.py", 
        "--target", target, 
        "--fold", str(fold)
    ], check=True)

@app.function(image=image_main, volumes={"/persistent": volume}, gpu="A100", timeout=86400)
def train_tabular(target):
    setup_persistence()
    print(f"Training Tabular: {target}")
    subprocess.run([
        "python", "tabular/train.py", 
        "--target", target
    ], check=True)

@app.function(image=image_main, volumes={"/persistent": volume}, gpu="A100", timeout=86400)
def train_metric_predictor():
    setup_persistence()
    print("Training Metric Predictor...")
    subprocess.run(["python", "simulations/train_metric_predictor.py"], check=True)

@app.function(image=image_unimol, volumes={"/persistent": volume}, gpu="A100", timeout=86400)
def train_unimol_pipeline():
    setup_persistence()
    print("Training Uni-Mol Pipeline...")
    subprocess.run(["python", "uni_mol/train.py"], check=True)

# --- Orchestration ---

@app.local_entrypoint()
def main():
    print(">>> Launching Data Download...")
    download_data.remote()
    
    print("\n>>> Launching Relabeling Phase 1: Prepare SMILES")
    relabel_prepare_smiles.remote()
    
    print("\n>>> Launching Relabeling Phase 2: Parallel Predictions")
    bert_tasks = [(target, fold) for target in TARGET_NAMES for fold in range(5)]
    dmpnn_tasks = TARGET_NAMES
    tabular_tasks = TARGET_NAMES
    unimol_tasks = UNIMOL_TARGETS
    
    # Launch all relabeling tasks in parallel
    print(f"Spawning {len(bert_tasks)} BERT tasks, {len(dmpnn_tasks)} DMPNN tasks, {len(tabular_tasks)} Tabular tasks, {len(unimol_tasks)} Uni-Mol tasks...")
    
    # We use list() to force execution and wait for completion
    results_bert = list(relabel_bert.starmap(bert_tasks))
    results_dmpnn = list(relabel_dmpnn.map(dmpnn_tasks))
    results_tabular = list(relabel_tabular.map(tabular_tasks))
    results_unimol = list(relabel_unimol.map(unimol_tasks))
    
    print("\n>>> Launching Relabeling Phase 3: Merge Results")
    relabel_merge.remote()
    
    print("\n>>> Launching Training Phase")
    
    # Training Tasks
    bert_train_tasks = [(target, fold) for target in TARGET_NAMES for fold in range(5)]
    tabular_train_tasks = TARGET_NAMES
    
    print(f"Spawning {len(bert_train_tasks)} BERT training tasks...")
    # BERT training can be massive, launch parallel
    bert_train_results = list(train_bert.starmap(bert_train_tasks))
    
    print(f"Spawning {len(tabular_train_tasks)} Tabular training tasks...")
    tabular_train_results = list(train_tabular.map(tabular_train_tasks))
    
    print("Spawning Metric Predictor training...")
    train_metric_predictor.remote()
    
    print("Spawning Uni-Mol pipeline...")
    train_unimol_pipeline.remote()
    
    print("\n>>> All Orchestration Completed Successfully!")
