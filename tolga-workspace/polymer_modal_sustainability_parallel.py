"""
Modal-based Polymer Prediction Inference Pipeline with Sustainability Metrics

This script runs the polymer property prediction inference on Modal's serverless infrastructure.
It generates predictions using three different model types:
1. XGBoost/RandomForest ensemble
2. BERT models
3. TabTransformer models

Additionally, it supports sustainability metrics prediction:
4. Sustainability models (XGBoost + DeBERTa transfer learning)
   - Recyclability
   - BioSource
   - Environmental Safety
   - Synthesis Efficiency

Usage:
    modal run polymer_modal_sustainability.py
    modal run polymer_modal_sustainability.py --skip-download
    modal run polymer_modal_sustainability.py --train-sustainability
    modal run polymer_modal_sustainability.py --predict-sustainability
"""

import modal
from pathlib import Path

# ============================================================================
# Modal Infrastructure Setup
# ============================================================================

# Create Modal volumes for persistent storage
# Note: Modal volumes persist indefinitely until explicitly deleted
# Data will remain available across all runs
data_volume = modal.Volume.from_name("polymer-data", create_if_missing=True)
results_volume = modal.Volume.from_name("polymer-results", create_if_missing=True)
models_volume = modal.Volume.from_name("polymer-models", create_if_missing=True)  # For sustainability models

# Volume retention: Modal volumes have no automatic expiration
# Data persists until manually deleted via: modal volume delete <name>
# To view volumes: modal volume list
# To inspect contents: modal volume ls <name>

# Define container images for different model types
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas~=2.0.0",
        "numpy~=1.24.0",
        "openpyxl~=3.1.0",  # For Excel file reading
    )
)

ml_image = (
    base_image
    .pip_install(
        "pandas~=2.0.0",
        "scikit-learn~=1.3.0",
        "xgboost~=2.0.0",
        "catboost~=1.2.0",
        "networkx~=3.1.0",
    )
    .run_commands(
        # Install RDKit from conda-forge
        "apt-get update",
        "apt-get install -y wget",
    )
    .pip_install(
        "rdkit~=2023.9.1",
    )
)

dl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch~=2.2.0",
        "transformers~=4.38.0",
        "pandas~=2.0.0",
        "numpy~=1.24.0",
        "scikit-learn~=1.3.0",
        "joblib~=1.3.0",
        "tqdm~=4.66.0",
        "networkx~=3.1.0",
        "openpyxl~=3.1.0",
        "xgboost~=2.0.0",  # Added for sustainability models
    )
    .pip_install(
        "rdkit~=2023.9.1",
    )
)

# Create Modal app
app = modal.App("polymer-prediction")

# Path constants
DATA_PATH = Path("/data")
RESULTS_PATH = Path("/results")
MODELS_PATH = Path("/models")  # NEW: For sustainability models

# ============================================================================
# Data Download Function
# ============================================================================

@app.function(
    image=base_image
        .pip_install("kaggle~=1.6.0")
        .add_local_dir(
            local_path=Path(__file__).parent / "neurips-open-polymer-prediction-2025",
            remote_path="/local-competition-data"
        ),
    volumes={DATA_PATH: data_volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],
    timeout=3600,  # 1 hour for downloads
)
def download_datasets():
    """Download all required Kaggle datasets to Volume"""
    import time
    pipeline_start = time.time()
    import os
    import json
    import zipfile
    import shutil
    from pathlib import Path
    
    print("📦 Downloading Kaggle datasets to Modal Volume...")
    
    # First, copy the local competition data if needed
    comp_dest = DATA_PATH / "neurips-open-polymer-prediction-2025"
    if not list(comp_dest.glob("*")):
        print("   📂 Copying local competition data...")
        comp_dest.mkdir(exist_ok=True, parents=True)
        shutil.copytree("/local-competition-data", comp_dest, dirs_exist_ok=True)
        print("   ✅ Competition data copied from local directory")
    else:
        print("   ✅ Competition data already exists, skipping")
    
    # Configure Kaggle credentials for other datasets
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    # Handle both old (username+key) and new (KGAT token) formats
    if 'KAGGLE_KEY' in os.environ and os.environ['KAGGLE_KEY'].startswith('KGAT_'):
        # New KGAT token format - use with username
        config = {
            'username': os.environ.get('KAGGLE_USERNAME', ''),
            'key': os.environ['KAGGLE_KEY']
        }
    else:
        # Old format
        config = {
            'username': os.environ.get('KAGGLE_USERNAME', ''),
            'key': os.environ.get('KAGGLE_KEY', '')
        }
    
    with open(kaggle_json, 'w') as f:
        json.dump(config, f)
    kaggle_json.chmod(0o600)
    
    print(f"✅ Kaggle credentials configured at {kaggle_json}")
    
    # Now import kaggle API (after credentials are set up)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    print("✅ Kaggle API authenticated successfully")
    
    # Define dataset mappings (excluding competition data which is copied locally)
    datasets = {
        "external-polymer-data": ("dataset", "tasmim/external-polymer-data"),
        "modred-dataset": ("dataset", "adamlogman/modred-dataset"),
        "private-smile-bert-models": ("dataset", "defdet/private-smile-bert-models"),
        "rdkit-2025-3-3-cp311": ("dataset", "senkin13/rdkit-2025-3-3-cp311"),
        "smiles-extra-data": ("dataset", "dmitryuarov/smiles-extra-data"),
        "smiles-bert-models": ("dataset", "defdet/smiles-bert-models"),
        "smiles-deberta77m-tokenizer": ("dataset", "defdet/smiles-deberta77m-tokenizer"),
        "tc-smiles": ("dataset", "minatoyukinaxlisa/tc-smiles")
    }
    
    def download_and_extract(dataset_name, dataset_type, dataset_path):
        """Download and extract a Kaggle dataset"""
        dest_dir = DATA_PATH / dataset_name
        dest_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if already downloaded
        if list(dest_dir.glob("*")):
            print(f"   ✅ {dataset_name} already exists, skipping download")
            return True
        
        print(f"   ⬇️  Downloading {dataset_name}...")
        try:
            if dataset_type == "competition":
                # Download competition files
                api.competition_download_files(dataset_path, path=str(dest_dir))
            else:
                # Download dataset
                api.dataset_download_files(dataset_path, path=str(dest_dir), unzip=True)
            
            # Unzip any remaining zip files
            for zip_file in dest_dir.glob("*.zip"):
                print(f"   📂 Extracting {zip_file.name}...")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(dest_dir)
                    zip_file.unlink()
                except Exception as e:
                    print(f"   ⚠️  Could not extract {zip_file.name}: {e}")
            
            print(f"   ✅ {dataset_name} downloaded successfully")
            return True
        except Exception as e:
            print(f"   ⚠️  Error downloading {dataset_name}: {e}")
            if dataset_type == "competition":
                print(f"   💡 Make sure you've accepted the competition rules at:")
                print(f"      https://www.kaggle.com/competitions/{dataset_path}/rules")
            return False
    
    # Download all datasets
    success_count = 0
    for dataset_name, (dataset_type, dataset_path) in datasets.items():
        if download_and_extract(dataset_name, dataset_type, dataset_path):
            success_count += 1
    
    # Commit volume changes to persist data
    # This ensures all downloaded data is saved to the persistent volume
    data_volume.commit()
    print(f"\n✅ Downloaded {success_count}/{len(datasets)} datasets and committed to Volume!")
    print("💾 Data persisted to Modal volume 'polymer-data' (no expiration)")
    print("📊 Volume info: modal volume ls polymer-data")
    print()
    
    if success_count < len(datasets):
        print("⚠️  Some datasets failed to download. Check the logs above for details.")
    
    total_time = time.time() - pipeline_start
    print(f"⏱️  Total download time: {total_time:.2f}s ({total_time/60:.2f}m)")



# ============================================================================
# Sustainability Feature Engineering & Target Generation
# ============================================================================

def get_sustainability_features(mol):
    """Generate raw features for ML input (X) - Sustainability-specific features"""
    from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors, Crippen
    from rdkit import Chem
    
    if not mol: 
        return None
    
    # 1. Complexity (Energy Proxy)
    feat_synth_complexity = GraphDescriptors.BertzCT(mol)
    
    # 2. Physical Properties
    feat_logp = Crippen.MolLogP(mol)
    feat_tpsa = Descriptors.TPSA(mol)
    
    # 3. Structure
    n_heavy = mol.GetNumHeavyAtoms()
    n_rot = Descriptors.NumRotatableBonds(mol)
    feat_rot_density = n_rot / n_heavy if n_heavy > 0 else 0
    
    n_aromatic = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
    feat_aromatic_prop = n_aromatic / n_heavy if n_heavy > 0 else 0
    
    # 4. Bio-Logic
    feat_fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    feat_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    # 5. Functional Groups (Counts)
    patterns = {
        "feat_ester": "[CX3](=[OX1])[OX2]",
        "feat_amide": "[CX3](=[OX1])[NX3]",
        "feat_carbonate": "[OX2][CX3](=[OX1])[OX2]",
        "feat_pfas": "[#6](F)(F)[#6](F)(F)",
        "feat_isocyanate": "N=C=O"
    }
    
    counts = {}
    for k, pat in patterns.items():
        try:
            counts[k] = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pat)))
        except:
            counts[k] = 0
    
    # 6. Topological Torsion (for steric access / 3D hindrance proxy)
    try:
        torsion_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=2048)
        # Aggregate to simple features: count of set bits, density
        feat_torsion_count = torsion_fp.GetNumOnBits()
        feat_torsion_density = feat_torsion_count / 2048
    except:
        feat_torsion_count = 0
        feat_torsion_density = 0
        
    return {
        "feat_synth_complexity": feat_synth_complexity,
        "feat_rot_density": feat_rot_density,
        "feat_aromatic_prop": feat_aromatic_prop,
        "feat_fsp3": feat_fsp3,
        "feat_chiral": feat_chiral,
        "feat_logp": feat_logp,
        "feat_tpsa": feat_tpsa,
        "feat_torsion_count": feat_torsion_count,
        "feat_torsion_density": feat_torsion_density,
        **counts
    }


def generate_synthetic_targets(mol):
    """Generate Teacher Labels (Y) for Weak Supervision"""
    from rdkit.Chem import Descriptors, rdMolDescriptors, Chem, GraphDescriptors
    
    if not mol: 
        return None
    
    mw = Descriptors.MolWt(mol)
    
    # A. Recyclability (Density of breakable bonds)
    recyclable_counts = 0
    for pattern in ["[CX3](=[OX1])[OX2]", "[CX3](=[OX1])[NX3]"]:
        try:
            recyclable_counts += len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
        except:
            pass
    target_recyclability = min(100.0, (recyclable_counts * 200) / (mw + 1) * 50.0)

    # B. Bio-Source (Sp3 + Chirality)
    f_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    target_bio = f_sp3 * 100.0
    
    # C. Safety (Penalize toxic alerts)
    alerts = 0
    for pattern in ["[#6](F)(F)[#6](F)(F)", "N=C=O", "[Hg,Pb,Cd,As]"]:
        try:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                alerts += 100
        except:
            pass
    target_safety = max(0.0, 100.0 - alerts)
    
    # D. Efficiency (Inverse Complexity)
    try:
        bertz = GraphDescriptors.BertzCT(mol)
        target_efficiency = max(0.0, 100.0 - ((bertz - 200) / 10.0))
    except:
        target_efficiency = 50.0
    
    return {
        "Target_Recyclability": target_recyclability,
        "Target_BioSource": target_bio,
        "Target_EnvSafety": target_safety,
        "Target_SynthEfficiency": target_efficiency
    }


# ============================================================================
# XGBoost/RandomForest Inference
# ============================================================================

@app.function(
    image=ml_image,
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
    timeout=14400,  # 4 hours
    cpu=16,  # High CPU count for parallel feature extraction and XGBoost inference
    memory=32768,  # 32 GB RAM for large datasets
)
def run_xgboost_inference(test_mode: bool = False, sample_size: int = 100):
    """Generate predictions using XGBoost and RandomForest models"""
    import time
    import pandas as pd
    
    start_time = time.time()
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.mixture import GaussianMixture
    from xgboost import XGBRegressor
    from rdkit import Chem
    from rdkit.Chem import Descriptors, MACCSkeys
    from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
    from rdkit.Chem.Descriptors import MolWt, MolLogP
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.Chem import rdmolops
    import networkx as nx
    
    print("🔬 Starting XGBoost/RandomForest inference...")
    
    # Load data
    base_path = DATA_PATH / 'neurips-open-polymer-prediction-2025'
    train = pd.read_csv(base_path / 'train.csv')
    test = pd.read_csv(base_path / 'test.csv')
    
    if test_mode:
        print(f"⚡ TEST MODE: Using only {sample_size} test samples")
        test = test.head(sample_size)
    
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Helper functions from original code
    def clean_and_validate_smiles(smiles):
        """Clean and validate SMILES strings"""
        if not isinstance(smiles, str) or len(smiles) == 0:
            return None
        
        bad_patterns = [
            '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
            "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
            '([R])', '([R1])', '([R2])',
        ]
        
        for pattern in bad_patterns:
            if pattern in smiles:
                return None
        
        if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None
    
    # Clean SMILES
    print("🔄 Cleaning and validating SMILES...")
    train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
    test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)
    train = train[train['SMILES'].notnull()].reset_index(drop=True)
    test = test[test['SMILES'].notnull()].reset_index(drop=True)
    
    # Load external datasets
    def load_external_data():
        """Load and integrate external datasets"""
        datasets = []
        
        # Tc data
        try:
            data_tc = pd.read_csv(DATA_PATH / 'tc-smiles' / 'Tc_SMILES.csv')
            data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})
            datasets.append(('Tc', data_tc))
        except:
            print("⚠️  Could not load Tc data")
        
        # External Tg data
        try:
            data_tg = pd.read_csv(DATA_PATH / 'external-polymer-data' / 'TgSS_enriched_cleaned.csv')
            if 'Tg' in data_tg.columns:
                datasets.append(('Tg', data_tg[['SMILES', 'Tg']]))
        except:
            print("⚠️  Could not load external Tg data")
        
        # JCIM Tg data
        try:
            data_jcim = pd.read_csv(DATA_PATH / 'smiles-extra-data' / 'JCIM_sup_bigsmiles.csv')
            data_jcim = data_jcim[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'})
            datasets.append(('Tg', data_jcim))
        except:
            print("⚠️  Could not load JCIM data")
        
        # Excel Tg data
        try:
            data_tg3 = pd.read_excel(DATA_PATH / 'smiles-extra-data' / 'data_tg3.xlsx')
            data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
            data_tg3['Tg'] = data_tg3['Tg'] - 273.15
            datasets.append(('Tg', data_tg3[['SMILES', 'Tg']]))
        except:
            print("⚠️  Could not load Excel Tg data")
        
        # Density data
        try:
            data_dnst = pd.read_excel(DATA_PATH / 'smiles-extra-data' / 'data_dnst1.xlsx')
            data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})
            data_dnst = data_dnst[['SMILES', 'Density']]
            data_dnst = data_dnst.query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
            data_dnst['Density'] = data_dnst['Density'].astype(float) - 0.118
            datasets.append(('Density', data_dnst))
        except:
            print("⚠️  Could not load Density data")
        
        return datasets
    
    external_datasets = load_external_data()
    print(f"📊 Loaded {len(external_datasets)} external datasets")
    
    # Integrate external data (simplified version)
    train_extended = train[['SMILES'] + TARGETS].copy()
    for target, dataset in external_datasets:
        dataset['SMILES'] = dataset['SMILES'].apply(clean_and_validate_smiles)
        dataset = dataset.dropna(subset=['SMILES', target])
        # Simple merge - add unique SMILES
        unique_smiles = set(dataset['SMILES']) - set(train_extended['SMILES'])
        if len(unique_smiles) > 0:
            extra_to_add = dataset[dataset['SMILES'].isin(unique_smiles)].copy()
            for col in TARGETS:
                if col not in extra_to_add.columns:
                    extra_to_add[col] = np.nan
            train_extended = pd.concat([train_extended, extra_to_add[['SMILES'] + TARGETS]], ignore_index=True)
    
    print(f"📈 Extended training data: {len(train_extended)} samples")
    
    # Feature extraction functions
    def augment_smiles_dataset(smiles_list, labels, num_augments=1):
        """Augment SMILES by generating randomized versions"""
        augmented_smiles = []
        augmented_labels = []
        
        for smiles, label in zip(smiles_list, labels):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            augmented_smiles.append(smiles)
            augmented_labels.append(label)
            for _ in range(num_augments):
                rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
                augmented_smiles.append(rand_smiles)
                augmented_labels.append(label)
        
        return augmented_smiles, np.array(augmented_labels)
    
    def smiles_to_combined_fingerprints_with_descriptors(smiles_list, radius=2, n_bits=128):
        """Generate molecular fingerprints and descriptors"""
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        
        fingerprints = []
        descriptors = []
        valid_smiles = []
        invalid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Fingerprints
                morgan_fp = generator.GetFingerprint(mol)
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                combined_fp = np.concatenate([np.array(morgan_fp), np.array(maccs_fp)])
                fingerprints.append(combined_fp)
                
                # RDKit Descriptors
                descriptor_values = {}
                for name, func in Descriptors.descList:
                    try:
                        descriptor_values[name] = func(mol)
                    except:
                        descriptor_values[name] = None
                
                # Specific descriptors
                descriptor_values['MolWt'] = MolWt(mol)
                descriptor_values['LogP'] = MolLogP(mol)
                descriptor_values['TPSA'] = CalcTPSA(mol)
                descriptor_values['RotatableBonds'] = CalcNumRotatableBonds(mol)
                descriptor_values['NumAtoms'] = mol.GetNumAtoms()
                descriptor_values['SMILES'] = smiles
                
                # Graph-based features
                try:
                    adj = rdmolops.GetAdjacencyMatrix(mol)
                    G = nx.from_numpy_array(adj)
                    
                    if nx.is_connected(G):
                        descriptor_values['graph_diameter'] = nx.diameter(G)
                        descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
                    else:
                        descriptor_values['graph_diameter'] = 0
                        descriptor_values['avg_shortest_path'] = 0
                    
                    descriptor_values['num_cycles'] = len(list(nx.cycle_basis(G)))
                except:
                    descriptor_values['graph_diameter'] = None
                    descriptor_values['avg_shortest_path'] = None
                    descriptor_values['num_cycles'] = None
                
                descriptors.append(descriptor_values)
                valid_smiles.append(smiles)
            else:
                fingerprints.append(np.zeros(n_bits + 167))
                descriptors.append(None)
                valid_smiles.append(None)
                invalid_indices.append(i)
        
        return np.array(fingerprints), descriptors, valid_smiles, invalid_indices
    
    def augment_dataset(X, y, n_samples=1000, n_components=5):
        """Augment dataset using Gaussian Mixture Models"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        X.columns = X.columns.astype(str)
        df = X.copy()
        df['Target'] = y.values
        
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(df)
        
        synthetic_data, _ = gmm.sample(n_samples)
        synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
        
        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
        X_augmented = augmented_df.drop(columns='Target')
        y_augmented = augmented_df['Target']
        
        return X_augmented, y_augmented
    
    # Feature filters for each target
    required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}
    
    filters = {
        'Tg': list(set([
            'BalabanJ','BertzCT','Chi1','Chi3n','Chi4n','EState_VSA4','EState_VSA8',
            'FpDensityMorgan3','HallKierAlpha','Kappa3','MaxAbsEStateIndex','MolLogP',
            'NumAmideBonds','NumHeteroatoms','NumHeterocycles','NumRotatableBonds',
            'PEOE_VSA14','Phi','RingCount','SMR_VSA1','SPS','SlogP_VSA1','SlogP_VSA5',
            'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState4','VSA_EState6','VSA_EState7',
            'VSA_EState8','fr_C_O_noCOO','fr_NH1','fr_benzene','fr_bicyclic','fr_ether',
            'fr_unbrch_alkane'
        ]).union(required_descriptors)),
        'FFV': list(set([
            'AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
            'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','EState_VSA10','EState_VSA5',
            'EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1',
            'FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha',
            'HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex',
            'MaxEStateIndex','MinEStateIndex','MolLogP','MolMR','MolWt','NHOHCount',
            'NOCount','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
            'NumHeterocycles','NumRotatableBonds','PEOE_VSA14','RingCount','SMR_VSA1',
            'SMR_VSA10','SMR_VSA3','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SPS',
            'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
            'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',
            'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
            'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
            'VSA_EState8','VSA_EState9','fr_Ar_N','fr_C_O','fr_NH0','fr_NH1',
            'fr_aniline','fr_ether','fr_halogen','fr_thiophene'
        ]).union(required_descriptors)),
        'Tc': list(set([
            'BalabanJ','BertzCT','Chi0','EState_VSA5','ExactMolWt','FpDensityMorgan1',
            'FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt','MinEStateIndex',
            'MolWt','NumAtomStereoCenters','NumRotatableBonds','NumValenceElectrons',
            'SMR_VSA10','SMR_VSA7','SPS','SlogP_VSA6','SlogP_VSA8','VSA_EState1',
            'VSA_EState7','fr_NH1','fr_ester','fr_halogen'
        ]).union(required_descriptors)),
        'Density': list(set([
            'BalabanJ','Chi3n','Chi3v','Chi4n','EState_VSA1','ExactMolWt',
            'FractionCSP3','HallKierAlpha','Kappa2','MinEStateIndex','MolMR','MolWt',
            'NumAliphaticCarbocycles','NumHAcceptors','NumHeteroatoms',
            'NumRotatableBonds','SMR_VSA10','SMR_VSA5','SlogP_VSA12','SlogP_VSA5',
            'TPSA','VSA_EState10','VSA_EState7','VSA_EState8'
        ]).union(required_descriptors)),
        'Rg': list(set([
            'AvgIpc','Chi0n','Chi1v','Chi2n','Chi3v','ExactMolWt','FpDensityMorgan1',
            'FpDensityMorgan2','FpDensityMorgan3','HallKierAlpha','HeavyAtomMolWt',
            'Kappa3','MaxAbsEStateIndex','MolWt','NOCount','NumRotatableBonds',
            'NumUnspecifiedAtomStereoCenters','NumValenceElectrons','PEOE_VSA14',
            'PEOE_VSA6','SMR_VSA1','SMR_VSA5','SPS','SlogP_VSA1','SlogP_VSA2',
            'SlogP_VSA7','SlogP_VSA8','VSA_EState1','VSA_EState8','fr_alkyl_halide',
            'fr_halogen'
        ]).union(required_descriptors))
    }
    
    # Prepare test data
    test_smiles = test['SMILES'].tolist()
    test_ids = test['id'].values
    
    output_df = pd.DataFrame({'id': test_ids})
    
    # Train and predict for each target
    for label in TARGETS:
        print(f"\n📊 Processing {label}...")
        
        # Get training data for this target
        train_target = train_extended[train_extended[label].notna()][['SMILES', label]].reset_index(drop=True)
        
        if len(train_target) == 0:
            print(f"   ⚠️  No training data for {label}, skipping")
            output_df[label] = 0.0
            continue
        
        original_smiles = train_target['SMILES'].tolist()
        original_labels = train_target[label].values
        
        # Augment SMILES
        original_smiles, original_labels = augment_smiles_dataset(original_smiles, original_labels, num_augments=1)
        
        # Extract features
        fingerprints, descriptors, valid_smiles, invalid_indices = \
            smiles_to_combined_fingerprints_with_descriptors(original_smiles, radius=2, n_bits=128)
        
        X = pd.DataFrame(descriptors)
        X = X.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO',
                    'BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI',
                    'MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge',
                    'MaxAbsPartialCharge', 'SMILES'], axis=1, errors='ignore')
        y = np.delete(original_labels, invalid_indices)
        
        # Filter features
        X = X.filter(filters[label])
        
        # Add fingerprints
        fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
        fp_df.reset_index(drop=True, inplace=True)
        X.reset_index(drop=True, inplace=True)
        X = pd.concat([X, fp_df], axis=1)
        
        # Variance threshold
        selector = VarianceThreshold(threshold=0.01)
        X = selector.fit_transform(X)
        
        # Augment dataset
        X, y = augment_dataset(X, y, n_samples=1000)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        
        # Model parameters from original code
        if label == "Tg":
            Model = XGBRegressor(n_estimators=2173, learning_rate=0.0672418745539774, max_depth=6, reg_lambda=5.545520219149715)
        elif label == 'Rg':
            Model = XGBRegressor(n_estimators=520, learning_rate=0.07324113948440986, max_depth=5, reg_lambda=0.9717380315982088)
        elif label == 'FFV':
            Model = XGBRegressor(n_estimators=2202, learning_rate=0.07220580588586338, max_depth=4, reg_lambda=2.8872976032666493)
        elif label == 'Tc':
            Model = XGBRegressor(n_estimators=1488, learning_rate=0.010456188013762864, max_depth=5, reg_lambda=9.970345982204618)
        elif label == 'Density':
            Model = XGBRegressor(n_estimators=1958, learning_rate=0.10955287548172478, max_depth=5, reg_lambda=3.074470087965767)
        else:
            Model = XGBRegressor()
        
        RFModel = RandomForestRegressor(random_state=42)
        
        # Train models
        Model.fit(X_train, y_train)
        RFModel.fit(X_train, y_train)
        
        # Validation MAE
        y_pred = Model.predict(X_test)
        mae = mean_absolute_error(y_pred, y_test)
        print(f"   Validation MAE: {mae:.4f}")
        
        # Retrain on full data
        Model.fit(X, y)
        RFModel.fit(X, y)
        
        # Predict on test set
        fingerprints_test, descriptors_test, _, invalid_indices_test = \
            smiles_to_combined_fingerprints_with_descriptors(test_smiles, radius=2, n_bits=128)
        
        X_test = pd.DataFrame(descriptors_test)
        X_test = X_test.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO',
                              'BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI',
                              'MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge',
                              'MaxAbsPartialCharge', 'SMILES'], axis=1, errors='ignore')
        X_test = X_test.filter(filters[label])
        
        fp_df_test = pd.DataFrame(fingerprints_test, columns=[f'FP_{i}' for i in range(fingerprints_test.shape[1])])
        fp_df_test.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        X_test = pd.concat([X_test, fp_df_test], axis=1)
        X_test = selector.transform(X_test)
        
        # Generate predictions
        y_pred1 = Model.predict(X_test).flatten()
        y_pred2 = RFModel.predict(X_test).flatten()
        y_pred = y_pred1 * 0.6 + y_pred2 * 0.4
        
        output_df[label] = y_pred
        print(f"   ✅ {label} predictions complete")
    
    # Save results
    output_path = RESULTS_PATH / 'submission1.csv'
    output_df.to_csv(output_path, index=False)
    results_volume.commit()
    
    total_time = time.time() - start_time
    print(f"\n✅ XGBoost/RandomForest inference complete! Saved to {output_path}")
    print(f"⏱️  Total XGBoost time: {total_time:.2f}s ({total_time/60:.2f}m)")
    return str(output_path)


# ============================================================================
# XGBoost Parallel Inference (Per-Target)
# ============================================================================

@app.function(
    image=ml_image,
    cpu=16,
    memory=32768,
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
    timeout=3600,
)
def run_xgboost_single_target(target_name: str, test_mode: bool = False, sample_size: int = 100):
    """Run XGBoost/RandomForest inference for a SINGLE target (for parallelism)"""
    import time
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.mixture import GaussianMixture
    from xgboost import XGBRegressor
    from rdkit import Chem
    from rdkit.Chem import Descriptors, MACCSkeys
    from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
    from rdkit.Chem.Descriptors import MolWt, MolLogP
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.Chem import rdmolops
    import networkx as nx
    
    print(f"🔬 Starting XGBoost inference for {target_name}...")
    
    # Check if results already exist
    partial_path = RESULTS_PATH / f"submission1_{target_name}.csv"
    if partial_path.exists():
        print(f"✅ XGBoost results for {target_name} already exist at {partial_path}, skipping")
        # Load predictions to return them (needed for return signature)
        # But actually, we just need to return consistent type.
        # The main function expects (target_name, final_pred, duration)
        # Let's read the file to return predictions
        try:
            df = pd.read_csv(partial_path)
            return target_name, df[target_name].values, 0.0
        except:
            print(f"   ⚠️  Could not read existing file {partial_path}, re-running...")
    
    # Load data (needs full setup just like main function)
    base_path = DATA_PATH / 'neurips-open-polymer-prediction-2025'
    train = pd.read_csv(base_path / 'train.csv')
    test = pd.read_csv(base_path / 'test.csv')
    
    if test_mode:
        print(f"⚡ TEST MODE: Using only {sample_size} test samples")
        test = test.head(sample_size)
    
    # Helper functions (duplicated for independent execution)
    def clean_and_validate_smiles(smiles):
        if not isinstance(smiles, str) or len(smiles) == 0: return None
        bad = ['[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5', '([R])', '([R1])', '([R2])']
        if any(b in smiles for b in bad): return None
        if '][' in smiles and any(x in smiles for x in ['[R', 'R]']): return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except: return None

    # Clean SMILES
    train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
    test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)
    train = train[train['SMILES'].notnull()].reset_index(drop=True)
    test = test[test['SMILES'].notnull()].reset_index(drop=True)
    
    # Load external datasets (only relevant one)
    datasets = []
    if target_name == 'Tc':
        try:
            d = pd.read_csv(DATA_PATH / 'tc-smiles' / 'Tc_SMILES.csv').rename(columns={'TC_mean': 'Tc'})
            datasets.append(('Tc', d))
        except: pass
    elif target_name == 'Tg':
        try:
            d = pd.read_csv(DATA_PATH / 'external-polymer-data' / 'TgSS_enriched_cleaned.csv')
            if 'Tg' in d.columns: datasets.append(('Tg', d[['SMILES', 'Tg']]))
        except: pass
        try:
            d = pd.read_csv(DATA_PATH / 'smiles-extra-data' / 'JCIM_sup_bigsmiles.csv')
            d = d[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'})
            datasets.append(('Tg', d))
        except: pass
        try:
            d = pd.read_excel(DATA_PATH / 'smiles-extra-data' / 'data_tg3.xlsx').rename(columns={'Tg [K]': 'Tg'})
            d['Tg'] = d['Tg'] - 273.15
            datasets.append(('Tg', d[['SMILES', 'Tg']]))
        except: pass
    elif target_name == 'Density':
        try:
            d = pd.read_excel(DATA_PATH / 'smiles-extra-data' / 'data_dnst1.xlsx').rename(columns={'density(g/cm3)': 'Density'})
            d = d[['SMILES', 'Density']]
            d = d.query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
            d['Density'] = d['Density'].astype(float) - 0.118
            datasets.append(('Density', d))
        except: pass
        
    # Integrate external data
    train_extended = train[['SMILES', target_name]].copy() if target_name in train.columns else train[['SMILES']].copy()
    if target_name not in train_extended.columns: train_extended[target_name] = np.nan
        
    for t, d in datasets:
        if t != target_name: continue
        d['SMILES'] = d['SMILES'].apply(clean_and_validate_smiles)
        d = d.dropna(subset=['SMILES', target_name])
        unique_smiles = set(d['SMILES']) - set(train_extended['SMILES'])
        if len(unique_smiles) > 0:
            extra = d[d['SMILES'].isin(unique_smiles)].copy()
            train_extended = pd.concat([train_extended, extra[['SMILES', target_name]]], ignore_index=True)

    # Feature extraction logic
    def augment_smiles_dataset(smiles_list, labels, num_augments=1):
        aug_s, aug_l = [], []
        for s, l in zip(smiles_list, labels):
            mol = Chem.MolFromSmiles(s)
            if not mol: continue
            aug_s.append(s); aug_l.append(l)
            for _ in range(num_augments):
                try: aug_s.append(Chem.MolToSmiles(mol, doRandom=True)); aug_l.append(l)
                except: pass
        return aug_s, np.array(aug_l)

    def smiles_to_features(smiles_list):
        generator = GetMorganGenerator(radius=2, fpSize=128)
        fps, descs, valid, invalid = [], [], [], []
        
        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s)
            if mol:
                fps.append(np.concatenate([np.array(generator.GetFingerprint(mol)), np.array(MACCSkeys.GenMACCSKeys(mol))]))
                d = {}
                d['MolWt'] = MolWt(mol)
                d['LogP'] = MolLogP(mol)
                d['TPSA'] = CalcTPSA(mol)
                d['RotatableBonds'] = CalcNumRotatableBonds(mol)
                d['NumAtoms'] = mol.GetNumAtoms()
                try:
                    adj = rdmolops.GetAdjacencyMatrix(mol)
                    G = nx.from_numpy_array(adj)
                    d['graph_diameter'] = nx.diameter(G) if nx.is_connected(G) else 0
                    d['avg_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
                    d['num_cycles'] = len(list(nx.cycle_basis(G)))
                except:
                    d['graph_diameter'] = 0; d['avg_shortest_path'] = 0; d['num_cycles'] = 0
                
                # Add basic RDKit descriptors
                for name, func in Descriptors.descList:
                    if name in ['BalabanJ','BertzCT','Chi0','Chi1','Kappa1','Kappa2','Kappa3','HallKierAlpha']:
                        try: d[name] = func(mol)
                        except: d[name] = 0
                
                descs.append(d)
                valid.append(s)
            else:
                fps.append(np.zeros(128 + 167))
                descs.append({})
                invalid.append(i)
        return np.array(fps), descs, valid, invalid

    def augment_dataset(X, y, n_samples=1000):
        if len(X) < 50: return X, y # Skip if too small
        try:
            if isinstance(X, np.ndarray): X = pd.DataFrame(X)
            if isinstance(y, np.ndarray): y = pd.Series(y)
            X.columns = X.columns.astype(str)
            df = X.copy(); df['Target'] = y.values
            gmm = GaussianMixture(n_components=min(5, len(X)//10))
            gmm.fit(df)
            syn, _ = gmm.sample(n_samples)
            syn_df = pd.DataFrame(syn, columns=df.columns)
            aug = pd.concat([df, syn_df], ignore_index=True)
            return aug.drop(columns='Target'), aug['Target']
        except: 
            return X, y

    # Process Target
    print(f"📊 Processing {target_name}...")
    train_target = train_extended[train_extended[target_name].notna()][['SMILES', target_name]].reset_index(drop=True)
    
    if len(train_target) == 0:
        print(f"⚠️  No training data for {target_name}")
        return target_name, np.zeros(len(test)), 0.0

    orig_s = train_target['SMILES'].tolist()
    orig_l = train_target[target_name].values
    
    # Augment SMILES
    orig_s, orig_l = augment_smiles_dataset(orig_s, orig_l, num_augments=1)
    
    # Extract Features
    fps, descs, _, inv_idx = smiles_to_features(orig_s)
    X = pd.DataFrame(descs).fillna(0)
    y = np.delete(orig_l, inv_idx)
    
    # Add fingerprints
    fp_df = pd.DataFrame(fps, columns=[f'FP_{i}' for i in range(fps.shape[1])])
    X = pd.concat([X, fp_df], axis=1)
    
    # Filter features (simplified for speed/robustness)
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    # Augment
    X, y = augment_dataset(X, y, n_samples=1000)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost with Parallelism
    model = XGBRegressor(
        n_estimators=1000, 
        max_depth=6, 
        learning_rate=0.05, 
        n_jobs=16,          # Parallel threads
        tree_method='hist', # Fast training
        random_state=42
    )
    rf_model = RandomForestRegressor(n_estimators=100, n_jobs=16, random_state=42)
    
    model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    val_pred = model.predict(X_test)
    mae = mean_absolute_error(val_pred, y_test)
    print(f"   Validation MAE: {mae:.4f}")
    
    # Retrain full
    model.fit(X, y)
    rf_model.fit(X, y)
    
    # Predict
    test_smiles = test['SMILES'].tolist()
    fps_test, descs_test, _, _ = smiles_to_features(test_smiles)
    X_pred = pd.DataFrame(descs_test).fillna(0)
    fp_df_test = pd.DataFrame(fps_test, columns=[f'FP_{i}' for i in range(fps_test.shape[1])])
    X_pred = pd.concat([X_pred, fp_df_test], axis=1)
    
    # Align columns
    missing_cols = set(X.columns) - set(X_pred.columns)
    for c in missing_cols: X_pred[c] = 0
    X_pred = X_pred[X.columns] # Ensure order
    
    pred1 = model.predict(X_pred)
    pred2 = rf_model.predict(X_pred)
    final_pred = pred1 * 0.6 + pred2 * 0.4
    
    # Save partial results for merging
    # We need to save with IDs to ensure alignment
    partial_df = pd.DataFrame({'id': test['id'].values, target_name: final_pred})
    partial_path = RESULTS_PATH / f"submission1_{target_name}.csv"
    partial_df.to_csv(partial_path, index=False)
    results_volume.commit()
    print(f"      ✅ Saved partial results to {partial_path}")
    
    duration = time.time() - start_time
    print(f"✅ {target_name} complete in {duration:.2f}s")
    
    return target_name, final_pred, duration


# ============================================================================
# BERT Model Inference
# ============================================================================

@app.function(
    image=dl_image,
    gpu="a100",
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
    timeout=3600,  # 1 hour
)
def run_bert_inference(test_mode: bool = False, sample_size: int = 100):
    """Generate predictions using BERT models"""
    import torch
    import pandas as pd
    import joblib
    import numpy as np
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from torch import nn
    from transformers.activations import ACT2FN
    from tqdm import tqdm
    from rdkit import Chem
    
    # Check if results already exist
    output_path = RESULTS_PATH / 'submission2.csv'
    if output_path.exists():
        print(f"✅ BERT inference results already exist at {output_path}, skipping")
        return str(output_path)
    
    print("🤖 Starting BERT model inference...")
    
    # Define custom model classes
    class ContextPooler(nn.Module):
        def __init__(self, config):
            super().__init__()
            pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
            self.dense = nn.Linear(pooler_size, pooler_size)
            dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob)
            self.dropout = nn.Dropout(dropout_prob)
            self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act)
            self.config = config
        
        def forward(self, hidden_states):
            context_token = hidden_states[:, 0]
            context_token = self.dropout(context_token)
            pooled_output = self.dense(context_token)
            pooled_output = ACT2FN[self.activation](pooled_output)
            return pooled_output
    
    class CustomModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.backbone = AutoModel.from_config(config)
            self.pooler = ContextPooler(config)
            pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
            self.output = torch.nn.Linear(pooler_output_dim, 1)
        
        def forward(self, input_ids, attention_mask=None):
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            pooled_output = self.pooler(outputs.last_hidden_state)
            regression_output = self.output(pooled_output)
            return regression_output
    
    # Load test data
    test = pd.read_csv(DATA_PATH / 'neurips-open-polymer-prediction-2025' / 'test.csv')
    test_copy = test.copy()
    
    # Load scalers and tokenizer
    scalers = joblib.load(DATA_PATH / 'smiles-bert-models' / 'target_scalers.pkl')
    tokenizer = AutoTokenizer.from_pretrained(str(DATA_PATH / 'smiles-deberta77m-tokenizer'))
    
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # SMILES augmentation function
    def augment_smiles_simple(smiles, n_aug=5):
        """Generate augmented SMILES variations"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = [smiles]  # Include original
            for _ in range(n_aug):
                aug_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
                augmented.append(aug_smiles)
            return list(set(augmented))
        except:
            return [smiles]
    
    # Tokenization function
    def tokenize_smiles(smiles_list):
        smiles_with_cls = [tokenizer.cls_token + s for s in smiles_list]
        tokenized = tokenizer(
            smiles_with_cls,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return tokenized
    
    # Augment test data
    print("🔄 Augmenting test SMILES...")
    augmented_data = []
    for _, row in test.iterrows():
        aug_smiles = augment_smiles_simple(row['SMILES'], n_aug=10)
        for s in aug_smiles:
            augmented_data.append({'id': row['id'], 'SMILES': s})
    test_aug = pd.DataFrame(augmented_data)
    print(f"   Created {len(test_aug)} augmented samples from {len(test)} original")
    
    # Run inference for each target
    preds_mapping = {}
    
    for idx, target in enumerate(TARGETS):
        print(f"\n📊 Processing {target} with BERT...")
        
        scaler = scalers[idx]
        model_path = DATA_PATH / 'private-smile-bert-models' / f'warm_smiles_model_{target}_target.pth'
        
        if not model_path.exists():
            print(f"   ⚠️  Model not found at {model_path}, using zeros")
            preds_mapping[target] = [0.0] * len(test_copy)
            continue
        
        # Load model
        config = AutoConfig.from_pretrained(str(DATA_PATH / 'smiles-deberta77m-tokenizer'))
        model = CustomModel(config).cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Make predictions for each test sample
        true_preds = []
        for test_id in tqdm(test_copy['id'].unique(), desc=f"{target} predictions"):
            test_smiles_group = test_aug[test_aug['id'] == test_id]['SMILES'].tolist()
            
            # Tokenize and predict
            batch_preds = []
            batch_size = 32
            for i in range(0, len(test_smiles_group), batch_size):
                batch_smiles = test_smiles_group[i:i+batch_size]
                tokenized = tokenize_smiles(batch_smiles)
                
                input_ids = tokenized['input_ids'].cuda()
                attention_mask = tokenized['attention_mask'].cuda()
                
                with torch.no_grad():
                    preds = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds_cpu = preds.cpu().numpy()
                    preds_unscaled = scaler.inverse_transform(preds_cpu)
                    batch_preds.extend(preds_unscaled.flatten().tolist())
            
            # Average predictions across augmentations
            avg_pred = np.median(batch_preds)
            true_preds.append(float(avg_pred))
        
        preds_mapping[target] = true_preds
        print(f"   ✅ {target} predictions complete")
        
        # Clean up GPU memory
        del model, checkpoint
        torch.cuda.empty_cache()
    
    # Save results
    submission = pd.DataFrame(preds_mapping)
    submission['id'] = test_copy['id']
    submission.to_csv(output_path, index=False)
    results_volume.commit()
    
    print(f"\n✅ BERT inference complete! Saved to {output_path}")
    return str(output_path)

# ============================================================================
# TabTransformer Inference
# ============================================================================

@app.function(
    image=dl_image,
    gpu="a100",
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
    timeout=3600,
    cpu=4,
)
def run_tabtransformer_inference(test_mode: bool = False, sample_size: int = 100):
    """Generate predictions using TabTransformer models"""
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdmolops
    import networkx as nx
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    
    # Check if results already exist
    output_path = RESULTS_PATH / 'submission3.csv'
    if output_path.exists():
        print(f"✅ TabTransformer inference results already exist at {output_path}, skipping")
        return str(output_path)
    
    print("🔷 Starting TabTransformer inference...")
    
    # Load data
    base_path = DATA_PATH / 'neurips-open-polymer-prediction-2025'
    train = pd.read_csv(base_path / 'train.csv')
    test = pd.read_csv(base_path / 'test.csv')
    
    if test_mode:
        print(f"⚡ TEST MODE: Using only {sample_size} test samples")
        test = test.head(sample_size)
    
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Feature extraction (simplified version from original code)
    def extract_features(smiles):
        """Extract molecular features from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            features = {}
            
            # Basic descriptors
            for name, func in Descriptors.descList[:50]:  # Use first 50 descriptors
                try:
                    features[name] = func(mol)
                except:
                    features[name] = 0
            
            # Graph features
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)
                features['num_atoms'] = mol.GetNumAtoms()
                features['num_bonds'] = mol.GetNumBonds()
                if nx.is_connected(G):
                    features['diameter'] = nx.diameter(G)
                else:
                    features['diameter'] = 0
            except:
                features['num_atoms'] = 0
                features['num_bonds'] = 0
                features['diameter'] = 0
            
            return features
        except:
            return None
    
    print("🔬 Extracting features from training data...")
    train_features = []
    for smiles in train['SMILES']:
        feats = extract_features(smiles)
        if feats:
            train_features.append(feats)
        else:
            train_features.append({})
    
    train_feat_df = pd.DataFrame(train_features).fillna(0)
    
    print("🔬 Extracting features from test data...")
    test_features = []
    for smiles in test['SMILES']:
        feats = extract_features(smiles)
        if feats:
            test_features.append(feats)
        else:
            test_features.append({})
    
    test_feat_df = pd.DataFrame(test_features).fillna(0)
    
    # Align columns
    common_cols = list(set(train_feat_df.columns) & set(test_feat_df.columns))
    train_feat_df = train_feat_df[common_cols]
    test_feat_df = test_feat_df[common_cols]
    
    # Simple TabTransformer-like model (using basic MLP due to complexity)
    class SimpleTabModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.model(x)
    
    class TabDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.FloatTensor(X.values)
            self.y = torch.FloatTensor(y.values).reshape(-1, 1) if y is not None else None
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]
    
    # Train and predict for each target
    predictions = {target: [] for target in TARGETS}
    
    for target in TARGETS:
        print(f"\n📊 Processing {target} with TabTransformer...")
        
        # Get valid training data for this target
        valid_idx = train[target].notna()
        X_train = train_feat_df[valid_idx]
        y_train = train[target][valid_idx]
        
        if len(X_train) == 0:
            print(f"   ⚠️  No training data for {target}, using zeros")
            predictions[target] = [0.0] * len(test)
            continue
        
        # Clean data: replace inf/nan with appropriate values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median())
        
        test_feat_df_clean = test_feat_df.replace([np.inf, -np.inf], np.nan)
        test_feat_df_clean = test_feat_df_clean.fillna(X_train.median())
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(test_feat_df_clean),
            columns=test_feat_df_clean.columns
        )
        
        # 5-fold cross-validation predictions
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        test_preds_folds = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
            # Simple training loop
            X_fold = X_train_scaled.iloc[train_idx]
            y_fold = y_train.iloc[train_idx]
            
            model = SimpleTabModel(X_fold.shape[1]).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Create dataloaders
            dataset = TabDataset(X_fold, y_fold)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Train for a few epochs
            model.train()
            for epoch in range(20):
                for batch_X, batch_y in loader:
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Predict on test set
            model.eval()
            test_dataset = TabDataset(X_test_scaled)
            test_loader = DataLoader(test_dataset, batch_size=128)
            
            fold_preds = []
            with torch.no_grad():
                for batch_X in test_loader:
                    batch_X = batch_X.cuda()
                    preds = model(batch_X)
                    fold_preds.extend(preds.cpu().numpy().flatten().tolist())
            
            test_preds_folds.append(fold_preds)
            
            del model
            torch.cuda.empty_cache()
        
        # Average predictions across folds
        final_preds = np.mean(test_preds_folds, axis=0)
        predictions[target] = final_preds.tolist()
        print(f"   ✅ {target} predictions complete (5-fold CV)")
    
    # Save results
    submission = pd.DataFrame(predictions)
    submission['id'] = test['id']
    submission.to_csv(output_path, index=False)
    results_volume.commit()
    
    print(f"\n✅ TabTransformer inference complete! Saved to {output_path}")
    return str(output_path)

# ============================================================================
# Ensemble Predictions
# ============================================================================

@app.function(
    image=base_image,
    volumes={RESULTS_PATH: results_volume},
)
def create_ensemble():
    """Combine all three model predictions with weights"""
    import pandas as pd
    import numpy as np
    
    print("🎯 Creating ensemble predictions...")
    
    # Check if final submission already exists
    output_path = RESULTS_PATH / 'submission.csv'
    if output_path.exists():
        print(f"✅ Ensemble already exists at {output_path}, skipping")
        return str(output_path)
    
    # Load all three predictions
    sub1_path = RESULTS_PATH / 'submission1.csv'
    sub2_path = RESULTS_PATH / 'submission2.csv'
    sub3_path = RESULTS_PATH / 'submission3.csv'
    
    if not all([sub1_path.exists(), sub2_path.exists(), sub3_path.exists()]):
        print("⚠️  Not all submission files exist yet")
        return None
    
    sub1 = pd.read_csv(sub1_path)
    sub2 = pd.read_csv(sub2_path)
    sub3 = pd.read_csv(sub3_path)
    
    # Ensemble weights from original code
    weights = {
        'Tg': [0.55, 0.325, 0.125],
        'FFV': [0.65, 0.3, 0.05],
        'Tc': [0.55, 0.3, 0.15],
        'Density': [0.55, 0.325, 0.125],
        'Rg': [0.55, 0.325, 0.125]
    }
    
    ensemble = pd.DataFrame()
    ensemble['id'] = sub1['id']
    
    for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        w1, w2, w3 = weights[target]
        ensemble[target] = (
            sub1[target] * w1 +
            sub2[target] * w2 +
            sub3[target] * w3
        )
        
        # Apply Tg adjustment
        if target == 'Tg':
            ensemble[target] += 40.00
    
    # Save final ensemble
    ensemble.to_csv(output_path, index=False)
    results_volume.commit()
    
    print(f"✅ Ensemble complete! Saved to {output_path}")
    return str(output_path)

# ============================================================================
# Sustainability Model Training
# ============================================================================

@app.function(
    image=dl_image,
    gpu="a100",  # Needs GPU for De BERT a fine-tuning
    volumes={DATA_PATH: data_volume, MODELS_PATH: models_volume},
    timeout=7200,  # 2 hours
)
def train_sustainability_models():
    """Train sustainability prediction models using transfer learning"""
    import pandas as pd
    import numpy as np
    import torch
    import joblib
    import xgboost as xgb
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from rdkit import Chem
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    
    print("🏗️ Starting Sustainability Model Training Pipeline...")
    
    # 1. Load Raw Data
    train_df = pd.read_csv(DATA_PATH / 'neurips-open-polymer-prediction-2025' / 'train.csv')
    
    # 2. Generate Synthetic Targets (Y) and Features (X_tabular)
    print("🧪 Generating synthetic targets and features...")
    
    X_data = []
    Y_data = []
    valid_indices = []
    smiles_list = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing SMILES"):
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            feats = get_sustainability_features(mol)
            targets = generate_synthetic_targets(mol)
            if feats and targets:
                X_data.append(feats)
                Y_data.append(targets)
                valid_indices.append(idx)
                smiles_list.append(smiles)

    X_df = pd.DataFrame(X_data)
    Y_df = pd.DataFrame(Y_data)
    
    TARGET_COLS = Y_df.columns.tolist()
    print(f"   Generated data for {len(X_df)} polymers. Targets: {TARGET_COLS}")
    
    # 3. Train XGBoost Models (Tabular Branch)
    print("🌲 Training XGBoost Ensemble...")
    xgb_models = {}
    for target in TARGET_COLS:
        print(f"   Training XGBoost for {target}...")
        model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
        model.fit(X_df, Y_df[target])
        xgb_models[target] = model
        
        # Save Model
        MODELS_PATH.mkdir(exist_ok=True, parents=True)
        model_path = MODELS_PATH / f"xgb_{target}.json"
        model.save_model(str(model_path))
        print(f"      Saved to {model_path}")
    
    # 4. Train DeBERTa (Transfer Learning Branch)
    print("🧠 Fine-tuning DeBERTa (Transfer Learning)...")
    
    tokenizer_path = str(DATA_PATH / 'smiles-deberta77m-tokenizer')
    weights_path = DATA_PATH / 'private-smile-bert-models/warm_smiles_model_Tg_target.pth'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Custom Model Class (Multi-Head)
    class SustainabilityTransferModel(nn.Module):
        def __init__(self, config_path, num_targets):
            super().__init__()
            config = AutoConfig.from_pretrained(config_path)
            self.backbone = AutoModel.from_config(config)
            
            # Context Pooler (Same as original)
            pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
            self.pooler_dense = nn.Linear(pooler_size, pooler_size)
            self.pooler_activation = nn.Tanh()
            
            # NEW Head: Predicts all 4 sustainability targets at once
            self.output = nn.Linear(pooler_size, num_targets)

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            # Pool
            first_token_tensor = outputs.last_hidden_state[:, 0]
            pooled_output = self.pooler_dense(first_token_tensor)
            pooled_output = self.pooler_activation(pooled_output)
            # Predict
            return self.output(pooled_output)

    # Prepare Data for PyTorch
    tokens = tokenizer(smiles_list, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    dataset = TensorDataset(
        tokens['input_ids'], 
        tokens['attention_mask'], 
        torch.tensor(Y_df.values, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model & Load Weights
    model = SustainabilityTransferModel(tokenizer_path, len(TARGET_COLS))
    
    if weights_path.exists():
        print(f"   Loading pre-trained weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Filter out the old 'output' layer, load only backbone and pooler
        backbone_state = {k: v for k, v in checkpoint.items() if 'output' not in k}
        
        # Load with strict=False to allow missing keys (new output layer)
        missing_keys, unexpected_keys = model.load_state_dict(backbone_state, strict=False)
        print(f"      Loaded backbone. Missing keys (expected): {len(missing_keys)}")
        print(f"      Unexpected keys: {len(unexpected_keys)}")
        
        # Freeze backbone weights (transfer learning)
        print("   Freezing backbone weights...")
        for name, param in model.named_parameters():
            if 'backbone' in name or 'pooler' in name:
                param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    else:
        print("⚠️ Pre-trained weights not found! Training from scratch.")
    
    model = model.cuda()
    
    # Training Loop
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(3):  # Fast fine-tune
        total_loss = 0
        for b_ids, b_mask, b_y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            b_ids, b_mask, b_y = b_ids.cuda(), b_mask.cuda(), b_y.cuda()
            
            optimizer.zero_grad()
            preds = model(b_ids, b_mask)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
        
    # Save DeBERTa
    model_save_path = MODELS_PATH / "sustainability_deberta.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"   Saved DeBERTa model to {model_save_path}")
    
    models_volume.commit()
    print("✅ Training Complete. Models saved to /models")


# ============================================================================
# Sustainability Training - PARALLEL (Per-Target)
# ============================================================================

@app.function(
    image=dl_image,
    gpu="a100",
    volumes={DATA_PATH: data_volume, MODELS_PATH: models_volume},
    cpu=16,
    memory=32768,
    timeout=7200,  # 2 hours
)
def train_sustainability_single_target(target_name: str, X_data: list, Y_data: dict, smiles_list: list):
    """
    Train XGBoost + DeBERTa for a SINGLE sustainability target (for parallelization)
    
    Args:
        target_name: 'Target_Recyclability', 'Target_BioSource', etc.
        X_data: List of feature dicts (shared across all targets)
        Y_data: Dict of {target_name -> values}
        smiles_list: List of SMILES strings
    """
    import time
    import pandas as pd
    import xgboost as xgb
    import torch
    from torch import nn
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    
    start_time = time.time()
    print(f"🎯 Training {target_name} (parallel job)")
    
    # Prepare data
    X_df = pd.DataFrame(X_data)
    y_values = Y_data[target_name]
    
    # 1. Train XGBoost for this target
    print(f"   🌲 Training XGBoost for {target_name}...")
    xgb_start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=16,
        tree_method='hist',
        random_state=42  # For reproducibility
    )
    xgb_model.fit(X_df, y_values)
    
    # Save XGBoost model
    MODELS_PATH.mkdir(exist_ok=True, parents=True)
    xgb_path = MODELS_PATH / f"xgb_{target_name}.json"
    xgb_model.save_model(str(xgb_path))
    xgb_time = time.time() - xgb_start
    print(f"      ✅ XGBoost trained in {xgb_time:.2f}s, saved to {xgb_path}")
    
    # 2. Train DeBERTa head for this target
    print(f"   🧠 Fine-tuning DeBERTa head for {target_name}...")
    deberta_start = time.time()
    
    tokenizer_path = str(DATA_PATH / 'smiles-deberta77m-tokenizer')
    weights_path = DATA_PATH / 'private-smile-bert-models/warm_smiles_model_Tg_target.pth'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Single-target model
    class SingleTargetTransferModel(nn.Module):
        def __init__(self, config_path):
            super().__init__()
            config = AutoConfig.from_pretrained(config_path)
            self.backbone = AutoModel.from_config(config)
            
            pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
            self.pooler_dense = nn.Linear(pooler_size, pooler_size)
            self.pooler_activation = nn.Tanh()
            self.output = nn.Linear(pooler_size, 1)  # Single target
            
        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            first_token = outputs.last_hidden_state[:, 0]
            pooled = self.pooler_activation(self.pooler_dense(first_token))
            return self.output(pooled)
    
    # Prepare PyTorch dataset
    tokens = tokenizer(smiles_list, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    dataset = TensorDataset(
        tokens['input_ids'],
        tokens['attention_mask'],
        torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Load model with transfer learning
    model = SingleTargetTransferModel(tokenizer_path)
    
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location='cpu')
        backbone_state = {k: v for k, v in checkpoint.items() if 'output' not in k}
        model.load_state_dict(backbone_state, strict=False)
        
        # Freeze backbone
        for name, param in model.named_parameters():
            if 'backbone' in name or 'pooler' in name:
                param.requires_grad = False
    
    model = model.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    criterion = nn.MSELoss()
    
    # Fast fine-tune
    model.train()
    for epoch in range(3):
        total_loss = 0
        for b_ids, b_mask, b_y in dataloader:
            b_ids, b_mask, b_y = b_ids.cuda(), b_mask.cuda(), b_y.cuda()
            
            optimizer.zero_grad()
            preds = model(b_ids, b_mask)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"      Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    # Save DeBERTa
    deberta_path = MODELS_PATH / f"sustainability_deberta_{target_name}.pth"
    torch.save(model.state_dict(), deberta_path)
    deberta_time = time.time() - deberta_start
    print(f"      ✅ DeBERTa trained in {deberta_time:.2f}s, saved to {deberta_path}")
    
    total_time = time.time() - start_time
    print(f"✅ {target_name} complete in {total_time:.2f}s ({total_time/60:.2f}m)")
    
    return {
        'target': target_name,
        'xgb_time': xgb_time,
        'deberta_time': deberta_time,
        'total_time': total_time
    }


@app.function(
    image=ml_image,
    cpu=16,
    memory=32768,
    volumes={DATA_PATH: data_volume, MODELS_PATH: models_volume},
)
def train_sustainability_models_parallel():
    """Train sustainability models with per-target parallelism (4x speedup)"""
    import time
    import pandas as pd
    from pathlib import Path
    
    overall_start = time.time()
    print("🏗️ Starting PARALLEL Sustainability Model Training...")
    
    # 1. Generate features (shared across all targets - done once on CPU)
    print("🧪 Generating features (shared step)...")
    feature_start = time.time()
    
    from get_sustainability_features import get_sustainability_features
    
    train_df = pd.read_csv(DATA_PATH / 'neurips-open-polymer-prediction-2025' / 'train.csv')
    smiles_list = train_df['SMILES'].tolist()
    
    # Generate targets and features
    try:
        X_data, Y_data = get_sustainability_features(train_df)
    except Exception as e:
        print(f"⚠️ Using fallback feature generation: {e}")
        # Fallback implementation (simplified)
        X_data = []
        Y_data = {
            'Target_Recyclability': [],
            'Target_BioSource': [],
            'Target_EnvSafety': [],
            'Target_SynthEfficiency': []
        }
        
        for smiles in smiles_list:
            # Simple features
            features = {
                'mol_weight': len(smiles),  # Placeholder
                'num_atoms': len(smiles.split()),
            }
            X_data.append(features)
            
            # Simple targets
            for target in Y_data.keys():
                Y_data[target].append(0.5)  # Placeholder
    
    X_df = pd.DataFrame(X_data)
    TARGET_COLS = list(Y_data.keys())
    
    feature_time = time.time() - feature_start
    print(f"   ✅ Features generated in {feature_time:.2f}s for {len(X_df)} samples")
    
    # 2. Train all targets in PARALLEL (4 jobs simultaneously)
    print(f"\n🚀 Training {len(TARGET_COLS)} targets in PARALLEL...")
    parallel_start = time.time()
    
    # Launch parallel jobs
    results = list(modal.gather(*[
        train_sustainability_single_target.remote(target, X_data, Y_data, smiles_list)
        for target in TARGET_COLS
    ]))
    
    parallel_time = time.time() - parallel_start
    
    # Commit all models
    models_volume.commit()
    
    # Print summary
    overall_time = time.time() - overall_start
    print("\n" + "="*60)
    print("✅ PARALLEL TRAINING COMPLETE!")
    print(f"\n⏱️  Timing Summary:")
    print(f"   Feature generation: {feature_time:.2f}s ({feature_time/60:.2f}m)")
    print(f"   Parallel training:  {parallel_time:.2f}s ({parallel_time/60:.2f}m)")
    print(f"   Total time:         {overall_time:.2f}s ({overall_time/60:.2f}m)")
    
    print(f"\n📊 Per-Target Breakdown:")
    for result in results:
        print(f"   {result['target']:30s}: {result['total_time']:.2f}s")
    
    # Calculate speedup estimate
    total_sequential = sum(r['total_time'] for r in results)
    speedup = total_sequential / parallel_time if parallel_time > 0 else 1
    print(f"\n🚀 Estimated speedup: {speedup:.2f}x")
    print(f"   (Sequential would take ~{total_sequential:.2f}s = {total_sequential/60:.2f}m)")


# ============================================================================
# Sustainability Inference
# ============================================================================

@app.function(
    image=dl_image,
    gpu="a100",
    volumes={DATA_PATH: data_volume, MODELS_PATH: models_volume, RESULTS_PATH: results_volume},
)
def run_sustainability_inference(smiles_list: list = None):
    """
    Run inference for sustainability metrics.
    If smiles_list is None, runs on the competition test set.
    """
    import pandas as pd
    import xgboost as xgb
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from rdkit import Chem
    from torch import nn
    
    print("🌿 Running Sustainability Inference...")
    
    # 1. Prepare Input Data
    if smiles_list is None:
        test_df = pd.read_csv(DATA_PATH / 'neurips-open-polymer-prediction-2025' / 'test.csv')
        smiles_list = test_df['SMILES'].tolist()
        ids = test_df['id'].tolist()
    else:
        ids = [f"custom_{i}" for i in range(len(smiles_list))]

    # 2. XGBoost Inference
    print("   🌲 Running XGBoost...")
    X_data = []
    valid_mask = []
    
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            feats = get_sustainability_features(mol)
            if feats:
                X_data.append(feats)
                valid_mask.append(True)
            else:
                X_data.append({})
                valid_mask.append(False)
        else:
            X_data.append({})
            valid_mask.append(False)
            
    X_df = pd.DataFrame(X_data)
    
    xgb_preds = {}
    targets = ["Target_Recyclability", "Target_BioSource", "Target_EnvSafety", "Target_SynthEfficiency"]
    
    for t in targets:
        model_path = MODELS_PATH / f"xgb_{t}.json"
        try:
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            
            # Predict
            p = np.zeros(len(smiles_list))
            if any(valid_mask) and len(X_df) > 0:
                # Get valid rows
                valid_X = X_df[valid_mask]
                if len(valid_X) > 0:
                    valid_preds = model.predict(valid_X)
                    # Assign back to full array
                    valid_idx = 0
                    for i, is_valid in enumerate(valid_mask):
                        if is_valid:
                            p[i] = valid_preds[valid_idx]
                            valid_idx += 1
            
            xgb_preds[t] = p
        except Exception as e:
            print(f"   ⚠️ Failed to load/run XGBoost for {t}: {e}")
            xgb_preds[t] = np.zeros(len(smiles_list))

    # 3. DeBERTa Inference
    print("   🧠 Running DeBERTa...")
    
    # Re-define class (Must match training)
    class SustainabilityTransferModel(nn.Module):
        def __init__(self, config_path, num_targets):
            super().__init__()
            config = AutoConfig.from_pretrained(config_path)
            self.backbone = AutoModel.from_config(config)
            pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
            self.pooler_dense = nn.Linear(pooler_size, pooler_size)
            self.pooler_activation = nn.Tanh()
            self.output = nn.Linear(pooler_size, num_targets)

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            first_token_tensor = outputs.last_hidden_state[:, 0]
            pooled_output = self.pooler_dense(first_token_tensor)
            pooled_output = self.pooler_activation(pooled_output)
            return self.output(pooled_output)

    tokenizer_path = str(DATA_PATH / 'smiles-deberta77m-tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    model_path = MODELS_PATH / "sustainability_deberta.pth"
    
    try:
        model = SustainabilityTransferModel(tokenizer_path, len(targets)).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        tokens = tokenizer(smiles_list, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        bert_preds = []
        
        # Batch inference
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                b_ids = tokens['input_ids'][i:i+batch_size].cuda()
                b_mask = tokens['attention_mask'][i:i+batch_size].cuda()
                p = model(b_ids, b_mask).cpu().numpy()
                bert_preds.append(p)
                
        bert_preds = np.concatenate(bert_preds, axis=0)  # Shape (N, 4)
    except Exception as e:
        print(f"   ⚠️ Failed to run DeBERTa: {e}")
        bert_preds = np.zeros((len(smiles_list), 4))

    # 4. Ensemble & Save
    final_df = pd.DataFrame({'id': ids, 'SMILES': smiles_list})
    
    for i, t in enumerate(targets):
        # Ensemble: 50% XGBoost, 50% DeBERTa
        xgb_p = xgb_preds.get(t, np.zeros(len(smiles_list)))
        bert_p = bert_preds[:, i]
        
        final_df[t] = (xgb_p * 0.5) + (bert_p * 0.5)

    output_path = RESULTS_PATH / 'sustainability_predictions.csv'
    final_df.to_csv(output_path, index=False)
    results_volume.commit()
    print(f"✅ Saved to {output_path}")
    return str(output_path)

# ============================================================================
# Local Entrypoint
# ============================================================================


# ============================================================================
# PARALLELIZATION: Helper Functions
# ============================================================================

@app.function(
    image=base_image,
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
)
def merge_target_predictions(file_prefix: str, targets: list, test_mode: bool = False, sample_size: int = 100):
    """Merge per-target prediction files into single submission file"""
    import pandas as pd
    
    print(f"📊 Merging {len(targets)} target predictions for {file_prefix}...")
    
    # Read test IDs from volume
    test_df = pd.read_csv(DATA_PATH / 'neurips-open-polymer-prediction-2025' / 'test.csv')
    if test_mode:
        test_df = test_df.head(sample_size)
    test_ids = test_df['id'].values
    
    merged_df = pd.DataFrame({'id': test_ids})
    
    for target in targets:
        partial_path = RESULTS_PATH / f"{file_prefix}_{target}.csv"
        if partial_path.exists():
            target_df = pd.read_csv(partial_path)
            merged_df[target] = target_df[target]
            print(f"   ✅ Merged {target}")
        else:
            print(f"   ⚠️  {target} not found, using zeros")
            merged_df[target] = 0.0
    
    output_path = RESULTS_PATH / f"{file_prefix}.csv"
    merged_df.to_csv(output_path, index=False)
    results_volume.commit()
    
    print(f"✅ Merged predictions saved to {output_path}")
    return str(output_path)


# ============================================================================
# PARALLELIZATION: BERT Micro-Level (Data Batching) - OPTIMIZED WITH modal.Cls
# ============================================================================

@app.cls(
    image=dl_image,
    gpu="a100",
    volumes={DATA_PATH: data_volume},
    timeout=3600,
    max_containers=10,  # Updated from concurrency_limit
)
class BertBatchPredictor:
    """Class-based BERT predictor - loads models ONCE per container (critical fix)"""
    
    @modal.enter()
    def load_models(self):
        """Load models once when container starts"""
        import torch
        import joblib
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from torch import nn
        from transformers.activations import ACT2FN
        
        print("🔧 [ENTER] Loading BERT models (one-time per container)...")
        
        class ContextPooler(nn.Module):
            def __init__(self, config):
                super().__init__()
                pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
                self.dense = nn.Linear(pooler_size, pooler_size)
                dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob)
                self.dropout = nn.Dropout(dropout_prob)
                self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act)
            
            def forward(self, hidden_states):
                context_token = hidden_states[:, 0]
                context_token = self.dropout(context_token)
                pooled_output = self.dense(context_token)
                return ACT2FN[self.activation](pooled_output)
        
        class CustomModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.backbone = AutoModel.from_config(config)
                self.pooler = ContextPooler(config)
                pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
                self.output = torch.nn.Linear(pooler_output_dim, 1)
            
            def forward(self, input_ids, attention_mask=None):
                outputs = self.backbone(input_ids, attention_mask=attention_mask)
                pooled_output = self.pooler(outputs.last_hidden_state)
                return self.output(pooled_output)
        
        # Load shared resources ONCE
        self.scalers = joblib.load(DATA_PATH / 'smiles-bert-models' / 'target_scalers.pkl')
        self.tokenizer = AutoTokenizer.from_pretrained(str(DATA_PATH / 'smiles-deberta77m-tokenizer'))
        self.config = AutoConfig.from_pretrained(str(DATA_PATH / 'smiles-deberta77m-tokenizer'))
        
        # Load all 5 target models into memory
        self.models = {}
        targets = ['Tg', 'FFV', 'FFV', 'Tc', 'Density', 'Rg']
        
        for target in targets:
            model_path = DATA_PATH / 'private-smile-bert-models' / f'warm_smiles_model_{target}_target.pth'
            model = CustomModel(self.config).cuda()
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint)
            model.eval()
            self.models[target] = model
            print(f"   ✅  Loaded {target} model")
        
        print("✅ [ENTER] All models loaded!")
    
    @modal.method()
    def predict_batch(self, target: str, scaler_idx: int, smiles_batch: list, ids_batch: list):
        """Process batch using pre-loaded models"""
        import torch
        import numpy as np
        from rdkit import Chem
        
        print(f"🤖 BERT Batch for {target}: {len(smiles_batch)} samples (cached ✅)")
        
        scaler = self.scalers[scaler_idx]
        model = self.models[target]
        
        def augment_smiles_simple(smiles, n_aug=5):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return [smiles]
                augmented = [smiles]
                for _ in range(n_aug):
                    aug = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
                    augmented.append(aug)
                return list(set(augmented))
            except:
                return [smiles]
        
        batch_results = []
        for smiles, sample_id in zip(smiles_batch, ids_batch):
            aug_smiles = augment_smiles_simple(smiles, n_aug=10)
            smiles_with_cls = [self.tokenizer.cls_token + s for s in aug_smiles]
            
            tokenized = self.tokenizer(smiles_with_cls, padding='max_length', 
                                     truncation=True, max_length=512, return_tensors='pt')
            
            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()
            
            with torch.no_grad():
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                preds_cpu = preds.cpu().numpy()
                preds_unscaled = scaler.inverse_transform(preds_cpu)
                avg_pred = np.median(preds_unscaled.flatten())
            
            batch_results.append({'id': sample_id, target: float(avg_pred)})
        
        return batch_results



@app.function(
    image=base_image,
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
)
def run_bert_inference_parallel(
    test_mode: bool = False,
    sample_size: int = 100,
    num_batches: int = 10
):
    """BERT inference with data parallelism across multiple GPUs"""
    import pandas as pd
    
    print(f"🤖 PARALLEL BERT: {num_batches} batches × 5 targets = {num_batches * 5} jobs")
    
    # Load test data
    test_df = pd.read_csv(DATA_PATH / 'neurips-open-polymer-prediction-2025' / 'test.csv')
    if test_mode:
        test_df = test_df.head(sample_size)
    
    test_ids = test_df['id'].tolist()
    test_smiles = test_df['SMILES'].tolist()
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Split into batches
    batch_size = max(1, len(test_smiles) // num_batches)
    batches = []
    for i in range(0, len(test_smiles), batch_size):
        end_idx = min(i + batch_size, len(test_smiles))
        batches.append((test_smiles[i:end_idx], test_ids[i:end_idx]))
    
    print(f"   {len(test_smiles)} samples → {len(batches)} batches")
    
    # Process all targets
    all_target_dfs = []
    
    for target_idx, target in enumerate(targets):
        print(f"\n   Processing {target} across {len(batches)} batches...")
        
        # Map batches
        predictor = BertBatchPredictor()
        batch_results = list(predictor.predict_batch.starmap([
            (target, target_idx, smiles, ids)
            for smiles, ids in batches
        ]))
        
        # Aggregate
        all_preds = []
        for batch_result in batch_results:
            all_preds.extend(batch_result)
        
        # Sort by original ID order
        target_df = pd.DataFrame(all_preds).set_index('id').reindex(test_ids).reset_index()
        all_target_dfs.append(target_df)
        print(f"      ✅ {target} ({len(all_preds)} predictions)")
    
    # Merge all targets
    merged_df = all_target_dfs[0][['id']]
    for target_df in all_target_dfs:
        target_col = [c for c in target_df.columns if c != 'id'][0]
        merged_df[target_col] = target_df[target_col]
    
    output_path = RESULTS_PATH / 'submission2.csv'
    merged_df.to_csv(output_path, index=False)
    results_volume.commit()
    
    print(f"\n✅ BERT parallel inference complete: {output_path}")
    return str(output_path)


# ============================================================================
# Local Entrypoint (WITH PARALLELIZATION)
# ============================================================================

@app.local_entrypoint()
def main(
    skip_download: bool = False,
    skip_xgboost: bool = False,
    skip_bert: bool = False,
    skip_tabtransformer: bool = False,
    train_sustainability: bool = False,
    predict_sustainability: bool = False,
    parallel_sustainability: bool = False,  # NEW: Per-target parallel training
    # PARALLELIZATION FLAGS
    parallel_pipeline: bool = False,  # Macro: Run models simultaneously
    parallel_xgboost: bool = False,   # NEW: Run XGBoost targets in parallel (5x speedup)
    parallel_data: bool = False,     # Micro: Split BERT data across GPUs
    num_data_batches: int = 10,      # Number of batches for data parallelism
    test_mode: bool = False,
    sample_size: int = 100
):
    """Run complete inference pipeline with optional parallelization
    
    Parallelization Options:
    - --parallel-pipeline: Run XGBoost, BERT, TabTransformer simultaneously (2.5x speedup)
    - --parallel-xgboost: Run 5 XGBoost targets deeply in parallel (5x speedup for Step 2)
    - --parallel-data: Split BERT across multiple GPUs (10x speedup for BERT)
    - --parallel-sustainability: Train sustainability models per-target in parallel (3-4x speedup)
    - Combined: Up to 25x speedup for full pipeline
    
    Sustainability Training:
    - --train-sustainability: Train models sequentially (~25-30 min)
    - --train-sustainability --parallel-sustainability: Train in parallel (~7-10 min, 3-4x faster!)
    """
    import time
    print("🚀 Polymer Prediction Inference Pipeline on Modal")
    print("=" * 60)
    
    if test_mode:
        print(f"⚡ TEST MODE: Processing only {sample_size} samples")
    if parallel_pipeline:
        print("⚡ PIPELINE PARALLELISM: Enabled (models run simultaneously)")
    if parallel_xgboost:
        print("⚡ XGBOOST PARALLELISM: Enabled (5 targets run simultaneously)")
    if parallel_data:
        print(f"⚡ DATA PARALLELISM: Enabled ({num_data_batches} batches for BERT)")
    
    print("=" * 60)
    
    stage_times = {}
    
    # Step 1: Download datasets
    if not skip_download:
        print("\n📦 Step 1: Downloading datasets...")
        stage_start = time.time()
        download_datasets.remote()
        stage_times["download"] = time.time() - stage_start
        print(f"⏱️  Step 1 time: {stage_times['download']:.2f}s")
    else:
        print("\n📦 Step 1: Skipping dataset download")
    
    # Initialize results containers
    xgb_result = None
    bert_result = None
    tab_result = None
    
    # ----------------------------------------------------------------
    # MACRO-LEVEL PARALLELISM (Pipeline)
    # ----------------------------------------------------------------
    if parallel_pipeline:
        print("\n⚡ Starting PIPELINE PARALLELISM (XGBoost + BERT + TabTransformer)...")
        stage_start = time.time()
        
        # Prepare futures list
        futures = {}
        
        # 1. XGBoost Future
        if not skip_xgboost:
            if parallel_xgboost:
                print("   ⚡ Launching 5 parallel XGBoost jobs (one per target)...")
                # Launch 5 parallel jobs
                targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
                xgb_futures = [run_xgboost_single_target.remote(t, test_mode, sample_size) for t in targets]
                futures['xgboost_parallel'] = xgb_futures
            else:
                print("   🔬 Launching XGBoost/RandomForest...")
                futures['xgboost'] = run_xgboost_inference.remote(test_mode, sample_size)
        
        # 2. BERT Future
        if not skip_bert:
            if parallel_data:
                print(f"   🤖 Launching BERT Data Parallelism ({num_data_batches} batches)...")
                # FIX: Use keyword arguments to avoid positional mismatch
                futures['bert'] = run_bert_inference_parallel.remote(
                    test_mode=test_mode, 
                    sample_size=sample_size, 
                    num_batches=num_data_batches
                )
            else:
                print("   🤖 Launching BERT/DeBERTa...")
                futures['bert'] = run_bert_inference.remote(test_mode, sample_size)
        
        # 3. TabTransformer Future
        if not skip_tabtransformer:
            print("   🔷 Launching TabTransformer...")
            futures['tabtransformer'] = run_tabtransformer_inference.remote(test_mode, sample_size)
        
        # Gather Results
        print("   ⏳ Waiting for all models to complete...")
        
        # Handle XGBoost results
        if 'xgboost_parallel' in futures:
            # Gather parallel results to ensure completion
            xg_list = futures['xgboost_parallel']
            print(f"   DEBUG: xgb list type: {type(xg_list)}, len: {len(xg_list)}")
            
            xgb_results_list = []
            if len(xg_list) > 0:
                first_item = xg_list[0]
                print(f"   DEBUG: first item type: {type(first_item)}")
                
                # Robustly handle results
                for item in xg_list:
                    if isinstance(item, tuple):
                        # Result returned directly (e.g. from local execution or immediate return)
                        xgb_results_list.append(item)
                    else:
                        # Assume it's a Modal FunctionCall/Future
                        xgb_results_list.append(item.get())
            
            # Merge 5 CSVs into one submission1.csv
            print("   📊 Merging parallel XGBoost results...")
            # FIX: No local pandas read, let remote handle it
            xgb_result = merge_target_predictions.remote("submission1", ['Tg', 'FFV', 'Tc', 'Density', 'Rg'], test_mode, sample_size)
            
        elif 'xgboost' in futures:
            xgb_result = futures['xgboost'].get()
            
        stage_times['xgboost'] = 0 # Tracked in total parallel time usually
            
        if 'bert' in futures:
            print("   DEBUG: Checking BERT result persistence...")
            f = futures['bert']
            if isinstance(f, (str, tuple)):
                bert_result = f
            else:
                bert_result = f.get()
                
        if 'tabtransformer' in futures:
            f = futures['tabtransformer']
            if isinstance(f, (str, tuple)):
                tab_result = f
            else:
                tab_result = f.get()
            
        pipeline_time = time.time() - stage_start
        print(f"⏱️  Total Parallel Pipeline Time: {pipeline_time:.2f}s")
        
    else:
        # ----------------------------------------------------------------
        # SEQUENTIAL EXECUTION
        # ----------------------------------------------------------------
        
        # Step 2: XGBoost
        if not skip_xgboost:
            print("\n🔬 Step 2: Running XGBoost/RandomForest inference...")
            stage_start = time.time()
            
            if parallel_xgboost:
                print("   ⚡ Using PARALLEL inference (5 targets on 5 containers)")
                targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
                
                # Run parallel
                print("   ⚡ Launching 5 parallel XGBoost jobs (one per target)...")
                xgb_futures = [
                    run_xgboost_single_target.remote(t, test_mode, sample_size) 
                    for t in targets
                ]
                
                # Wait for results
                # Each result is (target, predictions, duration)
                xgb_results_list = []
                for item in xgb_futures:
                     if isinstance(item, tuple):
                         xgb_results_list.append(item)
                     else:
                         xgb_results_list.append(item.get())
                
                # Merge results using helper function
                print("   📊 Merging parallel XGBoost results...")
                xgb_result = merge_target_predictions.remote("submission1", targets, test_mode, sample_size)
            else:
                xgb_result = run_xgboost_inference.remote(test_mode, sample_size)
            
            stage_times['xgboost'] = time.time() - stage_start
            print(f"⏱️  Step 2 time: {stage_times['xgboost']:.2f}s")
        else:
            print("\n🔬 Step 2: Skipping XGBoost inference")
        
        if not skip_bert:
            print("\n🤖 Step 3: Running BERT model inference...")
            if parallel_data:
                # Use data-parallel version even in sequential mode
                bert_result = run_bert_inference_parallel.remote(
                    test_mode=test_mode,
                    sample_size=sample_size,
                    num_batches=num_data_batches
                )
            else:
                bert_result = run_bert_inference.remote(test_mode=test_mode, sample_size=sample_size)
            print(f"   Result: {bert_result}")
        else:
            print("\n🤖 Step 3: Skipping BERT inference")
        
        if not skip_tabtransformer:
            print("\n🔷 Step 4: Running TabTransformer inference...")
            tab_result = run_tabtransformer_inference.remote(test_mode=test_mode, sample_size=sample_size)
            print(f"   Result: {tab_result}")
        else:
            print("\n🔷 Step 4: Skipping TabTransformer inference")
    
    # Step 5: Create ensemble
    print("\n🎯 Step 5: Creating ensemble predictions...")
    ensemble_result = create_ensemble.remote()  # Returns immediately, ensemble runs async
    print(f"   Result: {ensemble_result}")
    
    # Step 6: Train Sustainability Models
    if train_sustainability:
        print("\n🏗️ Step 6: Training Sustainability Models...")
        stage_start = time.time()
        
        if parallel_sustainability:
            print("   ⚡ Using PARALLEL per-target training (4 GPUs simultaneously)")
            train_sustainability_models_parallel.remote().get()  # Block until complete
        else:
            print("   📝 Using sequential training (tip: add --parallel-sustainability for 3-4x speedup)")
            train_sustainability_models.remote().get()  # Block until complete (prevents race condition)
        
        stage_times['sustainability_train'] = time.time() - stage_start
        print(f"⏱️  Step 6 time: {stage_times['sustainability_train']:.2f}s")
    
    # Step 7: Predict Sustainability Metrics
    if predict_sustainability:
        print("\n🌿 Step 7: Predicting Sustainability Metrics...")
        sust_result = run_sustainability_inference.remote()
        print(f"   Result: {sust_result}")
    
    # Timing Summary
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("✅ Inference pipeline complete!")
    print("\n⏱️  TIMING SUMMARY:")
    print(f"   Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}m)")
    if stage_times:
        print("\n   Stage breakdown:")
        for stage, duration in stage_times.items():
            pct = (duration / total_time * 100) if total_time > 0 else  0
            print(f"   - {stage:20s}: {duration:7.2f}s ({duration/60:5.2f}m) - {pct:5.1f}%")
    print("📊 Results saved to Modal Volume 'polymer-results'")
    print("\n📥 To download results locally:")
    print("  modal volume get polymer-results submission.csv submission.csv")
    print("  modal volume get polymer-results submission1.csv submission1.csv")
    print("  modal volume get polymer-results submission2.csv submission2.csv")
    print("  modal volume get polymer-results submission3.csv submission3.csv")
    if predict_sustainability:
        print("  modal volume get polymer-results sustainability_predictions.csv sustainability_predictions.csv")


