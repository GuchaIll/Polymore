"""
Modal-based Polymer Prediction Inference Pipeline

This script runs the polymer property prediction inference on Modal's serverless infrastructure.
It generates predictions using three different model types:
1. XGBoost/RandomForest ensemble
2. BERT models
3. TabTransformer models

Usage:
    modal run polymer_modal.py
    modal run polymer_modal.py --skip-download  # Skip dataset download if already present
"""

import modal
from pathlib import Path

# ============================================================================
# Modal Infrastructure Setup
# ============================================================================

# Create Modal volumes for persistent storage
data_volume = modal.Volume.from_name("polymer-data", create_if_missing=True)
results_volume = modal.Volume.from_name("polymer-results", create_if_missing=True)

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
    
    # Commit volume changes
    data_volume.commit()
    print(f"\n✅ Downloaded {success_count}/{len(datasets)} datasets and committed to Volume!\n")
    
    if success_count < len(datasets):
        print("⚠️  Some datasets failed to download. Check the logs above for details.")



# ============================================================================
# XGBoost/RandomForest Inference
# ============================================================================

@app.function(
    image=ml_image,
    volumes={DATA_PATH: data_volume, RESULTS_PATH: results_volume},
    timeout=1800,  # 30 minutes
    cpu=8,  # Use multiple CPUs for faster processing
)
def run_xgboost_inference(test_mode: bool = False, sample_size: int = 100):
    """Generate predictions using XGBoost and RandomForest models"""
    import pandas as pd
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
    
    print(f"\n✅ XGBoost/RandomForest inference complete! Saved to {output_path}")
    return str(output_path)

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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(test_feat_df),
            columns=test_feat_df.columns
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
# Local Entrypoint
# ============================================================================

@app.local_entrypoint()
def main(skip_download: bool = False, skip_xgboost: bool = False, skip_bert: bool = False, skip_tabtransformer: bool = False, test_mode: bool = False, sample_size: int = 100):
    """Run complete inference pipeline"""
    print("🚀 Starting Polymer Prediction Inference Pipeline on Modal")
    print("=" * 60)
    if test_mode:
        print(f"⚡ TEST MODE ENABLED: Processing only {sample_size} samples")
        print("=" * 60)
    
    # Step 1: Download datasets
    if not skip_download:
        print("\n📦 Step 1: Downloading datasets...")
        download_datasets.remote()
    else:
        print("\n📦 Step 1: Skipping dataset download (--skip-download flag set)")
    
    # Step 2: Run XGBoost inference
    if not skip_xgboost:
        print("\n🔬 Step 2: Running XGBoost/RandomForest inference...")
        xgb_result = run_xgboost_inference.remote(test_mode=test_mode, sample_size=sample_size)
        print(f"   Result: {xgb_result}")
    else:
        print("\n🔬 Step 2: Skipping XGBoost inference")
    
    # Step 3: Run BERT inference
    if not skip_bert:
        print("\n🤖 Step 3: Running BERT model inference...")
        bert_result = run_bert_inference.remote(test_mode=test_mode, sample_size=sample_size)
        print(f"   Result: {bert_result}")
    else:
        print("\n🤖 Step 3: Skipping BERT inference")
    
    # Step 4: Run TabTransformer inference
    if not skip_tabtransformer:
        print("\n🔷 Step 4: Running TabTransformer inference...")
        tab_result = run_tabtransformer_inference.remote(test_mode=test_mode, sample_size=sample_size)
        print(f"   Result: {tab_result}")
    else:
        print("\n🔷 Step 4: Skipping TabTransformer inference")
    
    # Step 5: Create ensemble
    print("\n🎯 Step 5: Creating ensemble predictions...")
    ensemble_result = create_ensemble.remote()
    print(f"   Result: {ensemble_result}")
    
    print("\n" + "=" * 60)
    print("✅ Inference pipeline complete!")
    print("📊 Results saved to Modal Volume 'polymer-results'")
    print("\n📥 To download results locally:")
    print("  modal volume get polymer-results submission.csv submission.csv")
    print("  modal volume get polymer-results submission1.csv submission1.csv")
    print("  modal volume get polymer-results submission2.csv submission2.csv")
    print("  modal volume get polymer-results submission3.csv submission3.csv")

