# %%
# Download Kaggle Datasets
# Install Kaggle API if not already installed
import subprocess
import sys
import os
import zipfile
from pathlib import Path

# Install kaggle if not present
try:
    import kaggle
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    import kaggle

# Setup data directory
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

print("📦 Downloading Kaggle datasets...")

# Define dataset mappings
DATASETS = {
    "neurips-open-polymer-prediction-2025": "competitions/neurips-open-polymer-prediction-2025",
    "external-polymer-data": "tasmim/external-polymer-data",
    "modred-dataset": "adamlogman/modred-dataset",
    "private-smile-bert-models": "defdet/private-smile-bert-models",
    "rdkit-2025-3-3-cp311": "senkin13/rdkit-2025-3-3-cp311",
    "smiles-extra-data": "dmitryuarov/smiles-extra-data",
    "smiles-bert-models": "defdet/smiles-bert-models",
    "smiles-deberta77m-tokenizer": "defdet/smiles-deberta77m-tokenizer",
    "tc-smiles": "minatoyukinaxlisa/tc-smiles"
}

def download_and_extract(dataset_name, dataset_path, is_competition=False):
    """Download and extract a Kaggle dataset or competition"""
    dest_dir = DATA_DIR / dataset_name
    dest_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if already downloaded
    if list(dest_dir.glob("*")):
        print(f"   ✅ {dataset_name} already exists, skipping download")
        return
    
    print(f"   ⬇️  Downloading {dataset_name}...")
    try:
        if is_competition:
            # Download competition data
            kaggle.api.competition_download_files(dataset_path.replace("competitions/", ""), path=str(dest_dir))
        else:
            # Download dataset
            kaggle.api.dataset_download_files(dataset_path, path=str(dest_dir), unzip=True)
        
        # Unzip if not auto-unzipped
        for zip_file in dest_dir.glob("*.zip"):
            print(f"   📂 Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            zip_file.unlink()  # Remove zip file after extraction
        
        print(f"   ✅ {dataset_name} downloaded successfully")
    except Exception as e:
        print(f"   ⚠️  Error downloading {dataset_name}: {e}")

# Download all datasets
for dataset_name, dataset_path in DATASETS.items():
    is_competition = dataset_path.startswith("competitions/")
    download_and_extract(dataset_name, dataset_path, is_competition)

print("\n✅ All datasets downloaded!\n")

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# List downloaded files
print("📂 Downloaded data files:")
for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
# This cell skips the rest of the notebook except during competition scoring reruns, saving GPU quota.
# Uncomment the buttom two lines to enable.

from IPython import get_ipython
from IPython.core.interactiveshell import ExecutionResult, ExecutionInfo

import os

ipython = get_ipython()

def no_op_run_cell(*args, **kwargs):
    info = ExecutionInfo(
        raw_cell="",
        store_history=False,
        silent=True,
        shell_futures=True,
        cell_id=None
    )
    return ExecutionResult(info)

#if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#    ipython.run_cell = no_op_run_cell

# %%
# Install RDKit from downloaded wheel
rdkit_wheel_path = DATA_DIR / "rdkit-2025-3-3-cp311" / "rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl"
if rdkit_wheel_path.exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", str(rdkit_wheel_path)])
    print("✅ RDKit installed successfully")
else:
    print("⚠️ RDKit wheel not found, trying to use existing installation")

# %%
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Fragments, Lipinski
from rdkit.Chem import rdmolops
# Data paths
BASE_PATH = str(DATA_DIR / 'neurips-open-polymer-prediction-2025') + '/'
RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
def get_canonical_smiles(smiles):
        """Convert SMILES to canonical form for consistency"""
        if not RDKIT_AVAILABLE:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return smiles

# %%
#Cell 3: Robust Data Loading with Complete R-Group Filtering
"""
Load competition data with complete filtering of problematic polymer notation
"""

print("📂 Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

print(f"   Training samples: {len(train)}")
print(f"   Test samples: {len(test)}")

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    # Check for any bad patterns
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    # Try to parse with RDKit if available
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles

# Clean and validate all SMILES
print("🔄 Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

# Remove invalid SMILES
invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    # Clean external SMILES
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    # Remove invalid SMILES and missing targets
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    # Group by canonical SMILES and average duplicates
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Fill missing values
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    
    # Add unique SMILES
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    return df_train

# Load external datasets with robust error handling
print("\n📂 Loading external datasets...")

external_datasets = []

# Function to safely load datasets
def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f"   ✅ {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f"   ⚠️ {description} failed: {str(e)[:100]}")
        return False

# Load each dataset
safe_load_dataset(
    str(DATA_DIR / 'tc-smiles' / 'Tc_SMILES.csv'),
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)

safe_load_dataset(
    str(DATA_DIR / 'external-polymer-data' / 'TgSS_enriched_cleaned.csv'),
    'Tg', 
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)

safe_load_dataset(
    str(DATA_DIR / 'smiles-extra-data' / 'JCIM_sup_bigsmiles.csv'),
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)

safe_load_dataset(
    str(DATA_DIR / 'smiles-extra-data' / 'data_tg3.xlsx'),
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)

safe_load_dataset(
    str(DATA_DIR / 'smiles-extra-data' / 'data_dnst1.xlsx'),
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)

#safe_load_dataset(
#    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
#    'FFV', 
#    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
#    'dataset 4'
#)

# Integrate external data
print("\n🔄 Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

for target, dataset in external_datasets:
    print(f"   Processing {target} data...")
    train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\n📊 Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\n✅ Data integration complete with clean SMILES!")

# %%

def separate_subtables(train_df):
	
	labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
	subtables = {}
	for label in labels:
		subtables[label] = train_df[['SMILES', label]][train_df[label].notna()]
	return subtables


# %%

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
	"""
	Augments a list of SMILES strings by generating randomized versions.

	Parameters:
		smiles_list (list of str): Original SMILES strings.
		labels (list or np.array): Corresponding labels.
		num_augments (int): Number of augmentations per SMILES.

	Returns:
		tuple: (augmented_smiles, augmented_labels)
	"""
	augmented_smiles = []
	augmented_labels = []

	for smiles, label in zip(smiles_list, labels):
		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			continue
		# Add original
		augmented_smiles.append(smiles)
		augmented_labels.append(label)
		# Add randomized versions
		for _ in range(num_augments):
			rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
			augmented_smiles.append(rand_smiles)
			augmented_labels.append(label)

	return augmented_smiles, np.array(augmented_labels)

from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolWt, MolLogP
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator

import networkx as nx
def smiles_to_combined_fingerprints_with_descriptors(smiles_list, radius=2, n_bits=128):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints
            morgan_fp = generator.GetFingerprint(mol)
            #atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            #torsion_fp = torsion_gen.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                #np.array(atom_pair_fp),
                #np.array(torsion_fp),
                np.array(maccs_fp)
            ])
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
            #fingerprints.append(np.zeros(n_bits * 3 + 167))
            fingerprints.append(np.zeros(n_bits  + 167))
            descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    return np.array(fingerprints), descriptors, valid_smiles, invalid_indices

def smiles_to_combined_fingerprints_with_descriptorsOriginal(smiles_list, radius=2, n_bits=128):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints
            morgan_fp = generator.GetFingerprint(mol)
            #atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            #torsion_fp = torsion_gen.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                #np.array(atom_pair_fp),
                #np.array(torsion_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)

            # All RDKit Descriptors
            descriptor_values = {}
            for name, func in Descriptors.descList:
                try:
                    descriptor_values[name] = func(mol)
                except:
                    descriptor_values[name] = None

            # Add specific descriptors explicitly
            descriptor_values['MolWt'] = MolWt(mol)
            descriptor_values['LogP'] = MolLogP(mol)
            descriptor_values['TPSA'] = CalcTPSA(mol)
            descriptor_values['RotatableBonds'] = CalcNumRotatableBonds(mol)
            descriptor_values['NumAtoms'] = mol.GetNumAtoms()
            descriptor_values['SMILES'] = smiles
            #descriptor_values['RadiusOfGyration'] =CalcRadiusOfGyration(mol)

            descriptors.append(descriptor_values)
            valid_smiles.append(smiles)
        else:
            #fingerprints.append(np.zeros(n_bits * 3 + 167))
            fingerprints.append(np.zeros( 167))
            descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    return np.array(fingerprints), descriptors, valid_smiles, invalid_indices

def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
	try:
		mol = Chem.MolFromSmiles(smile)
		canon_smile = Chem.MolToSmiles(mol, canonical=True)
		return canon_smile
	except:
		return np.nan

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

# %%

#required_descriptors = {'MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}
#required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path'}
required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}
#required_descriptors = {}

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



# %%
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    """
    Augments a dataset using Gaussian Mixture Models.

    Parameters:
    - X: pd.DataFrame or np.ndarray — feature matrix
    - y: pd.Series or np.ndarray — target values
    - n_samples: int — number of synthetic samples to generate
    - n_components: int — number of GMM components
    - random_state: int — random seed for reproducibility

    Returns:
    - X_augmented: pd.DataFrame — augmented feature matrix
    - y_augmented: pd.Series — augmented target values
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    df = X.copy()
    df['Target'] = y.values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)

    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']

    return X_augmented, y_augmented


# %%
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor


train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
#labels = ['Tc']

output_df = pd.DataFrame({
	'id': test_ids
})


for label in labels:
	print(f"Processing label: {label}")
	print(subtables[label].head())
	print(subtables[label].shape)
	original_smiles = subtables[label]['SMILES'].tolist()
	original_labels = subtables[label][label].values

	original_smiles, original_labels = augment_smiles_dataset(original_smiles, original_labels, num_augments=1)
	fingerprints, descriptors, valid_smiles, invalid_indices\
		=smiles_to_combined_fingerprints_with_descriptors(original_smiles, radius=2, n_bits=128)
	# descriptors, valid_smiles, invalid_indices\
	#	 =smiles_to_descriptors_with_fingerprints(original_smiles, radius=2, n_bits=128)

	X=pd.DataFrame(descriptors)
	X=X.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge', 'SMILES'],axis=1)
	y = np.delete(original_labels, invalid_indices)
	
	# pd.DataFrame(X).to_csv(f"./mats/{label}.csv")
	# pd.DataFrame(y).to_csv(f"./mats/{label}label.csv", header=None)
	
	# binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
	# pd.DataFrame(binned).to_csv(f"./mats/{label}integerlabel.csv", header=None, index=False)
	X = X.filter(filters[label])
	# Convert fingerprints array to DataFrame
	fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])

	print(fp_df.shape)
	# Reset index to align with X
	fp_df.reset_index(drop=True, inplace=True)
	X.reset_index(drop=True, inplace=True)
	# Concatenate descriptors and fingerprints
	X = pd.concat([X, fp_df], axis=1)
    
	print(f"After concat: {X.shape}")
	
	# Set the variance threshold
	threshold = 0.01

	# Apply VarianceThreshold
	selector = VarianceThreshold(threshold=threshold)
	
	X = selector.fit_transform(X)

	print(f"After variance cut: {X.shape}")

	# Assuming you have X and y loaded
    
	n_samples = 1000

	X, y = augment_dataset(X, y, n_samples=n_samples)
	print(f"After augment cut: {X.shape}")


	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
	
	if label=="Tg":
		Model= XGBRegressor(n_estimators= 2173, learning_rate= 0.0672418745539774, max_depth= 6, reg_lambda= 5.545520219149715)
	if label=='Rg':
		Model = XGBRegressor(n_estimators= 520, learning_rate= 0.07324113948440986, max_depth= 5, reg_lambda=0.9717380315982088)
	if label=='FFV':
# Best parameters found: {'n_estimators': 2202, 'learning_rate': 0.07220580588586338, 'max_depth': 4, 'reg_lambda': 2.8872976032666493}
		Model = XGBRegressor(n_estimators= 2202, learning_rate= 0.07220580588586338, max_depth= 4, reg_lambda= 2.8872976032666493)
	if label=='Tc':
		Model = XGBRegressor(n_estimators= 1488, learning_rate= 0.010456188013762864, max_depth= 5, reg_lambda= 9.970345982204618)
#Best parameters found: {'n_estimators': 1488, 'learning_rate': 0.010456188013762864, 'max_depth': 5, 'reg_lambda': 9.970345982204618}
	if label=='Density':
		Model = XGBRegressor(n_estimators= 1958, learning_rate= 0.10955287548172478, max_depth= 5, reg_lambda= 3.074470087965767)

	RFModel=RandomForestRegressor(random_state=42)
	Model.fit(X_train,y_train)
	RFModel.fit(X_train,y_train)
	y_pred=Model.predict(X_test)
	print(mean_absolute_error(y_pred,y_test))

	Model.fit(X,y)
	RFModel.fit(X,y)
	# Predict on test set
	#test_smiles = test_df['SMILES'].str.replace('*', 'C')

	fingerprints, descriptors, valid_smiles, invalid_indices\
		=smiles_to_combined_fingerprints_with_descriptors(test_smiles, radius=2, n_bits=128)
	test=pd.DataFrame(descriptors)
	test=test.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge', 'SMILES'],axis=1)

	test = test.filter(filters[label])
    # Convert fingerprints array to DataFrame
	fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
    
	# Reset index to align with X
	fp_df.reset_index(drop=True, inplace=True)
	test.reset_index(drop=True, inplace=True)
    # Concatenate descriptors and fingerprints
	test = pd.concat([test, fp_df], axis=1)
	test = selector.transform(test)
	print(test.shape)

	y_pred1=Model.predict(test).flatten()
	y_pred2=RFModel.predict(test).flatten()
	y_pred=y_pred1*.6+y_pred2*.4
	print(y_pred)


	new_column_name = label
	output_df[new_column_name] = y_pred

print(output_df)


output_df.to_csv('submission1.csv', index=False)

# %%
# Install RDKit from downloaded wheel (second installation for BERT models)
if rdkit_wheel_path.exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", str(rdkit_wheel_path)])
    print("✅ RDKit installed successfully (BERT section)")
else:
    print("⚠️ RDKit wheel not found, using existing installation")

# %%
import torch
import pandas as pd
import joblib
from transformers import PreTrainedModel, AutoConfig, BertModel, BertTokenizerFast, BertConfig, AutoModel, AutoTokenizer
from sklearn.metrics import mean_absolute_error
from torch import nn
from transformers.activations import ACT2FN
from tqdm import tqdm
import numpy as np

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
        context_token = hidden_states[:, 0] # CLS token
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.activation](pooled_output)
        return pooled_output

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        
        self.pooler = ContextPooler(config)

        pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.output = torch.nn.Linear(pooler_output_dim, 1) # Still predicting one label at a time. Kinda stupid

    def forward(
        self,
        input_ids,
        scaler,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)
        
        # Final regression output
        regression_output = self.output(pooled_output)

        loss = None
        true_loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()

            unscaled_labels = scaler.inverse_transform(labels.cpu().numpy())
            unscaled_outputs = scaler.inverse_transform(regression_output.cpu().detach().numpy())
            
            loss = loss_fn(regression_output, labels)
            true_loss = mean_absolute_error(unscaled_outputs, unscaled_labels)

        return {
            "loss": loss,
            "logits": regression_output,
            "true_loss": true_loss
        }

# %%
BATCH_SIZE = 16

def tokenize_smiles(seq):
    seq = [tokenizer.cls_token + smiles for smiles in seq] # If we pass a string, tokenizer will smartly think we want to create a sequence for each symbol
    tokenized = tokenizer(seq, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    return tokenized

def load_model(path):
    config = AutoConfig.from_pretrained(str(DATA_DIR / 'smiles-deberta77m-tokenizer'))
    model = CustomModel(config).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model


def make_predictions(model, scaler, smiles_seq):
    aggregated_preds = []
    for smiles in smiles_seq:
        smiles = [smiles]
        smiles_tokenized = tokenize_smiles(smiles)

        input_ids = smiles_tokenized['input_ids'].cuda()
        attention_mask = smiles_tokenized['attention_mask'].cuda()
        with torch.no_grad():
            preds = model(input_ids=input_ids, scaler=scaler, attention_mask=attention_mask)['logits'].cpu().numpy()
        
        true_preds = scaler.inverse_transform(preds).flatten()
        aggregated_preds.append(true_preds.tolist())
    return np.array(aggregated_preds)


test = pd.read_csv(str(DATA_DIR / 'neurips-open-polymer-prediction-2025' / 'test.csv'))
test_copy = test.copy()

smiles_test = test['SMILES'].to_list()

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

scalers = joblib.load(str(DATA_DIR / 'smiles-bert-models' / 'target_scalers.pkl'))
tokenizer = AutoTokenizer.from_pretrained(str(DATA_DIR / 'smiles-deberta77m-tokenizer'))

# %%
import pandas as pd
import numpy as np
from rdkit import Chem
import random
from typing import Optional, List, Union

def augment_smiles_dataset(df: pd.DataFrame,
                               smiles_column: str = 'SMILES',
                               augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                               n_augmentations: int = 100,
                               preserve_original: bool = True,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            augmented_rows.append(original_row)
        
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

test = augment_smiles_dataset(test)

# %%
preds_mapping = {}

for i in tqdm(range(len(targets))):
    target = targets[i]
    scaler = scalers[i]

    model_path = str(DATA_DIR / 'private-smile-bert-models' / f'warm_smiles_model_{target}_target.pth') # Not actually warm
    
    model = load_model(model_path)
    true_preds = []

    for i, data in test.groupby('id'):
        test_smiles = data['SMILES'].to_list()
        augmented_preds = make_predictions(model, scaler, test_smiles)
    
        average_pred = np.median(augmented_preds)
    
        true_preds.append(float(average_pred.flatten()[0]))

    preds_mapping[target] = true_preds

# %%
submission = pd.DataFrame(preds_mapping)
submission['id'] = test_copy['id']
submission.to_csv('submission2.csv', index=False)

# %%
import gc
import torch
import pickle
import numpy as np
import pandas as pd
import polars as pl
from torch import nn
import seaborn as sns
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)




import torch
import torch.nn as nn



class EnhancedEmbedding(nn.Module):
    def __init__(self, categories, num_continuous, embedding_dim):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, embedding_dim) for num_cat in categories
        ])
        self.cont_emb = nn.Linear(num_continuous, embedding_dim) if num_continuous > 0 else None

        self.feature_type_embed = nn.Embedding(len(categories) + (1 if num_continuous > 0 else 0), embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        if self.cont_emb:
            nn.init.xavier_uniform_(self.cont_emb.weight)

    def forward(self, x_cat, x_cont):
        B = x_cat.size(0)
        cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        type_indices_cat = torch.arange(len(self.cat_embeddings), device=x_cat.device).unsqueeze(0).expand(B, -1)
        cat_tokens += self.feature_type_embed(type_indices_cat)

        if x_cont is not None and self.cont_emb:
            cont_token = self.cont_emb(x_cont).unsqueeze(1)
            cont_token += self.feature_type_embed(torch.full((B, 1), len(self.cat_embeddings), device=x_cat.device))
            tokens = torch.cat([cat_tokens, cont_token], dim=1)
        else:
            tokens = cat_tokens

        cls_tokens = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        return tokens

class GatedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        gate_input = torch.cat([x, attn_out], dim=-1)
        gated_output = torch.sigmoid(self.gate(gate_input))
        x = self.norm1((1 - gated_output) * x + gated_output * attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn_score(x), dim=1)
        return (weights * x).sum(dim=1)

class EnhancedFTTransformer(nn.Module):
    def __init__(self, categories, num_continuous, embedding_dim, num_heads,
                 num_layers, ff_hidden_dim, dropout=0.1, output_dim=1):
        super().__init__()
        self.embedding = EnhancedEmbedding(categories, num_continuous, embedding_dim)

        self.transformer_blocks = nn.Sequential(*[
            GatedTransformerBlock(embedding_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.attn_pooling = AttentionPooling(embedding_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, output_dim)
        )

    def forward(self, x_cat, x_cont):
        tokens = self.embedding(x_cat, x_cont)  # Shape: [B, T, D]
        tokens = self.transformer_blocks(tokens)

        cls = tokens[:, 0]
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1)[0]

        fused = torch.cat([cls, mean_pool, max_pool], dim=1)
        return self.output_layer(fused).squeeze()


# --- Configuration ---
class CFG:
    """
    Configuration class for defining global parameters.
    """
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5
    PATH = str(DATA_DIR / 'neurips-open-polymer-prediction-2025') + '/'
    TC_SMILES_PATH = str(DATA_DIR / 'tc-smiles' / 'Tc_SMILES.csv')
    JCIM_SMILES_PATH = str(DATA_DIR / 'smiles-extra-data' / 'JCIM_sup_bigsmiles.csv')
    DATA_TG3_PATH = str(DATA_DIR / 'smiles-extra-data' / 'data_tg3.xlsx')
    DATA_DNST1_PATH = str(DATA_DIR / 'smiles-extra-data' / 'data_dnst1.xlsx')
    NULL_FOR_SUBMISSION = -9999

    # TabTransformer specific parameters
    # TabTransformer specific parameters"
    EMBEDDING_DIM = 32 
    NUM_HEADS = 2 
    NUM_ENCODERS = 1
    FF_HIDDEN_DIM = 128 
    DROPOUT = 0.15
    TAB_LEARNING_RATE = 0.001
    TAB_EPOCHS = 100
    TAB_BATCH_SIZE = 32
    TAB_EARLY_STOPPING_PATIENCE = 15
    WEIGHT_DECAY = 0.001
    NUM_LAYERS = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR_DECAY_RATE = 0.9658825


# --- Data Loading ---
def load_data():
    """
    Loads training and testing datasets.
    """
    train_df = pd.read_csv(CFG.PATH + 'train.csv')
    test_df = pd.read_csv(CFG.PATH + 'test.csv')
    return train_df, test_df

# --- SMILES Canonicalization ---
def make_smile_canonical(smile):
    """
    Converts a SMILES string to its canonical form to ensure uniqueness.
    Returns np.nan if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except Exception:
        return np.nan

# --- Extra Data Loading and Preprocessing ---
def load_and_preprocess_extra_data():
    """
    Loads and preprocesses various external datasets.
    """
    data_tc = pd.read_csv(CFG.TC_SMILES_PATH).rename(columns={'TC_mean': 'Tc'})
    data_tc['SMILES'] = data_tc['SMILES'].apply(make_smile_canonical)
    data_tc = data_tc.groupby('SMILES', as_index=False)['Tc'].mean() 

    data_tg2 = pd.read_csv(CFG.JCIM_SMILES_PATH, usecols=['SMILES', 'Tg (C)']).rename(columns={'Tg (C)': 'Tg'})
    data_tg2['SMILES'] = data_tg2['SMILES'].apply(make_smile_canonical)
    data_tg2 = data_tg2.groupby('SMILES', as_index=False)['Tg'].mean() 

    data_tg3 = pd.read_excel(CFG.DATA_TG3_PATH).rename(columns={'Tg [K]': 'Tg'})
    data_tg3['Tg'] = data_tg3['Tg'] - 273.15
    data_tg3['SMILES'] = data_tg3['SMILES'].apply(make_smile_canonical)
    data_tg3 = data_tg3.groupby('SMILES', as_index=False)['Tg'].mean() 

    data_dnst = pd.read_excel(CFG.DATA_DNST1_PATH).rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
    data_dnst['SMILES'] = data_dnst['SMILES'].apply(make_smile_canonical)
    data_dnst = data_dnst[(data_dnst['SMILES'].notnull()) & (data_dnst['Density'].notnull()) & (data_dnst['Density'] != 'nylon')]
    data_dnst['Density'] = data_dnst['Density'].astype('float64')
    data_dnst['Density'] -= 0.118
    data_dnst = data_dnst.groupby('SMILES', as_index=False)['Density'].mean() 

    return data_tc, data_tg2, data_tg3, data_dnst

def add_extra_data(df_main, df_extra, target):
    """
    Adds extra data to the main DataFrame, prioritizing existing competition data.
    """
    n_samples_before = df_main[target].notnull().sum()

    merged_df = pd.merge(df_main, df_extra, on='SMILES', how='left', suffixes=('', '_extra'))
    df_main[target] = merged_df[target].fillna(merged_df[f'{target}_extra'])

    unique_smiles_main = set(df_main['SMILES'])
    unique_smiles_extra_only = df_extra[~df_extra['SMILES'].isin(unique_smiles_main)].copy()

    df_main = pd.concat([df_main, unique_smiles_extra_only], axis=0, ignore_index=True)

    n_samples_after = df_main[target].notnull().sum()
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES added: {len(unique_smiles_extra_only)}')

    if f'{target}_extra' in df_main.columns:
        df_main = df_main.drop(columns=[f'{target}_extra'])

    return df_main

# --- Feature Engineering ---
USELESS_DESCRIPTORS = {
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester',
    'fr_C_O_noCOO'
}

ALL_DESC_TUPLES = [(desc[0], desc[1]) for desc in Descriptors.descList if desc[0] not in USELESS_DESCRIPTORS]
DESC_NAMES = [desc[0] for desc in ALL_DESC_TUPLES]

def compute_molecular_descriptors(mol):
    """
    Computes RDKit molecular descriptors for a given RDKit molecule object.
    Returns a list of descriptor values.
    """
    if mol is None:
        return [np.nan] * len(DESC_NAMES)
    return [desc_func(mol) for _, desc_func in ALL_DESC_TUPLES]

def compute_graph_features_for_mol(mol):
    """
    Computes graph-based features for a given RDKit molecule object.
    Returns a dictionary of graph features.
    """
    if mol is None:
        return {'graph_diameter': 0, 'avg_shortest_path': 0, 'num_cycles': 0}

    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_diameter = 0
    avg_shortest_path = 0
    if nx.is_connected(G) and len(G) > 1:
        try:
            graph_diameter = nx.diameter(G)
            avg_shortest_path = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            pass

    num_cycles = len(list(nx.cycle_basis(G)))

    return {
        'graph_diameter': graph_diameter,
        'avg_shortest_path': avg_shortest_path,
        'num_cycles': num_cycles
    }

def generate_features(df):
    """
    Generates all molecular and graph features for a DataFrame.
    """
    mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]

    descriptors_list = [compute_molecular_descriptors(mol) for mol in mols]
    descriptors_df = pd.DataFrame(descriptors_list, columns=DESC_NAMES)

    graph_features_list = [compute_graph_features_for_mol(mol) for mol in mols]
    graph_features_df = pd.DataFrame(graph_features_list)

    result = pd.concat([descriptors_df, graph_features_df], axis=1)
    result = result.replace([-np.inf, np.inf], np.nan)
    return result

# --- TabTransformer Model Definition ---
class Embeddings(nn.Module):
    def __init__(self, categories, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_unique + 1, embedding_dim) 
            for num_unique in categories
        ])
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x):
        return torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], 1)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output)) 

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output)) 
        return x

class TabTransformer(nn.Module):
    def __init__(self,
                 categories, 
                 num_continuous, 
                 embedding_dim,
                 num_heads, 
                 num_encoders, 
                 ff_hidden_dim, 
                 dropout=0.1,
                 output_dim=1): 
        super().__init__()

        self.categorical_columns = len(categories)
        self.continuous_columns = num_continuous
        self.embedding_dim = embedding_dim

        # Categorical embeddings
        self.categorical_embeddings = Embeddings(categories, embedding_dim)

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_encoders)
        ])

        # Final projection for combined features
        total_feature_dim = self.categorical_columns * embedding_dim + self.continuous_columns
        self.mlp_head = nn.Sequential(
            nn.Linear(total_feature_dim, ff_hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, output_dim)
        )

    def forward(self, x_cat, x_cont):

        # Process categorical features
        if self.categorical_columns > 0:
            x_cat_embedded = self.categorical_embeddings(x_cat) 
            x_cat_embedded = x_cat_embedded.view(x_cat_embedded.size(0), self.categorical_columns, self.embedding_dim) 

            # Apply Transformer blocks
            for block in self.transformer_blocks:
                x_cat_embedded = block(x_cat_embedded)

            # Flatten the output of transformer for concatenation
            x_cat_processed = x_cat_embedded.view(x_cat_embedded.size(0), -1)
        else:
            x_cat_processed = torch.empty(x_cont.size(0), 0).to(CFG.DEVICE)

        # Concatenate with continuous features
        combined_features = torch.cat([x_cat_processed, x_cont], dim=1)

        # Final MLP head
        return self.mlp_head(combined_features).squeeze()


# --- Custom Dataset for PyTorch DataLoader ---
class TabularDataset(Dataset):
    def __init__(self, df, categorical_cols, continuous_cols, target_col=None):
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col

        self.categorical_data = torch.tensor(df[categorical_cols].values, dtype=torch.long)
        self.continuous_data = torch.tensor(df[continuous_cols].values, dtype=torch.float32)

        if target_col and target_col in df.columns:
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.categorical_data)

    def __getitem__(self, idx):
        cat_features = self.categorical_data[idx]
        cont_features = self.continuous_data[idx]
        if self.targets is not None:
            return cat_features, cont_features, self.targets[idx]
        return cat_features, cont_features


# --- Evaluation Metric ---
def mae(y_true, y_pred):
    """
    Calculates Mean Absolute Error (MAE).
    """
    return np.mean(np.abs(y_true - y_pred))

# --- TabTransformer Model Training and Prediction ---
def train_and_predict_tabtransformer(train_df_full, test_df_full):
    """
    Trains TabTransformer models for each target and generates OOF and test predictions.
    """
    test_preds_df = test_df_full[['id', 'SMILES']].copy()
    for target_col in CFG.TARGETS:
        test_preds_df[target_col] = 0.0

    oof_preds_df = train_df_full[['id'] + CFG.TARGETS].copy()
    for target_col in CFG.TARGETS:
        oof_preds_df[target_col] = np.nan

    

    CATEGORICAL_THRESHOLD = 20 

    all_features = [col for col in train_df_full.columns if col not in ['id', 'SMILES'] + CFG.TARGETS]

    # Pre-process features for TabTransformer
    # We need to map categorical features to integer indices
    # And scale continuous features

    # Store encoders and scalers
    categorical_encoders = {}
    scalers = {}

    # Determine which features are categorical and which are continuous based on *entire* dataset
    global_categorical_cols = []
    global_continuous_cols = []  

    for col in all_features:
        # Check if the feature exists in both train and test and is not entirely NaN
        if col in train_df_full.columns and col in test_df_full.columns and \
           not (train_df_full[col].isnull().all() and test_df_full[col].isnull().all()):

            combined_unique_count = pd.concat([train_df_full[col], test_df_full[col]]).nunique(dropna=False)

            if combined_unique_count <= CATEGORICAL_THRESHOLD:
                global_categorical_cols.append(col)
            else:
                global_continuous_cols.append(col)

    # Fit LabelEncoders for categorical features on the combined data
    for col in global_categorical_cols:
        le = LabelEncoder() 
        combined_values = pd.concat([train_df_full[col].astype(str), test_df_full[col].astype(str)])
        le.fit(combined_values)
        categorical_encoders[col] = le
        train_df_full[col] = le.transform(train_df_full[col].astype(str))
        test_df_full[col] = le.transform(test_df_full[col].astype(str))

    # Fit StandardScaler for continuous features on the training data
    for col in global_continuous_cols:
        scaler = StandardScaler()
        train_mean = train_df_full[col].mean()
        train_df_full[col].fillna(train_mean, inplace=True)
        test_df_full[col].fillna(train_mean, inplace=True) 

        train_df_full[col] = scaler.fit_transform(train_df_full[[col]])
        test_df_full[col] = scaler.transform(test_df_full[[col]])
        scalers[col] = scaler

    print(f"\nIdentified {len(global_categorical_cols)} categorical features and {len(global_continuous_cols)} continuous features.")


    for target in CFG.TARGETS:
    # for target in ['Rg']:        
        print(f'\n\nTARGET: {target}')

        train_part = train_df_full[train_df_full[target].notnull()].reset_index(drop=True)

        oof_tab_target = np.zeros(len(train_part))      
        scores = []

        kf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)

        categories_counts = [len(categorical_encoders[col].classes_) for col in global_categorical_cols]

        for i, (trn_idx, val_idx) in enumerate(kf.split(train_part)):
            print(f"\n--- Fold {i+1} ---")

            x_trn_cat = train_part.loc[trn_idx, global_categorical_cols]
            x_trn_cont = train_part.loc[trn_idx, global_continuous_cols]
            y_trn = train_part.loc[trn_idx, target]

            x_val_cat = train_part.loc[val_idx, global_categorical_cols]
            x_val_cont = train_part.loc[val_idx, global_continuous_cols]
            y_val = train_part.loc[val_idx, target]

            # Create PyTorch Datasets and DataLoaders
            train_dataset = TabularDataset(train_part.loc[trn_idx], global_categorical_cols, global_continuous_cols, target)
            val_dataset = TabularDataset(train_part.loc[val_idx], global_categorical_cols, global_continuous_cols, target)
            test_dataset = TabularDataset(test_df_full, global_categorical_cols, global_continuous_cols) 

            train_loader = DataLoader(train_dataset, batch_size=CFG.TAB_BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CFG.TAB_BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=CFG.TAB_BATCH_SIZE, shuffle=False)


            # EnhancedFTTransformer
            model = EnhancedFTTransformer(
                categories=categories_counts,
                num_continuous=len(global_continuous_cols),
                embedding_dim=CFG.EMBEDDING_DIM,
                num_heads=CFG.NUM_HEADS,
                num_layers=CFG.NUM_LAYERS,
                ff_hidden_dim=CFG.FF_HIDDEN_DIM,
                dropout=CFG.DROPOUT,
                output_dim=1
            ).to(CFG.DEVICE)

            if target == 'FFV':
                CFG.TAB_LEARNING_RATE = 1e-3 #5e-4
                CFG.TAB_EPOCHS = 100
                CFG.LR_DECAY_RATE = 0.99999        
            if target == 'Rg':
                CFG.TAB_LEARNING_RATE = 1e-3 #5e-4
                CFG.TAB_EPOCHS = 80
                CFG.LR_DECAY_RATE = 0.92588

            optimizer = torch.optim.Adam(model.parameters(), lr=CFG.TAB_LEARNING_RATE,weight_decay=CFG.WEIGHT_DECAY)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: CFG.LR_DECAY_RATE ** epoch)
            criterion = nn.L1Loss() 

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(CFG.TAB_EPOCHS):
                model.train()
                total_loss = 0
                for cat_batch, cont_batch, target_batch in train_loader:
                    cat_batch, cont_batch, target_batch = cat_batch.to(CFG.DEVICE), cont_batch.to(CFG.DEVICE), target_batch.to(CFG.DEVICE)

                    optimizer.zero_grad()
                    out = model(cat_batch, cont_batch)
                    loss = criterion(out,target_batch)
                    # loss = criterion(out.flatten(),target_batch.repeat_interleave(CFG.TAB_BATCH_SIZE))
                    # print(out.flatten().shape,target_batch.repeat_interleave(CFG.TAB_BATCH_SIZE).shape)
                    # print(loss.item())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(target_batch) 
                lr_scheduler.step()

                # Validation phase
                model.eval()
                val_preds_list = []
                val_true_list = []
                val_loss_epoch = 0
                with torch.no_grad():
                    for cat_batch, cont_batch, target_batch in val_loader:
                        cat_batch, cont_batch, target_batch = cat_batch.to(CFG.DEVICE), cont_batch.to(CFG.DEVICE), target_batch.to(CFG.DEVICE)
                        out = model(cat_batch, cont_batch)
                        # print(f" out shape {out.shape}")
                        # print(f"out mean shape {out.mean(1)}")
                        # loss = criterion(out.flatten(),target_batch.repeat_interleave(CFG.TAB_BATCH_SIZE))
                        loss = criterion(out,target_batch)
                        val_loss_epoch += loss.item() * len(target_batch)
                        # val_preds_list.extend(out.sum(1).cpu().tolist())
                        val_preds_list.extend(out.cpu().tolist())
                        val_true_list.extend(target_batch.cpu().tolist())

                current_val_loss = val_loss_epoch / len(val_dataset)
                current_val_mae = mae(np.array(val_true_list), np.array(val_preds_list))

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'./tabtransformer_model_{target}_fold_{i+1}_best.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= CFG.TAB_EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping at epoch {epoch+1} for fold {i+1}, target {target}")
                        break

                if epoch % 10 == 0 or epoch == CFG.TAB_EPOCHS -1:
                    print(f"Epoch {epoch+1}/{CFG.TAB_EPOCHS}, Train Loss: {total_loss/len(train_dataset):.5f}, Val Loss: {current_val_loss:.5f}, Val MAE: {current_val_mae:.5f}")

            # Load the best model for OOF and test predictions
            model.load_state_dict(torch.load(f'./tabtransformer_model_{target}_fold_{i+1}_best.pt'))
            model.eval()

            # OOF predictions
            oof_fold_preds = []
            with torch.no_grad():
                for cat_batch, cont_batch, _ in val_loader: 
                    cat_batch, cont_batch = cat_batch.to(CFG.DEVICE), cont_batch.to(CFG.DEVICE)
                    out = model(cat_batch, cont_batch)
                    oof_fold_preds.extend(out.cpu().tolist())
                    # oof_fold_preds.extend(out.mean(1).cpu().tolist())
                    # print(oof_fold_preds)

            # Assign OOF predictions back to the full OOF dataframe
            oof_tab_target[val_idx] = np.array(oof_fold_preds)
            # train.loc[train[target].notnull(), f'{target}_pred'] = np.concatenate(oof_fold_preds)
            # train_df_full.loc[val_idx, f'{target}_pred'] = oof_fold_preds

            fold_score = mae(np.array(val_true_list), np.array(oof_fold_preds))
            scores.append(fold_score)
            print(f'Final MAE for Fold {i+1}: {np.round(fold_score, 5)}')

            # Test predictions
            fold_test_preds = []
            with torch.no_grad():
                for cat_batch, cont_batch in test_loader:
                    cat_batch, cont_batch = cat_batch.to(CFG.DEVICE), cont_batch.to(CFG.DEVICE)
                    out = model(cat_batch, cont_batch)
                    # fold_test_preds.extend(out.mean(1).squeeze(1).cpu().tolist())
                    fold_test_preds.extend(out.cpu().tolist())

                    # print(np.concatenate(fold_test_preds).shape)
            test_preds_df[target] += np.array(fold_test_preds) / CFG.FOLDS      

        # train.loc[train[target].notnull(), f'{target}_pred']  = oof_preds_df[target].values   
        train_df_full.loc[train_df_full[target].notnull(),f'{target}_pred'] = oof_tab_target

        print(f'\nMean MAE for {target}: {np.mean(scores):.5f}')
        print(f'Std MAE for {target}: {np.std(scores):.5f}')
        print('-'*30)

    return oof_preds_df, test_preds_df, train_df_full

# --- Weighted Mean Absolute Error (WMAE) Metric ---
MINMAX_DICT = {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}

def scaling_error(labels, preds, property_name):
    """
    Calculates the scaled absolute error for a given property.
    """
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property_name]
    label_range = max_val - min_val
    return np.mean(error / (label_range + 1e-9))

def get_property_weights(labels_df):
    """
    Calculates weights for each property based on the number of non-null labels.
    """
    property_weight = []
    for property_name in MINMAX_DICT.keys():
        valid_num = np.sum(labels_df[property_name] != CFG.NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / (property_weight + 1e-9))
    return (property_weight / np.sum(property_weight)) * len(property_weight)

def wmae_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates the Weighted Mean Absolute Error (WMAE) score.
    """
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []

    solution_aligned = solution.set_index(row_id_column_name).reindex(submission[row_id_column_name]).reset_index()

    property_weights = get_property_weights(solution_aligned[chemical_properties])

    for i, property_name in enumerate(chemical_properties):
        is_labeled = solution_aligned[property_name] != CFG.NULL_FOR_SUBMISSION

        if np.any(is_labeled):
            mae_val = scaling_error(solution_aligned.loc[is_labeled, property_name],
                                     submission.loc[is_labeled, property_name],
                                     property_name)
            property_maes.append(mae_val)
        else:
            property_maes.append(0.0)

    if not property_maes or np.sum(property_weights) == 0:
        raise RuntimeError('No labels or all property weights are zero. Cannot calculate WMAE.')

    return float(np.average(property_maes, weights=property_weights))

# --- Main Execution Flow ---
if __name__ == '__main__':
    print(f"Using device: {CFG.DEVICE}")

    # Load Data
    train_df, test_df = load_data()

    # Canonicalize SMILES
    print("Canonicalizing SMILES strings...")
    train_df['SMILES'] = train_df['SMILES'].apply(make_smile_canonical)
    test_df['SMILES'] = test_df['SMILES'].apply(make_smile_canonical)
    initial_train_rows = len(train_df)
    initial_test_rows = len(test_df)
    train_df = train_df.dropna(subset=['SMILES']).reset_index(drop=True)
    test_df = test_df.dropna(subset=['SMILES']).reset_index(drop=True)
    print(f"Dropped {initial_train_rows - len(train_df)} rows from train due to invalid SMILES.")
    print(f"Dropped {initial_test_rows - len(test_df)} rows from test due to invalid SMILES.")
    print("SMILES canonicalization complete.")

    # Load and Add Extra Data
    print("\nLoading and integrating extra data...")
    data_tc, data_tg2, data_tg3, data_dnst = load_and_preprocess_extra_data()

    train_df = add_extra_data(train_df, data_tc, 'Tc')
    train_df = add_extra_data(train_df, data_tg2, 'Tg')
    train_df = add_extra_data(train_df, data_tg3, 'Tg')
    train_df = add_extra_data(train_df, data_dnst, 'Density')

    print('\n--- SMILES count for training after extra data addition ---')
    for t in CFG.TARGETS:
        print(f'"{t}": {train_df[t].notnull().sum()}')

    # Feature Engineering (Generate RDKit and Graph features as tabular input)
    print("\nGenerating molecular and graph features for train and test sets...")
    train_features = generate_features(train_df)
    test_features = generate_features(test_df)

    # Concatenate features back to the main dataframes for unified processing
    train_df_full_features = pd.concat([train_df, train_features], axis=1)
    test_df_full_features = pd.concat([test_df, test_features], axis=1)
    print("Feature generation complete.")

    # Clean up memory
    del train_features, test_features
    gc.collect()

    print("\nStarting TabTransformer training and prediction...")
    oof_preds_df, test_preds_df,train_df_full = train_and_predict_tabtransformer(train_df_full_features, test_df_full_features)
    print("TabTransformer training and prediction complete.")

    # Calculate WMAE Score (Out-of-Fold)
    print(f"\n--- Overall WMAE Score (Out-Of-Fold) ---")

    tr_solution = train_df_full[['id'] + CFG.TARGETS]
    tr_submission = train_df_full[['id'] + [t + '_pred' for t in CFG.TARGETS]]
    tr_submission.columns = ['id'] + CFG.TARGETS
    print("*"*25 +" FINAL SCORE WAME "+"*"*25)
    print(f"wMAE: {round(wmae_score(tr_solution, tr_submission, row_id_column_name='id'), 5)}")
    print("*"*50)

    # Handle Overlapping SMILES between Train and Test for Submission
    # print("\nHandling overlapping SMILES for final submission...")
    # for target in CFG.TARGETS:
    #     train_smiles_known = train_df[train_df[target].notnull()][['SMILES', target]].drop_duplicates(subset=['SMILES']).copy()

    #     merged_test = pd.merge(test_preds_df[['id', 'SMILES', target]], train_smiles_known, on='SMILES', how='left', suffixes=('_pred', '_known'))

    #     test_preds_df[target] = merged_test[f'{target}_known'].fillna(merged_test[f'{target}_pred'])

    # print("Overlapping SMILES handled.")

    # Create Submission File
    final_submission_df = test_preds_df[['id'] + CFG.TARGETS].copy()
    final_submission_df.to_csv('submission3.csv', index=False)
    print(f"\nSubmission file 'submission.csv' created successfully. 🎉")
    print("Submission Head:")
    print(final_submission_df.head())

    # Final cleanup
    del train_df, test_df, oof_preds_df, test_preds_df, data_tc, data_tg2, data_tg3, data_dnst
    del train_df_full_features, test_df_full_features
    gc.collect()

# %%
# 读取两个模型的预测结果
submission_xgb = pd.read_csv('submission1.csv')
submission_bert = pd.read_csv('submission2.csv')
submission_trans = pd.read_csv('submission3.csv')

# 检查ID顺序和列名是否一致
# assert (submission_xgb['id'] == submission_bert['id']).all(), "ID顺序不一致！"

# 你可以设置每个模型的权重，比如都设0.5就是简单平均
w_xgb = [0.55, 0.65, 0.55, 0.55, 0.55]
w_bert = [0.325, 0.3, 0.3, 0.325, 0.325]
w_trans = [0.125, 0.05, 0.15, 0.125, 0.125]

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# 融合
submission_fused = submission_xgb.copy()
for i in range(0,5):
    submission_fused[targets[i]] = w_xgb[i] * submission_xgb[targets[i]] + w_bert[i] * submission_bert[targets[i]] +w_trans[i] * submission_trans[targets[i]]
submission_fused['Tg'] += 40.00
# 输出融合后的结果
submission_fused.to_csv('submission.csv', index=False)
submission_fused.head()

# %%



