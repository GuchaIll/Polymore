from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors
from rdkit.Chem import QED

# Define SMARTS patterns for Recyclability (Chemically Recyclable linkages)
# These bonds can often be depolymerized via hydrolysis or glycolysis
recyclable_motifs = {
    "ester_linkage": "C(=O)O",
    "carbonate_linkage": "OC(=O)O",
    "amide_linkage": "C(=O)N",
    "carbamate_linkage": "OC(=O)N",
    "disulfide_linkage": "SS"
}

# Define SMARTS for Toxicity/Persistence (Substructures of Concern)
toxic_motifs = {
    "perfluorinated": "C(F)(F)C(F)(F)",  # PFAS-like chains (persistence)
    "isocyanate": "N=C=O",               # Reactive/Toxic
    "heavy_metal": "[Hg,Pb,Cd,As,Cr]",   # Heavy metals
    "halogenated_aromatic": "c[Cl,Br,I]" # Often persistent/toxic
}

def get_sustainability_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    features = {}

    # --- 1. Synthetic Accessibility Score (SAS) ---
    # Lower SAS (1) = Easy to make; High SAS (10) = Difficult/Energy intensive
    # Note: Requires rdkit.Contrib.SA_Score or an external implementation. 
    # If unavailable, BertzCT is a good built-in proxy for complexity.
    features['bertz_complexity'] = GraphDescriptors.BertzCT(mol)
    
    # --- 2. Recyclability Proxies ---
    # Calculate the density of "breakable" bonds per Molecular Weight
    total_recyclable_bonds = 0
    for name, smarts in recyclable_motifs.items():
        pattern = Chem.MolFromSmarts(smarts)
        count = len(mol.GetSubstructMatches(pattern))
        features[f'count_{name}'] = count
        total_recyclable_bonds += count
    
    # "Breakability Index": Higher = easier to chemically recycle
    features['recyclability_index'] = total_recyclable_bonds / (Descriptors.MolWt(mol) + 1e-6)

    # --- 3. Environmental Toxicity & Persistence ---
    # LogP: High LogP (>5) often indicates bioaccumulation potential
    features['logP_bioaccum'] = Descriptors.MolLogP(mol)
    
    # Topological Polar Surface Area (TPSA): 
    # Very low TPSA often correlates with high cell permeability (toxicity risk)
    features['tpsa_permeability'] = Descriptors.TPSA(mol)

    total_toxic_hits = 0
    for name, smarts in toxic_motifs.items():
        pattern = Chem.MolFromSmarts(smarts)
        count = len(mol.GetSubstructMatches(pattern))
        features[f'alert_{name}'] = count
        total_toxic_hits += count
        
    features['toxicity_alert_count'] = total_toxic_hits

    # --- 4. Bio-Sourcing Potential ---
    # Fraction of sp3 Carbon: Natural products tend to have higher Fsp3 than synthetic petrochemicals
    features['fraction_sp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
    
    # Number of Chiral Centers: Bio-sourced molecules often have high chirality
    features['chiral_centers'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    # --- 5. Energy Efficiency (Synthesis Proxy) ---
    # Number of Rings + Rotatable Bonds often correlates with synthesis entropy/energy
    features['synth_energy_proxy'] = rdMolDescriptors.CalcNumRings(mol) + rdMolDescriptors.CalcNumRotatableBonds(mol)

    return features