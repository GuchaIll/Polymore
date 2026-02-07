import math
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, GraphDescriptors, Crippen

# --- KNOWLEDGE BASE ---
RECYCLABLE_SMARTS = {
    "ester": "[CX3](=[OX1])[OX2]",        # Hydrolyzable
    "amide": "[CX3](=[OX1])[NX3]",        # Hydrolyzable (slower)
    "carbonate": "[OX2][CX3](=[OX1])[OX2]", # Hydrolyzable
    "disulfide": "[SX2][SX2]",            # Chemically reversible
    "imine": "[CX3]=[NX2]",               # Dynamic covalent
    "acetal": "[CX4]([OX2])([OX2])"       # Acid sensitive
}

TOXIC_ALERTS = {
    "pfas": "[#6](F)(F)[#6](F)(F)",       # Persistent
    "isocyanate": "N=C=O",                # Reactive lung irritant
    "heavy_metal": "[Hg,Pb,Cd,As]",       # Heavy metals
    "acryl_nitrile": "C=CC#N",            # Toxic monomer
    "epoxide": "[CX3]1[OX2][CX3]1"        # Mutagenic
}

# Pre-compile SMARTS patterns for efficiency
RECYCLABLE_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in RECYCLABLE_SMARTS.items()}
TOXIC_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in TOXIC_ALERTS.items()}

def predict_properties(smiles: str) -> dict:
    """
    Tier 1 Analysis: Fast, rule-based profiling using RDKit.
    
    Returns 0-10 scores based on chemical density metrics:
    - Higher strength = more rigid/cohesive material
    - Higher flexibility = more rotatable bonds (softer/flexible)
    - Higher degradability = easier to recycle/break down
    - Higher sustainability = bio-sourceable, non-toxic, synthetically accessible
    
    Also returns metadata with detailed breakdown for explainability.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    # Basic Descriptors
    mw = Descriptors.MolWt(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0: 
        return {
            "strength": 0.0, "flexibility": 0.0, 
            "degradability": 0.0, "sustainability": 0.0, 
            "sas_score": 0.0
        }

    # --- 1. FLEXIBILITY (Rotatable Bond Density) ---
    # Logic: High density of rotatable bonds = Flexible/Soft.
    n_rot = Descriptors.NumRotatableBonds(mol)
    # Normalize: 0.0 (Rigid) to 0.5+ (Very Flexible)
    rot_density = n_rot / n_heavy
    flexibility = min(rot_density * 20.0, 10.0)

    # --- 2. STRENGTH (Rigidity & Cohesion) ---
    # Logic: Aromatic rings (stiffness) + TPSA (Polarity/H-bonding for cohesion)
    n_aromatic = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
    aromatic_fraction = n_aromatic / n_heavy
    
    # Cohesion proxy (TPSA normalized)
    tpsa = Descriptors.TPSA(mol)
    cohesion_score = min(tpsa / 120.0, 1.0) # Cap at 1.0 (typical high polarity)
    
    # Strength = 70% Rigidity + 30% Cohesion
    raw_strength = (aromatic_fraction * 7.0) + (cohesion_score * 3.0)
    strength = min(raw_strength * 2.0, 10.0)

    # --- 3. DEGRADABILITY (Recyclability Potential) ---
    # Logic: Count "unzippable" bonds per unit of Molecular Weight.
    total_recyclable = 0
    breakdown_details = {}
    for name, pattern in RECYCLABLE_PATTERNS.items():
        if pattern:
            count = len(mol.GetSubstructMatches(pattern))
            if count > 0:
                total_recyclable += count
                breakdown_details[name] = count
    
    # Heuristic: 1 breakable bond per 150 MW is "Good" (Score ~5-6)
    density = (total_recyclable * 150.0) / (mw + 1.0)
    degradability = min(density * 5.0, 10.0)

    # --- 4. SUSTAINABILITY (Eco-Composite) ---
    # Logic: Bio-sourcing potential + Safety (Lack of toxicity) + Efficiency
    
    # A. Bio-Source Proxy (Fraction sp3 + Chirality)
    f_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    bio_score = (f_sp3 * 0.8) + (min(n_chiral, 3) * 0.1)
    
    # B. Toxicity Penalty
    alert_penalty = 0
    found_alerts = []
    for name, pattern in TOXIC_PATTERNS.items():
        if pattern and mol.HasSubstructMatch(pattern):
            alert_penalty += 4.0 # Heavy penalty per alert type
            found_alerts.append(name)
            
    # C. SAS Score Proxy (Complexity)
    # Using BertzCT as a proxy for Synthetic Accessibility
    try:
        bertz = GraphDescriptors.BertzCT(mol)
        # Bertz > 800 is hard. Bertz < 200 is easy.
        # Map Bertz to 1-10 (1=Easy, 10=Hard)
        sas_proxy = min(max(1.0, bertz / 100.0), 10.0)
    except:
        sas_proxy = 5.0  # Neutral default if calculation fails
    
    # Composite Sustainability Score
    # Start at 5.0, add Bio, subtract Tox and Complexity
    raw_sust = (5.0 + (bio_score * 4.0) - alert_penalty - (sas_proxy * 0.2))
    sustainability = max(0.0, min(raw_sust, 10.0))

    return {
        "strength": round(strength, 2),
        "flexibility": round(flexibility, 2),
        "degradability": round(degradability, 2),
        "sustainability": round(sustainability, 2),
        "sas_score": round(sas_proxy, 2),
        "meta": {
            "aromatic_fraction": round(aromatic_fraction, 2),
            "recyclable_groups": breakdown_details,
            "toxic_alerts": found_alerts,
            "bio_carbon_fraction": round(f_sp3, 2),
            "molecular_weight": round(mw, 2)
        }
    }
