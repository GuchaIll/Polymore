
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

def predict_properties(smiles: str) -> dict:
    """
    Predicts material properties (strength, flexibility, degradability, sustainability)
    based on SMILES string using RDKit heuristics.
    
    Returns a dictionary with scores normalized to 0-10.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    # 1. Strength (Proxy: Molecular Weight & Complexity)
    # Heuristic: Heavier, complex molecules with rings tend to be stronger/more rigid.
    mw = Descriptors.MolWt(mol)
    ring_count = Lipinski.RingCount(mol)
    # Normalize MW: 0-500 -> 0-10 roughly
    strength = min((mw / 50) + (ring_count * 2), 10)

    # 2. Flexibility (Proxy: Rotatable Bonds)
    # Heuristic: More rotatable bonds -> more flexible.
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    # Normalize: 0-20 -> 0-10
    flexibility = min(rotatable_bonds / 2, 10)

    # 3. Degradability (Proxy: Hydrolyzable groups)
    # Heuristic: Presence of esters, amides, etc. increases degradability.
    # Simple SMARTS for Ester and Amide
    ester_pattern = Chem.MolFromSmarts("C(=O)O") 
    amide_pattern = Chem.MolFromSmarts("C(=O)N")
    
    ester_count = len(mol.GetSubstructMatches(ester_pattern)) if ester_pattern else 0
    amide_count = len(mol.GetSubstructMatches(amide_pattern)) if amide_pattern else 0
    
    # Normalize: Each group adds points
    degradability = min((ester_count * 3) + (amide_count * 2), 10)

    # 4. Sustainability (Proxy: Atom Efficiency / Complexity)
    # Heuristic: Smaller, simpler molecules are often more sustainable to produce (atom economy).
    # This is a very rough proxy. Higher score = More sustainable.
    # Inverse of complexity/weight.
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms > 0:
        sustainability = max(10 - (heavy_atoms / 5), 0)
    else:
        sustainability = 0

    return {
        "strength": round(strength, 2),
        "flexibility": round(flexibility, 2),
        "degradability": round(degradability, 2),
        "sustainability": round(sustainability, 2)
    }
