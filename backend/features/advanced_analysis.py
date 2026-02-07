import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from tblite.interface import Calculator

logger = logging.getLogger(__name__)

def analyze_molecule_high_compute(smiles: str, progress_callback=None) -> dict:
    """
    Tier 2 Analysis: Physics-based profiling using GFN2-xTB.
    Calculates conformational entropy (Flexibility), chemical hardness (Stability), 
    and electronic polarity (Solubility/Energy).
    """
    if progress_callback: progress_callback(5, "Initializing quantum engine...")

    # 1. PREP: Generate Molecule
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        raise ValueError("Invalid SMILES string")
    mol = Chem.AddHs(mol)
    
    # 2. CONFORMER GENERATION (The "Flexibility" Test)
    if progress_callback: progress_callback(15, "Generating conformer ensemble...")
    
    # Generate up to 5 diverse shapes to test flexibility.
    # We prune RMSD to ensure shapes are actually different.
    num_confs = 5
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.5 
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    
    if not conf_ids:
        # Fallback to single random embedding if ETKDG fails
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        conf_ids = [0]

    atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    
    # Storage for ensemble properties
    energies = []
    gaps = []
    dipoles = []
    
    # 3. QUANTUM CALCULATION LOOP (Batch Processing)
    total_steps = len(conf_ids)
    for i, cid in enumerate(conf_ids):
        step_progress = 30 + int((i / total_steps) * 60)
        if progress_callback: 
            progress_callback(step_progress, f"Running xTB on conformer {i+1}/{total_steps}...")
            
        # Get coordinates in Bohr (xTB expects Bohr)
        pos_angstrom = mol.GetConformer(cid).GetPositions()
        pos_bohr = pos_angstrom * 1.8897259886
        
        # Run GFN2-xTB
        # GFN2 is robust for geometries and non-covalent interactions
        calc = Calculator("GFN2-xTB", atomic_numbers, pos_bohr)
        res = calc.singlepoint()
        
        # Capture Energy (Hartree)
        energies.append(res.get("energy"))
        
        # Capture Dipole (Vector) -> Magnitude
        dipole_vec = res.get("dipole")
        dipoles.append(np.linalg.norm(dipole_vec))
        
        # Capture HOMO-LUMO Gap (Chemical Hardness)
        # tblite returns orbital energies. We find the gap between occ and virt.
        orbital_energies = res.get("orbital_energies")
        occupations = res.get("orbital_occupations")
        
        try:
            # Find the index of the highest occupied orbital (HOMO)
            # Occupation is typically 2.0 or 1.0. We look for the drop-off.
            idx_homo = np.where(occupations > 0.5)[0][-1]
            e_homo = orbital_energies[idx_homo]
            e_lumo = orbital_energies[idx_homo + 1]
            gap_ev = (e_lumo - e_homo) * 27.2114 # Convert Hartree to eV
            gaps.append(gap_ev)
        except Exception as e:
            # Fallback for open-shell or weird convergence
            gaps.append(0.0)

    # 4. ANALYSIS & AGGREGATION
    if progress_callback: progress_callback(95, "Aggregating physics metrics...")
    
    # A. FLEXIBILITY (Boltzmann Proxy)
    # High energy standard deviation = Molecule has accessible "high energy" twisted states = Flexible
    # Low energy standard deviation = Molecule snaps to one global minimum = Rigid
    energy_std = np.std(energies)
    # Mapping: std dev 0.0 -> Score 0 (Rigid), std dev > 0.05 -> Score 10 (Flexible)
    # This is a robust physical proxy for Tg (Glass Transition)
    flexibility_score = min(10.0, energy_std * 200.0)

    # B. STRENGTH (Dipole Alignment)
    # Average dipole moment indicates intermolecular forces
    avg_dipole = np.mean(dipoles)
    # Dipole 0 -> Score 0, Dipole 5+ -> Score 10
    strength_score = min(10.0, avg_dipole * 2.0)

    # C. REACTIVITY (Chemical Hardness)
    # Small Gap = Soft/Reactive (Easier to degrade/oxidize, but potential toxicity)
    # Large Gap = Hard/Inert (Persistent, safe, but hard to recycle chemically)
    avg_gap = np.mean(gaps)
    
    # degradability_score: Soft molecules (small gap) degrade easier
    # Gap 1eV -> Score 10 (High Degradability), Gap 8eV -> Score 0
    degradability_score = max(0.0, 10.0 - (avg_gap - 1.0) * 1.5)
    
    # sustainability_score: In Tier 2, this represents "Chemical Stability"
    # Validates the "Safety" aspect. High Gap = Stable/Safe.
    stability_score = min(10.0, avg_gap)

    return {
        "strength": round(strength_score, 2),
        "flexibility": round(flexibility_score, 2), 
        "degradability": round(degradability_score, 2),
        "sustainability": round(stability_score, 2), # Represents Chemical Stability
        "meta": {
            "method": "GFN2-xTB (Ensemble)",
            "conformers_scanned": len(conf_ids),
            "avg_dipole_debye": round(avg_dipole, 3),
            "avg_homo_lumo_gap_ev": round(avg_gap, 3),
            "conformational_entropy_proxy": round(energy_std, 5)
        }
    }
