import { MoleculeLibrary } from '../types';
//TODO: For now these are placeholder molecules, drop in researched monomers, and functional groups here.
//TODO: Add a search function to the library, and a way to add custom molecules.
//TODO: Store chemical composition structure for displaying group info on mouse hover

//TODO: Add common every day items corresponding to these molecular groups
export const moleculeLibrary: MoleculeLibrary = {
  basic: [
    { name: 'Carbon', formula: 'C', smiles: 'C', icon: '⚫', color: '#909090', weight: 12 },
    { name: 'Oxygen', formula: 'O', smiles: 'O',  icon: '🔴', color: '#FF0D0D', weight: 16 },
    { name: 'Nitrogen', formula: 'N', smiles: 'N',  icon: '🔵', color: '#3050F8', weight: 14 },
    { name: 'Hydrogen', formula: 'H', smiles: '[H]', icon: '⚪', color: '#FFFFFF', weight: 1 },
  ],
  functional: [
    { name: 'Hydroxyl', formula: '-OH', smiles: 'O',  icon: '💧', color: '#FF6B6B', weight: 17 },
    { name: 'Carboxyl', formula: '-COOH', smiles: 'C(=O)O',  icon: '🧪', color: '#4ECDC4', weight: 45 },
    { name: 'Amino', formula: '-NH2', smiles: 'N',  icon: '💜', color: '#9B59B6', weight: 16 },
    { name: 'Methyl', formula: '-CH3', smiles: 'C', icon: '🟤', color: '#95A5A6', weight: 15 },
  ],
  monomers: [
    { name: 'Ethylene', formula: 'C2H4', smiles: 'C=C',  icon: '🔷', color: '#667EEA', weight: 28 },
    { name: 'Propylene', formula: 'C3H6', smiles: 'CC=C', icon: '🔶', color: '#F39C12', weight: 42 },
    { name: 'Styrene', formula: 'C8H8', smiles: 'c1ccccc1C=C',  icon: '💎', color: '#E74C3C', weight: 104 },
    { name: 'Vinyl Chloride', formula: 'C2H3Cl', smiles: 'C=CCl',  icon: '🟢', color: '#27AE60', weight: 62.5 },
    { name: 'Acrylate', formula: 'C3H4O2', smiles: 'C=CC(=O)O', icon: '🟡', color: '#F1C40F', weight: 72 },
    { name: 'Caprolactam', formula: 'C6H11NO', smiles: 'O=C1CCCCCN1', icon: '🟣', color: '#8E44AD', weight: 113 },
  ]
};