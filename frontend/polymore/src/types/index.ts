//Molecule represents both singular molecule and functional groups
export interface Molecule {
    name: string;
    formula: string;
    smiles: string;
    icon: string;
    color: string;
    weight: number;
}

export interface PlacedMolecule extends Molecule {
    id: number;
    position: Position;
    connections: number[]; // IDs of connected molecules
}

export interface Position {
    x: number;
    y: number;
    z: number;
}


export interface HeuristicPredictedProperties {

  strength: number;
  flexibility: number;
  degradability: number;
  sustainability: number;
}

export interface MLPredictedProperties {
  
    Tg :number; // Glass transition temperature 
    FFV : number; //Fractional free volume.
    Tc: number;//Thermal conductivity 
    Density: number;//Polymer density 
    Rg : number;// Radius of gyration 

}

export type PredictedProperties = HeuristicPredictedProperties & Partial<MLPredictedProperties>;

export type ToolType = 'select' | 'add' | 'remove' | 'connect' | 'move';
export type ViewMode = 'both' | 'structure' | 'volume';

    export interface PolyForgeState {
        tool: ToolType;
        gridSnap: boolean;
        gridSize: number;
        placedMolecules: PlacedMolecule[];
        selectedMolecule: Molecule | null;
        selectedObject: number | null; // ID of selected molecule or connection
        connectStart: number | null; // ID of first molecule in connection
        history: PlacedMolecule[][];
        historyIndex: number;
        viewMode: ViewMode;
    }

    export interface Toast {
        message: string;
        visible: boolean;
    }

    export interface MoleculeLibrary {
        basic: Molecule[];
        functional: Molecule[];
        monomers: Molecule[];
    }

