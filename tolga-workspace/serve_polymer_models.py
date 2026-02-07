import modal

from pathlib import Path

# ============================================================================
# 1. Image & Environment
# ============================================================================
image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "rdkit",
        "pandas",
        "numpy",
        "scikit-learn",
        channels=["conda-forge"]
    )
    .pip_install(
        "torch", 
        "transformers", 
        "xgboost", 
        "joblib",
        "fastapi[standard]"
    )
)

app = modal.App("polymer-serving")

# Volumes
MODELS_PATH = Path("/models")
DATA_PATH = Path("/data")
models_volume = modal.Volume.from_name("polymer-models")
data_volume = modal.Volume.from_name("polymer-data")

# ============================================================================
# 2. Serving Class
# ============================================================================

@app.cls(
    image=image,
    gpu="any", # Optional: can use CPU for cheaper inference
    volumes={MODELS_PATH: models_volume, DATA_PATH: data_volume},
    scaledown_window=300, # Keep warm for 5 minutes
)
class PolymerPredictor:
    @modal.enter()
    def load(self):
        import joblib
        import xgboost as xgb
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        
        print("🔧 Loading models...")
        
        # --- Model Classes Definition (Inside to avoid global imports) ---
        
        class ContextPooler(nn.Module):
            def __init__(self, config):
                super().__init__()
                pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
                self.dense = nn.Linear(pooler_size, pooler_size)
                dropout_prob = getattr(config, 'pooler_dropout', getattr(config, 'hidden_dropout_prob', 0.1))
                self.dropout = nn.Dropout(dropout_prob)
                act_fn_name = getattr(config, 'pooler_hidden_act', getattr(config, 'hidden_act', 'tanh'))
                self.activation = nn.Tanh() if act_fn_name == 'tanh' else nn.ReLU() 

            def forward(self, hidden_states):
                context_token = hidden_states[:, 0]
                context_token = self.dropout(context_token)
                pooled_output = self.dense(context_token)
                return self.activation(pooled_output)

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
        
        class SustainabilityTransferModel(nn.Module):
            def __init__(self, config_path, num_targets=1):
                super().__init__()
                config = AutoConfig.from_pretrained(config_path)
                self.backbone = AutoModel.from_config(config)
                pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
                self.pooler_dense = nn.Linear(pooler_size, pooler_size)
                self.pooler_activation = nn.Tanh()
                self.output = nn.Linear(pooler_size, num_targets)

            def forward(self, input_ids, attention_mask):
                outputs = self.backbone(input_ids, attention_mask=attention_mask)
                first_token = outputs.last_hidden_state[:, 0]
                pooled = self.pooler_activation(self.pooler_dense(first_token))
                return self.output(pooled)

        self.BertModelClass = CustomModel
        self.SustModelClass = SustainabilityTransferModel
        
        # 1. Load Original Targets (Tg, FFV, etc.) Configuration
        self.targets_orig = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.models_orig_bert = {}
        
        # BERT Original
        try:
            self.scalers = joblib.load(DATA_PATH / 'smiles-bert-models' / 'target_scalers.pkl')
            self.bert_tokenizer = AutoTokenizer.from_pretrained(str(DATA_PATH / 'smiles-deberta77m-tokenizer'))
            self.bert_config = AutoConfig.from_pretrained(str(DATA_PATH / 'smiles-deberta77m-tokenizer'))
            
            for t in self.targets_orig:
                path = DATA_PATH / 'private-smile-bert-models' / f'warm_smiles_model_{t}_target.pth'
                if path.exists():
                    m = self.BertModelClass(self.bert_config).cuda()
                    m.load_state_dict(torch.load(path, map_location='cuda'))
                    m.eval()
                    self.models_orig_bert[t] = m
        except Exception as e:
            print(f"⚠️ Failed to load BERT original models: {e}")

        # 2. Load Sustainability Models
        self.targets_sust = ["Target_Recyclability", "Target_BioSource", "Target_EnvSafety", "Target_SynthEfficiency"]
        self.models_sust_deberta = {}
        
        # DeBERTa Sustainability
        try:
            for t in self.targets_sust:
                path = MODELS_PATH / f"sustainability_deberta_{t}.pth"
                if path.exists():
                    m = self.SustModelClass(str(DATA_PATH / 'smiles-deberta77m-tokenizer'), num_targets=1).cuda()
                    m.load_state_dict(torch.load(path, map_location='cuda'))
                    m.eval()
                    self.models_sust_deberta[t] = m
        except Exception as e:
            print(f"⚠️ Failed to load DeBERTa sustainability models: {e}")

    @modal.method()
    def predict(self, smiles: str):
        import numpy as np
        import torch
        from rdkit import Chem
        
        results = {"smiles": smiles, "predictions": {}}
        mol = Chem.MolFromSmiles(smiles)
        
        if not mol:
            return {"error": "Invalid SMILES string"}

        # --- 1. Original Targets ---
        try:
            aug_smiles = [smiles] 
            smiles_with_cls = [self.bert_tokenizer.cls_token + s for s in aug_smiles]
            tokenized = self.bert_tokenizer(smiles_with_cls, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()

            for t_idx, t in enumerate(self.targets_orig):
                if t in self.models_orig_bert:
                    with torch.no_grad():
                        scaler = self.scalers[t_idx] 
                        preds = self.models_orig_bert[t](input_ids=input_ids, attention_mask=attention_mask)
                        val = scaler.inverse_transform(preds.cpu().numpy())
                        results["predictions"][t] = float(np.mean(val))
        except Exception as e:
            results["errors"] = results.get("errors", []) + [f"Original BERT failed: {e}"]

        # --- 2. Sustainability Targets ---
        try:
            tokens = self.bert_tokenizer([smiles], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            b_ids = tokens['input_ids'].cuda()
            b_mask = tokens['attention_mask'].cuda()
            
            for t in self.targets_sust:
                if t in self.models_sust_deberta:
                    with torch.no_grad():
                        pred = self.models_sust_deberta[t](b_ids, b_mask).cpu().numpy()[0][0]
                        results["predictions"][t] = float(pred)
        except Exception as e:
             results["errors"] = results.get("errors", []) + [f"Sustainability DeBERTa failed: {e}"]

        return results

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def api(data: dict):
    # Expects {"smiles": "C=CC"}
    smiles = data.get("smiles")
    if not smiles:
        return {"error": "No SMILES provided"}
    
    predictor = PolymerPredictor()
    return predictor.predict.remote(smiles)
