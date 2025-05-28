from pathlib import Path
from typing import List

import pandas as pd
import torch



def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    device="cuda",
):
    df = pd.read_parquet("/data/rdkit_embedding/rdkit2D_embedding_sciplex_lincs.parquet")

    if df is not None:

        emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)
    else:
        assert embedding_model == "zeros"
        emb = torch.zeros((len(smiles), 256))
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
