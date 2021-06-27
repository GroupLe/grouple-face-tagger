import torch
from .models.naive import BertLstm

def load_names_model(weights_path) -> BertLstm:
    model = BertLstm(2)
    model.lin.load_state_dict(torch.load(weights_path))
    model.eval()
    return model
