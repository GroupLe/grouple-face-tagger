import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import sklearn

BERT = 'DeepPavlov/rubert-base-cased-conversational'
    
    
class TransferLearningModel(nn.Module):
    """Model requires self.lin trainable head"""
    def __init__(self):
        super().__init__()
        
    def save(self, path):
        torch.save(self.lin.state_dict(), path)
        
    def load(self, path):
        self.lin.load_state_dict(torch.load(path))
        
        
    
class BertLstm(TransferLearningModel):
    """NER model with BERT and lstm head"""
    def __init__(self, n_labels):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained(BERT)
        self.bert = BertModel.from_pretrained(BERT)
        
        def block(dim_in, dim_out):
            return nn.Sequential(nn.Linear(dim_in, dim_out),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0.2))
        
        self.lin = nn.Sequential(block(768, 256),
                                 block(256, 128),
                                 block(128, 64),
                                 block(64, n_labels))
        self.device_param = nn.Parameter(torch.empty(0))
    
    def forward(self, x):
        assert isinstance(x, torch.Tensor), 'string'
        with torch.no_grad():
            word_embs = self.bert(x)['last_hidden_state']
        n_words = word_embs.size(1)
        word_embs = word_embs.view(n_words, -1)
        pred = self.lin(word_embs)
        return pred
    
    def get_names(self, sent: str) -> list:
        """Takes sentence, returns list of names"""
        # preprocess
        device = self.device_param.device
        t = torch.LongTensor(self.tokenizer.encode(sent)).unsqueeze(0).to(device)
        
        # predict
        with torch.no_grad():
            word_embs, sent_emb = self.bert(t)
            preds = self.lin(word_embs).argmax(dim=2).tolist()[0]
    
        # postprocess
        wps, labels = self._postprocess(self.tokenizer.tokenize(sent), preds)
        names = [wps[i] if '1' in labels[i] else None for i in range(len(wps))]
        names = list(filter(lambda x: x is not None, names))
        
        return names

    def _postprocess(self, wps: list, labels: list):
        """Takes list of BERT wordpieces and list of labels for every wordpeace. Returns list of detected names"""
        labels = labels[1:] + [0]

        s = ''
        s_labels = []

        for i, (wp, label) in enumerate(zip(wps, labels)):

            if wp.startswith('##'):
                wp = wp[2:]
                s_labels[-1] = s_labels[-1] + str(label)
            else:
                s_labels.append(str(label))
                s += ' '
            s += wp
        
        return s.split(), s_labels
