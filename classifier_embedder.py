import joblib
from transformers import Wav2Vec2Model, AutoFeatureExtractor
import torch.nn as nn
import torch

#####################
### LOAD WAV2VEC + LogReg
#####################
classifier,scaler, thresh = joblib.load(r'C:\Users\david\PycharmProjects\David2\model\logreg_margin_pruning_ALL_with_scaler_threshold.joblib')
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")
wav2vec2 = Wav2Vec2Model.from_pretrained(r"C:\Users\david\PycharmProjects\David2\model\wav2vec2-xls-r-2b_truncated")



class TorchLogReg(nn.Module):
    def __init__(self,coef,intercept):
        super(TorchLogReg,self).__init__()

        

        if coef.dim() == 1:
            coef = coef.unsqueeze(0)

        n_features = coef.size(1)
        self.linear = nn.Linear(n_features, 1)
        self.linear.weight = nn.Parameter(coef.clone(), requires_grad=False)
        self.linear.bias = nn.Parameter(intercept.clone(), requires_grad=False)


    def forward(self,x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)

        return probs


