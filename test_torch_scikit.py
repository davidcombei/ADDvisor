# import torch
# import librosa
# import librosa.display
# import numpy as np
# from transformers import Wav2Vec2Model, AutoFeatureExtractor
# import webrtcvad
# import joblib
# from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
# import os
# from classifier_embedder import TorchLogReg, TorchScaler, zero_mean_unit_var_norm
# import torch.optim as optim
# from tqdm import tqdm
# from scipy.optimize import brentq
# from scipy.interpolate import interp1d
# from sklearn.metrics import roc_curve
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn as nn
# import datetime
# from torchmetrics import Precision, Recall, ConfusionMatrix, Accuracy
# import random
# import scipy.signal as signal
# from pydub import AudioSegment
# from pydub.utils import which
# import torchaudio
# import torchaudio.transforms as T
# import torch.nn.functional as F
#
# MAX_SAMPLES = 1000
#
# metadata_in = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\meta.csv'
# #metadata_in = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\meta.csv'
# basedir = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav'
# #basedir = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav'
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# classifier,scaler, thresh = joblib.load(r'C:\Users\david\PycharmProjects\David2\model\logreg_margin_pruning_ALL_with_scaler_threshold.joblib')
#
# processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")
# wav2vec2 = Wav2Vec2Model.from_pretrained(r"C:\Users\david\PycharmProjects\David2\model\wav2vec2-xls-r-2b_truncated").to(device)
# wav2vec2.eval()
#
#
# def load_audio(audio_path, target_sr=16000):
#     audio, sr = torchaudio.load(audio_path)
#     if sr != target_sr:
#         resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
#         audio = resampler(audio)
#     return audio.squeeze(0), target_sr
#
#
# ###########################
# ####### EXTRACT FEATURES
# ###########################
#
# def extract_features(audio_path):
#     audio, sr = load_audio(audio_path)
#
#     length = int(5 * sr)
#     current_length = audio.shape[0]
#     if current_length < length:
#         audio = F.pad(audio, (0, length - current_length))
#     else:
#         audio = audio[:length]
#
#     # audio = zero_mean_unit_var_norm(audio)
#     #
#     # input_values = audio.unsqueeze(0).to(device)
#     input_values = processor(audio, return_tensors="pt", sampling_rate=16000,padding=True)#, normalize=True)
#     input_values = input_values["input_values"].to(device)
#     with torch.no_grad():
#         output = wav2vec2(input_values, output_hidden_states=True)
#
#     return output.hidden_states[9]#.squeeze(0)
#
# def run_logReg(audio):
#     features = extract_features(audio)
#     eer_threshold = thresh - 5e-3
#     features_avg = torch.mean(features, dim=1).cpu().numpy()
#     features_avg = features_avg.reshape(1, -1)
#     decision_score = classifier.decision_function(features_avg)
#     decision_score_scaled = scaler.transform(decision_score.reshape(-1, 1)).flatten()
#     return decision_score_scaled[0]
#
#
# def quick_eval_logreg(metadata, basedir):
#     y_true = []
#     y_scores = []
#
#     with open(metadata, 'r') as f:
#         for i, line in tqdm(enumerate(f), total=MAX_SAMPLES, desc="samples", leave=True):
#             if i > MAX_SAMPLES:
#                 break
#             parts = line.strip().split(',')
#             if len(parts) < 2:
#                 continue
#             audio = parts[0]
#             label = parts[2]
#             audio_path = os.path.join(basedir, audio)
#             score = run_logReg(audio_path)
#             if score is None:
#                 continue
#             y_scores.append(score)
#             y_true.append(0 if label == 'spoof' else 1)
#
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
#     eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#     y_pred = np.array([1 if p > thresh - 5e-3 else 0 for p in y_scores])
#     accuracy = accuracy_score(y_true, y_pred)
#     cm = confusion_matrix(y_true, y_pred)
#
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     print(f"EER: {eer * 100:.2f}%")
#     print(f"Confusion Matrix:")
#     print(cm)
#
#     return accuracy, eer
#
#
#
# quick_eval_logreg(metadata_in, basedir)


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################


import torch
import numpy as np
import os
import joblib
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from classifier_embedder import TorchLogReg, TorchScaler, zero_mean_unit_var_norm, scaler
from transformers import Wav2Vec2Model,AutoFeatureExtractor


MAX_SAMPLES = 1000
metadata_in = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\meta.csv'
basedir = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = TorchLogReg().to(device)
scaler_torch = TorchScaler().to(device)

_ , _, thresh = joblib.load(r'C:\Users\david\PycharmProjects\David2\model\logreg_margin_pruning_ALL_with_scaler_threshold.joblib')
wav2vec2 = Wav2Vec2Model.from_pretrained(r"C:\Users\david\PycharmProjects\David2\model\wav2vec2-xls-r-2b_truncated").to(device)
wav2vec2.eval()
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")


def load_audio(audio_path, target_sr=16000):
    audio, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
    return audio.squeeze(0), target_sr



def extract_features_torch(audio_path):
    audio, sr = load_audio(audio_path)

    length = int(5 * sr)
    current_length = audio.shape[0]
    if current_length < length:
        audio = F.pad(audio, (0, length - current_length))
    else:
        audio = audio[:length]

    audio = zero_mean_unit_var_norm(audio)
    input_values = audio.unsqueeze(0).to(device)
    output = wav2vec2(input_values, output_hidden_states=True)
    #print(output.hidden_states[9].shape)
    return output.hidden_states[9]





def run_logReg(audio_path):
    #features = extract_features(audio_path)
    features = extract_features_torch(audio_path)
    features = torch.mean(features.squeeze(0), dim=0)
    #print(features.shape)
    with torch.no_grad():
        logits, probs = classifier(features)
    #probs = scaler_torch(probs)


    return logits, probs


def quick_eval_logreg(metadata, basedir):
    y_true = []
    y_scores = []

    with open(metadata, 'r') as f:
        for i, line in tqdm(enumerate(f), total=MAX_SAMPLES, desc="samples", leave=True):
            if i >= MAX_SAMPLES:
                break
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            audio = parts[0]
            label = parts[2]
            audio_path = os.path.join(basedir, audio)

            logits, probs = run_logReg(audio_path)
            print("Logits Before Scaling:", logits.cpu().detach().numpy())

            scaled_output_torch = scaler_torch(logits)
            scaled_output_sklearn = scaler.transform(logits.cpu().detach().numpy().reshape(-1, 1)).flatten()

            print("Scaled Output (TorchScaler):", scaled_output_torch.cpu().detach().numpy())
            print("Scaled Output (Scikit-Learn):", scaled_output_sklearn)



            #score = run_logReg(audio_path)
            #print(score.item())
            # if score is None:
            #     continue
            # y_scores.append(score)
            y_true.append(0 if label == 'spoof' else 1)

    y_scores = np.array([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in y_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    y_pred = np.array([1 if p > thresh - 5e-3 else 0 for p in y_scores])
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"EER: {eer * 100:.2f}%")
    print(f"Confusion Matrix:")
    print(cm)
    return accuracy, eer


quick_eval_logreg(metadata_in, basedir)



