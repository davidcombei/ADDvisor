import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
#from ADDvisor import audioprocessor
from accelerate import Accelerator

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


accelerator = Accelerator()
device = accelerator.device


from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg, TorchScaler
audio_processor = AudioProcessor()
torch_logreg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)


class LMACLoss():
    def __init__(self, l_in_w=4, l_out_w=0.5, reg_w_l1=0.2, reg_w_tv=0.01):

        self.l_in_w = l_in_w #default =4.0
        self.l_out_w = l_out_w #default 0.2
        self.reg_w_l1 = reg_w_l1 #default 0.4
        self.reg_w_tv = reg_w_tv #default 0.00


    def loss_function(self, xhat, X_stft_power,X_stft_phase, class_pred,):


        Tmax = xhat.shape[1]
        power = 2
        relevant_mask_mag = xhat * X_stft_power[:, :Tmax, :]  # the relevant parts of the spectrogram
        irelevant_mask_mag = (1 - xhat) * X_stft_power[:, :Tmax, :] # the irelevant parts of the spectrogram
        relevant_mask = relevant_mask_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        irelevant_mask = irelevant_mask_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        # relevant_mask = relevant_mask.squeeze(0)
        # irelevant_mask = irelevant_mask.squeeze(0)
        istft_relevant_mask = audio_processor.compute_invert_stft(relevant_mask)
        istft_irelevant_mask = audio_processor.compute_invert_stft(irelevant_mask)
        # istft_relevant_mask = istft_relevant_mask.squeeze(0)
        # istft_irelevant_mask = istft_irelevant_mask.squeeze(0)
        relevant_mask_waveform = audio_processor.extract_features_istft(istft_relevant_mask)
        irelevant_mask_waveform = audio_processor.extract_features_istft(istft_irelevant_mask)
        relevant_mask_feats = torch.mean(relevant_mask_waveform.squeeze(0), dim=1)
        irelevant_mask_feats = torch.mean(irelevant_mask_waveform.squeeze(0), dim=1)
        relevant_mask_logits, relevant_mask_probs = torch_logreg(relevant_mask_feats)
        irelevant_mask_logits, irelevant_mask_probs = torch_logreg(irelevant_mask_feats)
        #print(relevant_mask_logits.shape, irelevant_mask_probs.shape)
        relevant_mask_logits = torch_scaler(relevant_mask_logits)
        irelevant_mask_logits = torch_scaler(irelevant_mask_logits)


        # issues might arrise here due to logits scaled applied to the cross entropy that applies sigmoid
        l_in = F.binary_cross_entropy_with_logits(relevant_mask_logits, class_pred)#.to(device))
        l_out = -F.binary_cross_entropy_with_logits(irelevant_mask_logits, class_pred)#.to(device))
        ao_loss = self.l_in_w * l_in + self.l_out_w * l_out


#        reg_l1 = xhat.abs().mean((-1, -2)).mean() * self.reg_w_l1
        reg_l1 = xhat.abs().mean((-1, -2, -3)) * self.reg_w_l1
        #tv_h = torch.mean(torch.abs(xhat[:, :, :-1] - xhat[:, :, 1:]))  # horizontal differences
        #tv_w = torch.mean(torch.abs(xhat[:, :-1, :] - xhat[:, 1:, :]))  # vertical differences
        tv_h = torch.sum(torch.pow(xhat[:, :, :-1] - xhat[:, :, 1:], power))  # horizontal differences
        tv_w = torch.sum(torch.pow(xhat[:, :-1, :] - xhat[:, 1:, :], power))  # vertical differences
        reg_tv = self.reg_w_tv * (tv_h + tv_w) / float(power* xhat.size(0)) 
        reg_loss = reg_l1 + reg_tv

        total_loss = ao_loss + reg_loss
        return total_loss
