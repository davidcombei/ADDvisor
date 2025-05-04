import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
#from ADDvisor import audioprocessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg, TorchScaler
audio_processor = AudioProcessor()
torch_logreg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)


class LMACLoss():
    def __init__(self, l_in_w=4.0, l_out_w=0.4, reg_w_l1=5, reg_w_tv=0.00):
        self.l_in_w = l_in_w
        self.l_out_w = l_out_w
        self.reg_w_l1 = reg_w_l1
        self.reg_w_tv = reg_w_tv



    def loss_function(self, xhat, X_stft_power,X_stft_phase, class_pred):


        Tmax = xhat.shape[1]
#        print(xhat.shape)
#        print(X_stft_power.shape)
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
        relevant_mask_waveform = audio_processor.extract_features(istft_relevant_mask)
        irelevant_mask_waveform = audio_processor.extract_features(istft_irelevant_mask)
        relevant_mask_feats = torch.mean(relevant_mask_waveform.squeeze(0), dim=1)
        irelevant_mask_feats = torch.mean(irelevant_mask_waveform.squeeze(0), dim=1)

        relevant_mask_logits, relevant_mask_probs = torch_logreg(relevant_mask_feats)
        irelevant_mask_logits, irelevant_mask_probs = torch_logreg(irelevant_mask_feats)
        
        l_in = F.binary_cross_entropy_with_logits(relevant_mask_logits, class_pred)#.to(device))
#        l_out = -F.binary_cross_entropy_with_logits(irelevant_mask_logits, class_pred)
        l_out = F.binary_cross_entropy_with_logits(irelevant_mask_logits, 1 - class_pred)
        ao_loss = self.l_in_w * l_in + self.l_out_w * l_out


        reg_l1 = xhat.abs().mean((-1, -2, -3)) * self.reg_w_l1
        tv_h = torch.sum(torch.abs(xhat[:, :, :-1] - xhat[:, :, 1:]))  # horizontal differences
        tv_w = torch.sum(torch.abs(xhat[:, :-1, :] - xhat[:, 1:, :]))  # vertical differences
        reg_tv = (tv_h + tv_w) * self.reg_w_tv
        reg_loss = reg_l1 + reg_tv

        #mask_binary = (xhat > self.hard_thresh).float()
#        proportion_active = xhat.mean(dim=(1,2)) #mask_binary.mean(dim=(1, 2))  # batch level proportions active ( just mean of the values, since it is working with sigmoid)
#        excess = F.relu(proportion_active - self.sparsity_max_proportion) # relu just in case proportion_active > desired proportion activated ( unlikely but better safe)
#        sparsity_penalty = self.sparsity_weight * excess.mean() #hard penalty on the excess of activated proprotion of the mask

        total_loss = ao_loss + reg_loss# + sparsity_penalty
        return total_loss, l_in, l_out, reg_l1, relevant_mask_logits, irelevant_mask_logits
