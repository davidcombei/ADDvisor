import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ADDvisor import audioprocessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg, TorchScaler
audio_processor = AudioProcessor()
torch_logreg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)


class LMACLoss():
    def __init__(self, l_in_w=4.0, l_out_w=0.2, reg_w_l1=0.4, reg_w_tv=0.00):

        self.l_in_w = l_in_w #default =4.0
        self.l_out_w = l_out_w #default 0.2
        self.reg_w_l1 = reg_w_l1 #default 0.4
        self.reg_w_tv = reg_w_tv #default 0.00


    def loss_function(self, xhat, X_stft_power,X_stft_phase, class_pred):


        Tmax = xhat.shape[1]

        relevant_mask_mag = xhat * X_stft_power[:, :Tmax, :]  # the relevant parts of the spectrogram
        irelevant_mask_mag = (1 - xhat) * X_stft_power[:, :Tmax, :] # the irelevant parts of the spectrogram
        relevant_mask = relevant_mask_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        irelevant_mask = irelevant_mask_mag * torch.exp(1j * X_stft_phase[:, :Tmax, :])
        print('relevant_mask grad_fn', relevant_mask.grad_fn) #-- OK
        print('irelevant_mask grad_fn', irelevant_mask.grad_fn) #-- OK
        istft_relevant_mask = audio_processor.compute_invert_stft(relevant_mask)
        istft_irelevant_mask = audio_processor.compute_invert_stft(irelevant_mask)
        print('istft_relevant_mask grad_fn', istft_relevant_mask.grad_fn) #-- OK
        print('istft_irelevant_mask grad_fn', istft_irelevant_mask.grad_fn) #-- OK
        istft_relevant_mask = istft_relevant_mask.squeeze(0)
        istft_irelevant_mask = istft_irelevant_mask.squeeze(0)
        print('istft_relevant_mask grad_fn :', istft_relevant_mask.grad_fn) #-- OK
        print('istft_irelevant_mask grad_fn:', istft_irelevant_mask.grad_fn) #-- OK
        relevant_mask_waveform = audio_processor.extract_features_istft(istft_relevant_mask)
        irelevant_mask_waveform = audio_processor.extract_features_istft(istft_irelevant_mask)
        print('relevant_mask_waveform grad_fn :', relevant_mask_waveform.grad_fn) #-- OK
        print('irelevant_mask_waveform grad_fn:', irelevant_mask_waveform.grad_fn) #-- OK

        # relevant_mask_waveform = torch.randn(1, 249, 1920).to(device)
        # irelevant_mask_waveform = torch.randn(1, 249, 1920).to(device)
        # print('relevant_mask_waveform grad_fn :', relevant_mask_waveform.grad_fn)
        # print('irelevant_mask_waveform grad_fn:', irelevant_mask_waveform.grad_fn)
        # print(irelevant_mask_waveform.shape)
        # relevant_mask_waveform = relevant_mask_waveform.squeeze(0)
        # irelevant_mask_waveform = irelevant_mask_waveform.squeeze(0)
        # relevant_mask_waveform = torch.mean(relevant_mask_waveform, dim=0)
        # irelevant_mask_waveform = torch.mean(irelevant_mask_waveform, dim=0)
        # lin = nn.Linear(1920,1).to(device)
        # for name, param in lin.named_parameters():
        #     param.requires_grad = False
        # relevant_mask_logits = lin(relevant_mask_waveform)
        # irelevant_mask_logits = lin(irelevant_mask_waveform)



        relevant_mask_feats = torch.mean(relevant_mask_waveform.squeeze(0), dim=0)
        print('relevant_mask_feats grad_fn:', relevant_mask_feats.grad_fn)
        irelevant_mask_feats = torch.mean(irelevant_mask_waveform.squeeze(0), dim=0)
        print('irelevant_mask_feats grad_fn:', irelevant_mask_feats.grad_fn)
        relevant_mask_logits, relevant_mask_probs = torch_logreg(relevant_mask_feats)
        print('relevant_mask_logits grad_fn:', relevant_mask_logits.grad_fn)
        irelevant_mask_logits, irelevant_mask_probs = torch_logreg(irelevant_mask_feats)
        print('irelevant_mask_logits grad_fn:', irelevant_mask_logits.grad_fn)
        relevant_mask_logits = torch_scaler(relevant_mask_logits)
        print('relevant_mask_logits grad_fn:', relevant_mask_logits.grad_fn)
        irelevant_mask_logits = torch_scaler(irelevant_mask_logits)
        print('irelevant_mask_logits grad_fn:', irelevant_mask_logits.grad_fn)


        # issues might arrise here due to logits scaled applied to the cross entropy that applies sigmoid
        l_in = F.binary_cross_entropy_with_logits(relevant_mask_logits, class_pred.view(1,1).to(device))
        l_out = -F.binary_cross_entropy_with_logits(irelevant_mask_logits, class_pred.view(1,1).to(device))
        ao_loss = self.l_in_w * l_in + self.l_out_w * l_out
        print('ao_loss :', ao_loss.grad_fn)


        # reg_l1 = xhat.abs().mean((-1, -2)) * self.reg_w_l1
        # tv_h = torch.sum(torch.abs(xhat[:, :, :-1] - xhat[:, :, 1:]))  # horizontal differences
        # tv_w = torch.sum(torch.abs(xhat[:, :-1, :] - xhat[:, 1:, :]))  # vertical differences
        # reg_tv = (tv_h + tv_w) * self.reg_w_tv
        # reg_loss = reg_l1 #+ reg_tv

        total_loss = ao_loss #+ reg_loss
        return total_loss
