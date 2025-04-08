# import torch.nn as nn
# import torch
# from audioprocessor import AudioProcessor
# import torch.nn.functional as F
# audio_processor = AudioProcessor()
#
#
# ####################################
# #### DEFINE THE DECODER ############
# ####################################
# class ADDvisor(nn.Module):
#     def __init__(self, wav2vec2_dim=1920, num_freq_bins=int((audio_processor.n_fft//2)+1), time_steps_w2v = (audio_processor.audio_length*50)-1 ,
#                  time_steps_stft =int(audio_processor.audio_length * (audio_processor.sampling_rate / audio_processor.hop_length))+1, threshold = 0.3):
#         super(ADDvisor, self).__init__()
#
#
#
#         self.threshold = threshold
#         self.num_freq_bins = num_freq_bins
#         self.time_steps_w2v = time_steps_w2v
#         self.time_steps_stft = time_steps_stft
#
#
#         self.conv1 = nn.Conv1d(wav2vec2_dim, 1024, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(1024, 768, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(768, self.num_freq_bins, kernel_size=3, stride=1, padding=1)
#
#
#         self.sigmoid = nn.Sigmoid()
#
#
#
#     def forward(self, h_w2v):
#
#         x = h_w2v.permute(0, 2, 1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)
#         mask = torch.sigmoid(x)
#
#         return mask
#
#

import torch
import torch.nn as nn
from audioprocessor import AudioProcessor
class ADDvisor(nn.Module):
    def __init__(self,wav2vec2_dim=1920,num_freq_bins=513,d_model=512,nhead=8,num_encoder_layers=6,num_decoder_layers=6,max_time_steps_stft=249):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead, modify one"
        self.input_proj = nn.Linear(wav2vec2_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers
        )
        # learnable decoder input (queries), one per target STFT time step, random initialization at the beginning
        self.query_pos_emb = nn.Parameter(torch.randn(1, max_time_steps_stft, d_model))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_decoder_layers
        )
        self.output_proj = nn.Linear(d_model, num_freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_w2v):

        batch_size = h_w2v.size(0)
        memory = self.input_proj(h_w2v)
        memory = self.encoder(memory)
        # learned query input to decoder to generate the mask
        tgt = self.query_pos_emb.expand(batch_size, -1, -1)
        out = self.decoder(tgt, memory)
        out = self.output_proj(out)
        mask = self.sigmoid(out)
        return mask.permute(0, 2, 1)
