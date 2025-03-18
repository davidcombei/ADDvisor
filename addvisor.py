import torch.nn as nn
import torch
from audioprocessor import AudioProcessor
import torch.nn.functional as F
audio_processor = AudioProcessor()


#########################
### important info from paper
"""
The decoder applies the mask
not directly to the specific input features of the pretrained
classifier but to the magnitude of Short-Time Fourier Transform (STFT) of the original input audio waveform


 
The decoder of
L-MAC consists of a series of transposed 2D convolutional layers. Each convolutional layer upsamples the time and
frequency axis. The classifier’s representations are fed to the
decoder in a U-Net-like fashion to incorporate information
at different time-frequency resolutions (as shown in Figure
5 in Appendix A). Specifically, the decoder takes the four
deepest representations of the classifier.


"""


#############
# notite pt mine
"""

binary mask care pastreaza doar partile relevante ( 1 ) iar partile irelevante vor fi silenced ( inmultire cu 0 ) 
 masca o voi lasa trainable ( doar sigmoid aplicat) dupa o voi binariza in momentul inferentei cu un threshold
 -- am pus deja parametrii in arhitectura
 
 By inheriting the phase from the original signal, we can perform the Inverse Short-Time Fourier Transform to generate listanable maps 
 = > acelasi lucru pentru clasificatorul nostru  pt yhat2
 
 
"""

####################################
#### DEFINE THE DECODER ############
####################################
class ADDvisor(nn.Module):
    def __init__(self, wav2vec2_dim=1920, num_freq_bins=int((audio_processor.n_fft//2)+1), time_steps_w2v = (audio_processor.audio_length*50)-1 ,
                 time_steps_stft =int(audio_processor.audio_length * (audio_processor.sampling_rate / audio_processor.hop_length))+1, threshold = 0.3 ,
                 training = True):
        super(ADDvisor, self).__init__()



        self.threshold = threshold
        self.num_freq_bins = num_freq_bins
        self.time_steps_w2v = time_steps_w2v
        self.time_steps_stft = time_steps_stft
        self.training = training

        #linear projection layers to lower the dimension space of wav2vec2 feats to number of frequency bins of STFT computation
        # self.linproj1 = nn.Linear(wav2vec2_dim, 1024)
        # self.linproj2 = nn.Linear(1024, self.num_freq_bins)

        self.conv1 = nn.Conv1d(wav2vec2_dim, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(1024, 768, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(768, self.num_freq_bins, kernel_size=3, stride=1, padding=1)

        #deconvolution layers to upscale the time steps of wav2vec2 to the time steps of the STFT
        # self.deconv1 = nn.ConvTranspose2d(1, 32, (3, 3), (1, 2), 1)
        # self.deconv2 = nn.ConvTranspose2d(32, 64, (3, 3), (1, 2), 1)
        # self.deconv3 = nn.ConvTranspose2d(64, 128, (3, 3), (1, 2), 1)
        # self.deconv4 = nn.ConvTranspose2d(128, 256, (3, 3), (1, 2), 1)
        # self.deconv5 = nn.ConvTranspose2d(256, 1, (3, 3), (1, 2),1)


        self.sigmoid = nn.Sigmoid()



    def forward(self, h_w2v):


        # B = batch size, T = time steps, F=  num frequency bins
        #x = F.relu(self.linproj1(h_w2v)) # B, T_w2v, 1024)
        #x = self.linproj2(x).permute(0,2,1) # B, F , T_w2v

        # x = x.unsqueeze(1) # B, 1 , F, T_w2v -- match STFT shape
        # #print(x.shape)
        # # upsampling the time steps of w2v2 to STFT time steps
        # x = F.relu(self.deconv1(x))
        # #print(x.shape)
        # x = F.relu(self.deconv2(x))
        # #print(x.shape)
        # x = F.relu(self.deconv3(x))
        # #print(x.shape)
        # x = F.relu(self.deconv4(x))
        # #print(x.shape)
        # x = self.deconv5(x).squeeze(1) # B, F, T_stft

        x = h_w2v.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        #x = x.permute(0, 2, 1)


        #print(self.time_steps_stft)
        mask = self.sigmoid(x)
        # if not self.training:
        #     mask = (mask > self.threshold).float()

        return mask


