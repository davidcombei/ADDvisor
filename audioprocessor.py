from classifier_embedder import wav2vec2, processor, classifier, scaler, thresh
import torch
import numpy as np
import librosa
import torchaudio.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#freeze the SSL model
wav2vec2 = wav2vec2.to(device)
#wav2vec2.eval()
thresh = thresh - 5e-3 # due to venv differences the precision of features might deviate
class AudioProcessor:
    def __init__(self, sampling_rate=16000, n_fft=1024, hop_length=322,win_length=644,  n_mels=80,audio_length =5):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.audio_length = audio_length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )

    #############
    ## JUST A WRAPPER FOR LIBROSA
    #############
    def load_audio(self,audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        return audio, sr


    ###########################
    ####### EXTRACT FEATURES
    ###########################
    def extract_features(self, audio_path):
        audio, sr = self.load_audio(audio_path)

        length = int(self.audio_length * sr)
        if len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))
        else:
            audio = audio[:length]

        audio = torch.tensor(audio, dtype=torch.float32).to(device)

        input_values = processor(audio, return_tensors="pt", sampling_rate=16000)
        input_values = input_values["input_values"].to(device)

        #with torch.no_grad():
        output = wav2vec2(input_values, output_hidden_states=True)

        return output.hidden_states[9]#.squeeze(0)

    def extract_features_istft(self, feats):
        audio = feats.to(device)

        input_values = processor(audio, return_tensors="pt", sampling_rate=16000)
        input_values = input_values["input_values"].to(device)

        #with torch.no_grad():
        output = wav2vec2(input_values, output_hidden_states=True)

        return output.hidden_states[9]#.squeeze(0)

    ############################
    ##### CLASSIFIER FORWARD PASS
    ############################
    def compute_forward_classifier(self, features):

        if isinstance(features, torch.Tensor):
            features = features.squeeze(0)
            features_avg = torch.mean(features, dim=0).cpu().numpy()
        elif isinstance(features, np.ndarray):
            features_avg = features.mean(axis=0, keepdims=True)
        else:
            raise TypeError("expected torch.Tensor or numpy.ndarray but got {}".format(type(features)))

        features_avg = features_avg.reshape(1, -1)
        decision_score_classifier = classifier.decision_function(features_avg)
        decision_score_scaled = scaler.transform(decision_score_classifier.reshape(-1, 1)).flatten()
        return decision_score_scaled[0]

    ###################################################################
    ### COMPUTE THE STFT TO GET THE SPECTROGRAMS AND PHASE INFORMATION
    ###################################################################
    def compute_stft(self,waveform):
        #obtain the spectrogram using stft
        length = int(self.audio_length * self.sampling_rate)
        if len(waveform) < length:
            waveform = np.pad(waveform, (0, length - len(waveform)))
        else:
            waveform = waveform[:length]
        waveform = torch.tensor(waveform, dtype=torch.float32)
        X_stft = torch.stft(waveform,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            return_complex=True # keep the magnitude and phase information
                            )
        #compute the stft power , whihc is the magnitude raise to power of 0.5 ( LMAC model )
        X_stft_power = torch.abs(X_stft) ** 2

        return  X_stft, X_stft_power



    #####################################################################################################
    ###### COMPUTE THE ISTFT TO GET THE AUDIO FROM THE MASKED SPECTROGRAM OBTAINED BY THE DECODER
    #####################################################################################################
    def compute_invert_stft(self, spectrogram):
        if not torch.is_complex(spectrogram):
            raise ValueError("ISTFT expects complex input!")

        expected_length = self.audio_length * self.sampling_rate

        waveform = torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=expected_length
        )

        return waveform









