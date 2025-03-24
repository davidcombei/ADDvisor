from tqdm import tqdm
from addvisor import ADDvisor
import torch
from audioprocessor import AudioProcessor
from loss_function import LMACLoss
from classifier_embedder import TorchLogReg, TorchScaler, thresh
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_processor = AudioProcessor()
addvisor = ADDvisor()
loss = LMACLoss()

model = ADDvisor().to(device)
torch_log_reg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)

class AudioDataset(Dataset):
    def __init__(self, directory, audio_processor, device):
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sr = self.audio_processor.load_audio(audio_path)
        return waveform#, audio_path


def collate_fn(batch):
    waveforms = torch.stack(batch, dim=0)
    #print(waveforms.shape)
    _, magnitude, phase = audio_processor.compute_stft(waveforms)
    #print(magnitude.shape)
    #print(phase.shape)
    features = audio_processor.extract_features(waveforms)
    feats_mean = torch.mean(features, dim=1)
    yhat_logits, _ = torch_log_reg(feats_mean)
    yhat_logits = torch_scaler(yhat_logits)
    thresh_tensor = torch.tensor(thresh, device=yhat_logits.device, dtype=yhat_logits.dtype)
    class_pred = (yhat_logits > thresh_tensor).float()

    return features, magnitude, phase, class_pred


def train_addvisor(model, num_epochs, loss_fn, data_loader, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"training ADDvisor... epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True)
        for i,batch in enumerate(progress_bar):
                features, magnitude, phase, class_pred = batch
                features = features.to(device)
                magnitude = magnitude.to(device)
                phase = phase.to(device)
                class_pred = class_pred.to(device)
                mask = model(features)
                loss_value = loss_fn.loss_function(mask, magnitude,phase, class_pred)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                total_loss += loss_value.item()
                # for name, param in model.named_parameters():
                #     print(f"{name}: requires_grad={param.requires_grad}")
                avg_loss = total_loss / len(data_loader)
                progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        checkpoint_path = os.path.join(save_path, f"addvisor_epoch_{epoch + 1}_loss_{avg_loss.item():.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)


dir_path = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav'
save_path = r'C:\Users\david\PycharmProjects\David2\ADDvisor\saved'


BATCH_SIZE = 8
dataset = AudioDataset(directory = dir_path,
                       audio_processor = audio_processor,
                       device = device)

data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
train_addvisor(model=model, num_epochs=10, loss_fn=loss, data_loader=data_loader, save_path=save_path)




