from tqdm import tqdm
from addvisor import ADDvisor
import torch
from audioprocessor import AudioProcessor
from loss_function import LMACLoss
from classifier_embedder import TorchLogReg, TorchScaler, thresh
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from collections import OrderedDict
accelerator = Accelerator()
device = accelerator.device


audio_processor = AudioProcessor()
addvisor = ADDvisor()
loss = LMACLoss()


#checkpoint_path = '/mnt/QNAP/comdav/addvisor_savedV2/addvisor_REGULARIZED_epoch_20_loss_2.5003.pth'
#checkpoint = torch.load(checkpoint_path, map_location=device)


#the saved checkpoint is accelerate format... remove it
#if any(k.startswith("module.") for k in checkpoint.keys()):
#    new_state_dict = OrderedDict()
#    for k, v in checkpoint.items():
#        new_key = k.replace("module.", "")
#        new_state_dict[new_key] = v
#    checkpoint = new_state_dict


model = ADDvisor().to(device)
#model.load_state_dict(checkpoint)


torch_log_reg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)



def find_all_wav_files(root_dir):
    video_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".wav"):
                full_path = os.path.join(dirpath, file)
                video_files.append(full_path)
    return video_files



class AudioDataset(Dataset):
    def __init__(self, directory, audio_processor, device):
        self.file_paths = find_all_wav_files(directory)
        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sr = self.audio_processor.load_audio(audio_path)
        return waveform.to(device)#, audio_path


def collate_fn(batch):
    waveforms = torch.stack(batch, dim=0)
    _, magnitude, phase = audio_processor.compute_stft(waveforms)
    features = audio_processor.extract_features(waveforms)
    feats_mean = torch.mean(features, dim=1)
    yhat_logits, _ = torch_log_reg(feats_mean)
    yhat_logits = torch_scaler(yhat_logits)
    thresh_tensor = torch.tensor(thresh, device=yhat_logits.device, dtype=yhat_logits.dtype)
    class_pred = (yhat_logits > thresh_tensor).float()

    return features, magnitude, phase, class_pred #, yhat_logits


def train_addvisor(model, num_epochs, loss_fn, data_loader, save_path):
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"training ADDvisor... epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True, ascii=True)
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
                progress_bar.set_postfix({'loss': f'{loss_value.item():.4f}'})
        avg_loss = total_loss / len(data_loader)
        checkpoint_path = os.path.join(save_path, f"addvisor_MLAAD_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth")
    
        accelerator.save(model.state_dict(), checkpoint_path)
                         

#dir_path = '/mnt/QNAP/comdav/partial_spoof/eval/con_wav/'
dir_path = '/mnt/QNAP/comdav/MLAAD_v5/'
save_path = '/mnt/QNAP/comdav/addvisor_savedV2/'


optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

BATCH_SIZE = 24
dataset = AudioDataset(directory = dir_path,
                       audio_processor = audio_processor,
                       device = device)

data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)#, scheduler)
train_addvisor(model=model, num_epochs=100, loss_fn=loss, data_loader=data_loader, save_path=save_path)




