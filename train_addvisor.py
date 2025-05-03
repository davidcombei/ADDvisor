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

model = ADDvisor().to(device)
#model.load_state_dict(checkpoint)


torch_log_reg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)



def find_all_wav_files1(root_dir):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".wav"):
                full_path = os.path.join(dirpath, file)
                audio_files.append(full_path)
    return audio_files



def find_all_wav_files2(root_dir, max_files=None):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".wav"):
                full_path = os.path.join(dirpath, file)
                audio_files.append(full_path)
                if max_files is not None and len(audio_files) >= max_files:
                    return audio_files
    return audio_files


class AudioDataset(Dataset):
    def __init__(self, directory1, directory2, audio_processor, device):
        files1 = find_all_wav_files2(directory1, max_files=10000)
        files2 = find_all_wav_files2(directory2, max_files=10000)
        self.file_paths = files1 + files2
        self.audio_processor = audio_processor
        self.device = device

        
    def __len__(self):
        return len(self.file_paths)
#        return 15000
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
    yhat = torch_scaler(yhat_logits)
    #thresh_tensor = torch.tensor(thresh, device=yhat_logits.device, dtype=yhat_logits.dtype)
#    class_pred = (yhat_logits > thresh_tensor).float()
    
    return features, magnitude, phase, yhat_logits

def train_addvisor(model, num_epochs, loss_fn, data_loader, save_path):
    for epoch in range(num_epochs):
        total_loss = 0
        total_l_in,total_l_out,total_l1 = 0.0,0.0,0.0
        total_nr_samples = len(data_loader)
#        print(total_nr_samples)
        progress_bar = tqdm(data_loader, desc=f"training ADDvisor... epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True, ascii=True)
        for i,batch in enumerate(progress_bar):
                features, magnitude, phase, yhat_logits = batch
                features = features.to(device)
                magnitude = magnitude.to(device)
                phase = phase.to(device)
 #               yhat = yhat.to(device)
                mask = model(features)
                loss_value,  l_in, l_out, reg_l1, yhat_relevant, yhat_irrelevant = loss_fn.loss_function(mask, magnitude,phase, torch.sigmoid(yhat_logits))
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
#                print(f"loss_value: {loss_value}, shape: {loss_value.shape}")
                total_l_in += l_in.item()
                #print(total_l_in)
                total_l_out += l_out.item()
                total_l1 += reg_l1.item()
                total_loss += loss_value.item()
                if i == len(data_loader) - 1:
                    last_yhat = yhat_logits
                    last_yhat_relevant = yhat_relevant
                    last_yhat_irrelevant = yhat_irrelevant
                
                progress_bar.set_postfix({'loss': f'{loss_value.item():.4f}'})
        avg_loss = total_loss / len(data_loader)
        checkpoint_path = os.path.join(save_path, f"addvisor_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth")
        log_line = f"Epoch {epoch+1}: l_in={total_l_in/total_nr_samples:.4f}, l_out={total_l_out/total_nr_samples:.4f}, L1={total_l1/total_nr_samples:.4f}, Logits_yhat_original:{last_yhat}, Logits_yhat_relevant: {last_yhat_relevant}, Logits_yhat_irrelevant: {last_yhat_irrelevant}\n"
        with open("/mnt/QNAP/comdav/logs/ADDvisor_loss_terms.txt", "a") as f:
            f.write(log_line)
        accelerator.save(model.state_dict(), checkpoint_path)
                         

#dir_path = '/mnt/QNAP/comdav/partial_spoof/eval/con_wav/'
dir_path1 = '/mnt/QNAP/comdav/MLAAD_v5/'
dir_path2 = '/mnt/QNAP/comdav/m-ailabs/'
save_path = '/mnt/QNAP/comdav/addvisor_savedV6/'

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


BATCH_SIZE = 20
dataset = AudioDataset(directory1 = dir_path1,
                       directory2 = dir_path2,
                       audio_processor = audio_processor,
                       device = device)
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)#, scheduler)
train_addvisor(model=model, num_epochs=100, loss_fn=loss, data_loader=data_loader, save_path=save_path)
