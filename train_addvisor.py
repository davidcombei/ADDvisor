from tqdm import tqdm
from addvisor import ADDvisor
import torch
from audioprocessor import AudioProcessor
from loss_function import LMACLoss
from classifier_embedder import TorchLogReg, TorchScaler, thresh
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_processor = AudioProcessor()
addvisor = ADDvisor()
loss = LMACLoss()


model = ADDvisor().to(device)
torch_log_reg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)

def train_addvisor(model, num_epochs, loss_fn, directory):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
    save_path = r'C:\Users\david\PycharmProjects\David2\ADDvisor\saved'
    for epoch in range(num_epochs):
        progress_bar = tqdm(os.listdir(directory), desc=f"epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)

        for file in progress_bar:
            if file.endswith(".wav"):
                audio_path = os.path.join(directory, file)

                features = audio_processor.extract_features(audio_path)
                waveform,sr = audio_processor.load_audio(audio_path)
                X_stft, magnitude, phase = audio_processor.compute_stft(waveform)
                magnitude = magnitude.unsqueeze(0)
                phase = phase.unsqueeze(0)
                mask = model(features)

                feats = torch.mean(features.squeeze(0), dim=0)
                yhat1_logits, yhat1_probs = torch_log_reg(feats)
                yhat1_probs = torch_scaler(yhat1_logits)
                class_pred = torch.tensor([1.0], device=device,requires_grad=False) if yhat1_probs.item() > thresh - 5e-6 else torch.tensor(
                    [0.0], device=device, requires_grad=False)
                #print('class yhat1:', class_pred)
                loss_value = loss_fn.loss_function(mask, magnitude,phase, class_pred)
                optimizer.zero_grad()
                loss_value.backward()
                #print("mask.grad:", mask.grad)
                optimizer.step()
                #print("mask grad:", mask.grad)
                progress_bar.set_description(f"epoch {epoch+1}/{num_epochs} - loss: {loss_value.item():.4f}")

        checkpoint_path = os.path.join(save_path, f"addvisor_epoch_{epoch + 1}_loss_{loss_value.item():.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)


dir_path = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav'
train_addvisor(model=model,num_epochs=20,loss_fn=loss, directory=dir_path)



