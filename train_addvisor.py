from tqdm import tqdm
from addvisor import ADDvisor
import torch
from audioprocessor import AudioProcessor
from loss_function import LMACLoss
import os
from classifier_embedder import thresh
audio_processor = AudioProcessor()
addvisor = ADDvisor()
loss = LMACLoss()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





model = ADDvisor(training=True).to(device)

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
                X_stft, X_stft_power = audio_processor.compute_stft(waveform)
                X_stft_phase = torch.angle(X_stft)
                X_stft_power = X_stft_power.unsqueeze(0).to(device) # for batch 1
                X_stft_phase = X_stft_phase.unsqueeze(0).to(device)
                mask = model(features)
                print(mask)
                raw_pred = torch.tensor(audio_processor.compute_forward_classifier(features)).to(device)
                class_pred = 1.0 if raw_pred.item() > thresh - 5e-3 else 0.0
                class_pred = torch.tensor(class_pred, dtype=torch.float32).to(device)
                print('prediction yhat1:', class_pred)
                loss_value = loss_fn.loss_function(mask, X_stft_power,X_stft_phase, class_pred)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                progress_bar.set_description(f"epoch {epoch+1}/{num_epochs} - loss: {loss_value.item():.4f}")

        checkpoint_path = os.path.join(save_path, f"addvisor_epoch_{epoch + 1}_loss_{loss_value.item():.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)


dir_path = r'C:\Machine_Learning_Data\Deepfake_datasets\in-the-wild\DATA\in-the-wild\wav'
train_addvisor(model=model,num_epochs=20,loss_fn=loss, directory=dir_path)



