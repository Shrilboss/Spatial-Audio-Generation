import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import os
import librosa
from audioldm.utils import default_audioldm_config, get_metadata
from audioldm.audio import read_wav_file, wav_to_fbank
from audioldm.audio.stft import TacotronSTFT
from audioldm import LatentDiffusion, seed_everything
import math
from transformers import get_scheduler
from tqdm import tqdm, trange
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys,glob


class Text2AudioDataset(Dataset):
    def __init__(self, csv_file,root_dir, text_column="caption", audio_column="audiocap_id", stft=None):
        
        self.df = pd.read_csv(csv_file)
        self.audio_column = audio_column
        self.text_column = text_column
        self.root_dir = root_dir
        self.stft = stft

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        audio_name = os.path.join(self.root_dir, f"{self.df[self.audio_column][index]}.wav")
        angle = np.random.randint(0,37) * 5
        swap = np.random.randint(0,2) # swap is for using more than 180 degrees
        use_channel = np.random.randint(0,2) # use_channel is for deciding which channel to use
        mel, _, waveform = wav_to_fbank(audio_name,angle=angle,swap=swap,fn_STFT=self.stft, channel=use_channel)
        text = self.df[self.text_column][index]
        if(use_channel != swap):
            new_angle = 360 - angle
        else:
            new_angle = angle
        if(use_channel == 1):
            text = f"RIGHT-{new_angle:03d}-{text}"
        else:
            text = f"LEFT-{new_angle:03d}-{text}"
        return mel,waveform, text

def train_step():
    pass

def train():
    pass

def build_model(ckpt_path=None,config=None,model_name="audioldm-s-full"):
    print("Load AudioLDM: %s", model_name)
    
    if(ckpt_path is None):
        ckpt_path = get_metadata()[model_name]["path"]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    config = default_audioldm_config(model_name)

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path
    
    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "text"

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )
    fn_STFT = fn_STFT.to("cpu")
    return latent_diffusion, fn_STFT

def setup(num_epochs=40, batch_size=2, lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):

    torch.cuda.empty_cache()
    audioldm_full, stft = build_model()

    train_dataset = Text2AudioDataset(csv_file="dataset/train_final.csv", root_dir="audio_data_20000_end",stft=stft)
    val_dataset = Text2AudioDataset(csv_file="dataset/val_filtered.csv", root_dir="music_data_val",stft=stft)
    test_dataset = Text2AudioDataset(csv_file="dataset/test_filtered.csv", root_dir="music_data_test",stft=stft)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    optimizer_parameters = audioldm_full.model.diffusion_model.parameters()
    print("Optimizing UNet parameters.")
    num_trainable_parameters = sum(p.numel() for p in audioldm_full.parameters() if p.requires_grad)
    print("Num trainable parameters: {}".format(num_trainable_parameters))

    optimizer = AdamW(optimizer_parameters, lr=lr,betas=betas, eps=eps, weight_decay=weight_decay)
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(name="linear",optimizer=optimizer,num_warmup_steps=0,num_training_steps=max_train_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    total_batch_size = batch_size

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_epochs}")
    print(f"  Instantaneous batch size per device = {batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {4}")
    print(f"  Total optimization steps = {max_train_steps}")


    duration, best_loss = 10, np.inf
    loss_list = []
    for epoch in range(num_epochs):
        audioldm_full.train()
        total_loss, total_val_loss = 0, 0
        progress_bar = tqdm(range(len(train_dataloader)))
        for step, batch in enumerate(train_dataloader):

            audioldm_full.zero_grad()
            device = audioldm_full.device
            mel,_,text = batch
            # target_length = 1024

            with torch.no_grad():
                # unwrapped_vae = accelerator.unwrap_model(vae)
                # mel, _, waveform = wav_to_fbank(audios, target_length, stft)
                mel = mel.unsqueeze(1).to(device)
                #Augment with mixed audio
                # mixed_mel, _, _, mixed_captions = torch_tools.augment_wav_to_fbank(audios, text, len(audios), target_length, stft)
                # mixed_mel = mixed_mel.unsqueeze(1).to(device)
                # mel = torch.cat([mel, mixed_mel], 0)
                # text += mixed_captions
                true_latent = audioldm_full.get_first_stage_encoding(audioldm_full.encode_first_stage(mel))

            loss, loss_dict = audioldm_full(true_latent, text,validation=False)
            loss_list.append(loss.detach().float())
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        print(f"Epoch: {epoch}/{num_epochs}, Loss: ", total_loss/len(train_dataloader))
        model_name = f"saved_models/audioldm_losstest_epoch_{epoch}.pt"
        torch.save({
            "audioldm_state_dict" : audioldm_full.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "lr_scheduler_state_dict" : lr_scheduler.state_dict(),
            "epoch" : epoch,
            "loss" : loss,
            "loss_dict" : loss_dict,
            }, model_name)
        print(f'saved model - {model_name}')
    with open('loss_list.pkl', 'wb') as f:
        pickle.dump(loss_list, f)
    # audioldm_full.eval()
    # model.uncondition = False

    # eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    # for step, batch in enumerate(eval_dataloader):
    #     with accelerator.accumulate(model) and torch.no_grad():
    #         device = model.device
    #         text, audios, _ = batch
    #         target_length = int(duration * 102.4)

    #         unwrapped_vae = accelerator.unwrap_model(vae)
    #         mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
    #         mel = mel.unsqueeze(1).to(device)
    #         true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

    #         val_loss = model(true_latent, text, validation_mode=True)
    #         total_val_loss += val_loss.detach().float()
    #         eval_progress_bar.update(1)

    # model.uncondition = args.uncondition

    # if accelerator.is_main_process:    
    #     result = {}
    #     result["epoch"] = epoch+1,
    #     result["step"] = completed_steps
    #     result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
    #     result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)

    #     wandb.log(result)

    #     result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(epoch, result["train_loss"], result["val_loss"])
        
    #     accelerator.print(result_string)

    #     with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
    #         f.write(json.dumps(result) + "\n\n")

    #     logger.info(result)

    #     if result["val_loss"] < best_loss:
    #         best_loss = result["val_loss"]
    #         save_checkpoint = True
    #     else:
    #         save_checkpoint = False

    # if args.with_tracking:
    #     accelerator.log(result, step=completed_steps)

    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process and args.checkpointing_steps == "best":
    #     if save_checkpoint:
    #         accelerator.save_state("{}/{}".format(args.output_dir, "best"))
            
    #     if (epoch + 1) % args.save_every == 0:
    #         accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

    # if accelerator.is_main_process and args.checkpointing_steps == "epoch":
    #     accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))


if __name__ == "__main__":
    setup(num_epochs=2, batch_size=4, lr=1e-5)