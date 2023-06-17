import torch
from audioldm.clap.encoders import CLAPAudioEmbeddingClassifierFreev2
model = CLAPAudioEmbeddingClassifierFreev2()
check =  torch.load(r'C:\Users\3924s\OneDrive\Desktop\Spring 2023\ECE 285 Deep gen\Project\630k-audioset-best.pt')
model.load_state_dict(check['state_dict'])
model.eval()
print(model)