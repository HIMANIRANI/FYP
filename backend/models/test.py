import faiss

print(faiss.get_num_gpus())  

import torch

print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs detected
print(torch.cuda.get_device_name(0)) 
