from ssd import SSD
import torch


PRETRAINED_PATH = 'saved_model.pth'
example_pic = 'guy_in_space.jpg'

model = SSD(phase='test', size=(1250,2222), base=300)
model.load_state_dict(torch.load(PRETRAINED_PATH))
model.eval()