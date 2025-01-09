import os
os.chdir("/depthanything")
print(os.getcwd())

import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Folder

image_folder = "/scratchdata/test_image"

cnt = 0

for image in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image)
    image = Image.open(image_path)
    depth = pipe(image)["depth"]
    print(depth)

    plt.imsave(f"{image_folder}/{cnt}.png", depth)

    cnt += 1