# %% [markdown]
# ## 1. Init

# %% [markdown]
# ### Imports

# %%
import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from dataclasses import dataclass
from modules.dataloader import MTMCTrackingDataset, Pi3Transform

# %% [markdown]
# ### Prepare

# %%
@dataclass
class Args:
    data_path: str = './data/MTMC_Tracking_2025' # Path to input image directory or a video file
    save_path: str = 'examples/result.ply' # Path to save the output .ply file
    interval: int = -1 # Interval to sample image. Default: 1 for images dir, 10 for video
    ckpt: str = './weights/pi3.safetensors' # Path to the model checkpoint file. Default: None
    device: str = 'cuda' # Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'

# %%
# 1. Prepare model
device = torch.device(Args().device if torch.cuda.is_available() else 'cpu')
if Args.ckpt is not None:
    model = Pi3().to(device).eval()
    if Args.ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        weight = load_file(Args.ckpt)
    else:
        weight = torch.load(Args.ckpt, map_location=device, weights_only=False)
    
    model.load_state_dict(weight)
else:
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

# %% [markdown]
# ## 2. Data + Preprocessing

# %%
dataset_train = MTMCTrackingDataset(data_root=Args.data_path, split='train', transform=Pi3Transform(pixel_limit=255000))

# %%
sample_data = dataset_train[0]
sample_data.keys()

# %%
sample_data['annotations']

# %%
# 2. Prepare input data
# The load_images_as_tensor function will print the loading path
imgs = torch.stack(sample_data['images']).to(device)

# %%
imgs.shape

# %%
import seaborn as sns

import matplotlib.pyplot as plt

# Set up the plot style
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('First 8 Images from Sample Data', fontsize=16)

# Plot the first 8 images
for i in range(8):
    row = i // 4
    col = i % 4
    
    # Convert tensor to numpy and rearrange dimensions for matplotlib (H, W, C)
    img = sample_data['images'][i].permute(1, 2, 0).cpu().numpy()
    
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Image {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Inference

# %%
# 3. Infer
print("Running model inference...")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        res = model(imgs[None]) # Add batch dimension

# %%
# 4. process mask
masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
masks = torch.logical_and(masks, non_edge)[0]

# %%
# 5. Save points
print(f"Saving point cloud to: {Args.save_path}")
write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], Args.save_path)
print("Done.")


