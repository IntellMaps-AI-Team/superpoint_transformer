import os
import sys
import psutil
import time

import numpy as np
import pandas as pd

# Add the project's files to the python path
# file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)

import laspy
import torch
from src.data import Data
from src.utils.color import to_float_rgb

print(torch.__version__)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB
    return cpu_mem, gpu_mem_alloc, gpu_mem_reserved

start_time = time.time()
start_memory = get_memory_usage()

def read_vancouver_tile(
        filepath,
        xyz=True,
        rgb=True,
        intensity=True,
        semantic=True,
        instance=False,
        remap=True,
        max_intensity=600):
    """Read a Vancouver tile saved as LAS.

    :param filepath: str
        Absolute path to the LAS file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param intensity: bool
        Whether intensity should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their Vancouver ID
        to their train ID
    :param max_intensity: float
        Maximum value used to clip intensity signal before normalizing
        to [0, 1]
    """
    # Create an emty Data object
    data = Data()

    las = laspy.read(filepath)

    # Print shape
    print("Shape: ", las.x.shape)
    print(las[69])

    # Populate data with point coordinates
    if xyz:
        print("xyz")
        # Apply the scale provided by the LAS header
        # points = np.vstack((las.x, las.y, las.z)).transpose()
        pos = torch.stack([
            torch.tensor(las[axis])
            for axis in ["x", "y", "z"]], dim=-1)
        print(pos[69])
        print(las.header.scale)
        # pos *= las.header.scale
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    # Populate data with point RGB colors
    if rgb:
        print("rgb")
        # RGB stored in uint16 lives in [0, 65535]
        data.rgb = to_float_rgb(torch.stack([
            torch.FloatTensor(las[axis].astype('float32') / 65535)
            for axis in ["red", "green", "blue"]], dim=-1))
    else: # fill with zeros
        data.rgb = torch.zeros(data.pos.shape[0], 3)


    # Populate data with point LiDAR intensity
    if intensity:
        print("intensity")
        # Heuristic to bring the intensity distribution in [0, 1]
        data.intensity = torch.FloatTensor(
            las['intensity'].astype('float32')
        ).clip(min=0, max=max_intensity) / max_intensity

    # Populate data with point semantic segmentation labels
    # if semantic:
    #     print("semantic")
    #     y = torch.LongTensor(las['classification'])
    #     data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

    # TODO other attributes

    # Populate data with point panoptic segmentation labels
    if instance:
        raise NotImplementedError("The dataset does not contain instance labels.")

    return data
#%%
# filepath = '/home/data/assets/Hybernska-data/Office_points_all.laz'
# filepath = '/home/data/assets/COOP/roof.las'
filepath = '/home/data/assets/poly/NP2_cast1.laz'
columns=['position', 'intensity', 'rgb']
data = read_vancouver_tile(filepath, semantic=False)

print("Data loaded in ", time.time() - start_time, "s")
end_cpu_mem, end_gpu_alloc, end_gpu_reserved = get_memory_usage()
print("Memory usage: CPU: ", end_cpu_mem - start_memory[0], "MB, GPU: ", end_gpu_alloc - start_memory[1], "MB")

#%%
data
#%%
from src.utils import init_config

cfg = init_config(overrides=[f"experiment=semantic/s3dis"])
#%%
from src.transforms import instantiate_datamodule_transforms

transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)
#%%
import hydra
from src.utils import init_config

# Path to the checkpoint file downloaded from https://zenodo.org/records/8042712
ckpt_path = "/home/data/models/spt-2_s3dis_fold1.ckpt"

cfg = init_config(overrides=[f"experiment=semantic/s3dis"])

# Instantiate the model and load pretrained weights
model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(ckpt_path)

#%%
# Save the pointcloud to a las file
las = laspy.read(filepath)

# Create a descriptive output filename from the input filepath

out_path = os.path.join(
    os.path.dirname(filepath),
    os.path.basename(filepath).replace('.las', '_semantic.las').replace('.laz', '_semantic.las'))

out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
del las

#%%
from src.transforms import SampleRecursiveMainXYAxisTiling
from src.transforms import NAGRemoveKeys

# Recursively tile the cloud into `2**pc_tiling` chunks with respect to
# principal components of the XY coordiantes
pc_tiling = 2

columns.extend(['tile', 'label'])
pointcloud = pd.DataFrame(columns=columns)

loop_time = time.time()

# Compute each chunk
chunks = []
for x in range(2**pc_tiling):
    iter_time = time.time()
    print("Chunk ", x+1, " of ", 2**pc_tiling, " started")
    end_cpu_mem, end_gpu_alloc, end_gpu_reserved = get_memory_usage()
    print("Memory usage: CPU: ", end_cpu_mem - start_memory[0], "MB, GPU: ", end_gpu_alloc - start_memory[1], "MB")
    # Extract the chunk at x in the recursive tiling
    chunk = SampleRecursiveMainXYAxisTiling(x=x, steps=pc_tiling)(data)

    # Add a 'tile' attribute to the points for visualization
    chunk.tile = torch.full((chunk.num_points,), x)

    nag = transforms_dict['pre_transform'](chunk)

    # Simulate the behavior of the dataset's I/O behavior with only
    # `point_load_keys` and `segment_load_keys` loaded from disk
    nag = NAGRemoveKeys(level=0, keys=[k for k in nag[0].keys if k not in cfg.datamodule.point_load_keys])(nag)
    nag = NAGRemoveKeys(level='1+', keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys])(nag)

    # Move to device
    nag = nag.cuda()

    # Apply on-device transforms
    nag = transforms_dict['on_device_test_transform'](nag)

    # Set the model in inference mode on the same device as the input
    model = model.eval().to(nag.device)

    # Inference, returns a task-specific ouput object carrying predictions
    with torch.no_grad():
        output = model(nag)

    print("...done")

    raw_semseg_y = output.full_res_semantic_pred(
        super_index_level0_to_level1=nag[0].super_index,
        sub_level0_to_raw=nag[0].sub)

    # append the chunk data to the pointcloud for each column
    chunk_data =pd.DataFrame({
        'position': chunk.pos.cpu().numpy().tolist(),
        'intensity': chunk.intensity.cpu().numpy().tolist(),
        'rgb': chunk.rgb.cpu().numpy().tolist(),
        'label': raw_semseg_y.cpu().numpy().tolist(),
        'tile': np.full((chunk.num_points,), x)
    })
    pointcloud = pd.concat([pointcloud, chunk_data])

    print("Chunk ", x+1, " of ", 2**pc_tiling, " done in ", time.time() - iter_time, "s")
    print("Total time elapsed: ", time.time() - loop_time, "s")

    end_cpu_mem, end_gpu_alloc, end_gpu_reserved = get_memory_usage()
    print("Memory usage: CPU: ", end_cpu_mem - start_memory[0], "MB, GPU: ", end_gpu_alloc - start_memory[1], "MB")

    # if x == 2:
    #     break

out_las.x = pointcloud['position'].apply(lambda x: x[0])
out_las.y = pointcloud['position'].apply(lambda x: x[1])
out_las.z = pointcloud['position'].apply(lambda x: x[2])
out_las.red = pointcloud['rgb'].apply(lambda x: int(65535 * x[0]))
out_las.green = pointcloud['rgb'].apply(lambda x: int(65535 * x[1]))
out_las.blue = pointcloud['rgb'].apply(lambda x: int(65535 * x[2]))
out_las.intensity = pointcloud['intensity'].apply(lambda x: int(65535 * x))
out_las.classification = pointcloud['label']
out_las.segment = pointcloud['tile']
out_las.write(out_path)
del pointcloud

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Default labels for S3DIS dataset
S3DIS_LABELS = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter"
}

def plot_top_down(x, y, title, output_dir, colors=None):
    plt.figure(figsize=(10, 10))
    if colors is not None:
        plt.scatter(x, y, s=0.5, c=colors, alpha=0.5)
    else:
        plt.scatter(x, y, s=0.5, c='black', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(False)
    plt.gca().set_facecolor((0, 0, 0, 0))  # Transparent background
    plt.savefig(os.path.join(output_dir, f"{title}.png"), dpi=300, transparent=True)
    plt.close()


def generate_views(las, output_dir="output_views"):
    os.makedirs(output_dir, exist_ok=True)

    x, y = las.x, las.y

    # Generate views for classification
    if hasattr(las, "classification"):
        classifications = np.unique(las.classification)
        for cls in classifications:
            label = S3DIS_LABELS.get(cls, f"classification_{cls}")
            mask = las.classification == cls
            plot_top_down(x[mask], y[mask], label, output_dir)

    # Generate a single view for the tile variable (if exists)
    if hasattr(las, "tile"):
        unique_tiles = np.unique(las.tile)
        print(f"Unique tiles: {unique_tiles}")
        colormap = cm.get_cmap("tab10", len(unique_tiles))  # Use tab10 for distinct colors
        color_dict = {tile: colormap(i) for i, tile in enumerate(unique_tiles)}
        colors = np.array([color_dict[tile] for tile in las.tile])
        plot_top_down(x, y, "tile_view", output_dir, colors=colors)

print('Saving the output las file...')
print('Output path: ', out_path)
print('Time elapsed: ', time.time() - start_time, 's')
mem = get_memory_usage()
print("Memory usage: CPU: ", mem[0] - start_memory[0], "MB, GPU: ", mem[1] - start_memory[1], "MB")

# Create folder views
out_folder = os.path.join(os.path.dirname(filepath), 'views_' + os.path.basename(filepath).split('.')[0])

# Generate views
generate_views(out_las, out_folder)