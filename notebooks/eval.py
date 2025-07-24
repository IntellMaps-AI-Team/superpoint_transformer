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
    print("las format: ", las.header.point_format)
    print("las version: ", las.header.version)
    print("Shape: ", las.x.shape)
    print(las[69])

    # Populate data with point coordinates
    if xyz:
        print("xyz")
        # Apply the scale provided by the LAS header
        # points = np.vstack((las.x, las.y, las.z)).transpose()
        coords = np.vstack((las.x, las.y, las.z)).T  # shape (N, 3)
        pos = torch.from_numpy(coords)
        print(pos[69])
        print(las.header.scale)
        # pos *= las.header.scale
        data.scales = las.header.scale
        data.offset = las.header.offset

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
# filepath = '/home/data/assets/poly/NP2_cast1.laz'
filepath = '/home/io/Mesice/general_pano_5514_v8211_Merge.las'
columns=['position', 'intensity', 'rgb']
data = read_vancouver_tile(filepath, semantic=False)

print("Data loaded in ", time.time() - start_time, "s")
end_cpu_mem, end_gpu_alloc, end_gpu_reserved = get_memory_usage()
print("Memory usage: CPU: ", end_cpu_mem - start_memory[0], "MB, GPU: ", end_gpu_alloc - start_memory[1], "MB")
# print(f"Memory footprint of data: {sys.getsizeof(data) / (1024 * 1024):.2f} MB, ")

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

# Create a descriptive output filename from the input filepath

out_path = os.path.join(
    os.path.dirname(filepath),
    os.path.basename(filepath).replace('.las', '_semantic.las').replace('.laz', '_semantic.las'))


# --- Use a new header and add extra dims for tile and cluster ---
import numpy as np
header = laspy.LasHeader(point_format=3, version="1.4")
# header = las.header
header.add_extra_dim(laspy.ExtraBytesParams(name="Tile", type=np.uint8, description="Tile index for recursive tiling"))
header.add_extra_dim(laspy.ExtraBytesParams(name="Cluster", type=np.uint32, description="Cluster index for NAG"))
header.offsets = data.offsets
header.scales = data.scales
# del las

#%%
from src.transforms import SampleRecursiveMainXYAxisTiling
from src.transforms import NAGRemoveKeys

# Recursively tile the cloud into `2**pc_tiling` chunks with respect to
# principal components of the XY coordiantes
pc_tiling = 3

columns.extend(['tile', 'label'])
# pointcloud = pd.DataFrame(columns=columns)  # REMOVE this line
# pointcloud_chunks = []  # REMOVE this line

loop_time = time.time()

# Compute each chunk
chunks = []
with laspy.open(out_path, mode="w", header=header) as writer:
    for x in range(2**pc_tiling):
        # if x == 2:
        #     break
        iter_time = time.time()
        print("Chunk ", x+1, " of ", 2**pc_tiling, " started")
        end_cpu_mem, end_gpu_alloc, end_gpu_reserved = get_memory_usage()
        print("Memory usage: CPU: ", end_cpu_mem - start_memory[0], "MB, GPU: ", end_gpu_alloc - start_memory[1], "MB")
        # Extract the chunk at x in the recursive tiling
        chunk = SampleRecursiveMainXYAxisTiling(x=x, steps=pc_tiling)(data)

        # Add a 'tile' attribute to the points for visualization
        chunk.tile = torch.full((chunk.num_points,), x)

        nag = transforms_dict['pre_transform'](chunk)
        nag = NAGRemoveKeys(level='1+', keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys])(nag)
        nag = nag.cuda()
        nag = transforms_dict['on_device_test_transform'](nag)
        model = model.eval().to(nag.device)
        with torch.no_grad():
            output = model(nag)
        print("...done")

        raw_semseg_y = output.full_res_semantic_pred(
            super_index_level0_to_level1=nag[0].super_index,
            sub_level0_to_raw=nag[0].sub)

        # Prepare numpy arrays for laspy
        pos_np = chunk.pos.cpu().numpy().astype(np.float32)
        rgb_np = chunk.rgb.cpu().numpy().astype(np.float32)
        intensity_np = chunk.intensity.cpu().numpy().astype(np.float32)
        label_np = raw_semseg_y.cpu().numpy().astype(np.uint8)
        tile_np = np.full((chunk.num_points,), x, dtype=np.uint8)
        cluster_np = nag[0].sub.to_super_index().cpu().numpy().astype(np.uint32)

        las_chunk = laspy.LasData(header)
        las_chunk.x = pos_np[:, 0]
        las_chunk.y = pos_np[:, 1]
        las_chunk.z = pos_np[:, 2]
        las_chunk.red = (65535 * rgb_np[:, 0]).astype(np.uint16)
        las_chunk.green = (65535 * rgb_np[:, 1]).astype(np.uint16)
        las_chunk.blue = (65535 * rgb_np[:, 2]).astype(np.uint16)
        las_chunk.intensity = (65535 * intensity_np).astype(np.uint16)
        las_chunk.classification = label_np.astype(np.uint8)
        las_chunk.tile = tile_np
        las_chunk.cluster = cluster_np


#         print(f"Memory footprint of las_chunk: {sys.getsizeof(las_chunk) / (1024 * 1024):.2f} MB, ")
#         print(f"Memory footprint of nag: {sys.getsizeof(nag) / (1024 * 1024):.2f} MB, ")
#         print(f"Memory footprint of chunk: {sys.getsizeof(chunk) / (1024 * 1024):.2f} MB, ")

        writer.write_points(las_chunk.points)

        # Explicitly delete large objects and clear CUDA cache
        del chunk, nag, output, raw_semseg_y, las_chunk, pos_np, rgb_np, intensity_np, label_np, tile_np, cluster_np
        torch.cuda.empty_cache()

        print("Chunk ", x+1, " of ", 2**pc_tiling, " done in ", time.time() - iter_time, "s")
        print("Total time elapsed: ", time.time() - loop_time, "s")

        end_cpu_mem, end_gpu_alloc, end_gpu_reserved = get_memory_usage()
        print("Memory usage: CPU: ", end_cpu_mem - start_memory[0], "MB, GPU: ", end_gpu_alloc - start_memory[1], "MB")

        # if x == 2:
        #     break

# REMOVE: Concatenate all chunks at once (much more memory efficient)
# pointcloud = pd.concat(pointcloud_chunks, ignore_index=True)
# del pointcloud_chunks

# REMOVE: Assign fields to the new LAS object
# out_las.x = pointcloud['position'].apply(lambda x: x[0])
# out_las.y = pointcloud['position'].apply(lambda x: x[1])
# out_las.z = pointcloud['position'].apply(lambda x: x[2])
# out_las.red = pointcloud['rgb'].apply(lambda x: int(65535 * x[0]))
# out_las.green = pointcloud['rgb'].apply(lambda x: int(65535 * x[1]))
# out_las.blue = pointcloud['rgb'].apply(lambda x: int(65535 * x[2]))
# out_las.intensity = pointcloud['intensity'].apply(lambda x: int(65535 * x))
# out_las.classification = pointcloud['label']
# out_las.tile = pointcloud['tile'].astype(np.int32)
# out_las.cluster = pointcloud['cluster'].astype(np.int32)
# out_las.write(out_path)
# del pointcloud

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
    # if hasattr(las, "tile"):
    #     unique_tiles, tile_indices = np.unique(las.tile, return_inverse=True)  # Get unique tiles and their indices
    #     print(f"Unique tiles: {unique_tiles}")
    #
    #     colormap = plt.get_cmap("tab10", len(unique_tiles))  # Use tab10 for distinct colors
    #     colors = colormap(tile_indices)  # Directly map indices to colors using the colormap
    #
    #     plot_top_down(x, y, "tile_view", output_dir, colors=colors)

print('Saving the output las file...')
print('Output path: ', out_path)
print('Time elapsed: ', time.time() - start_time, 's')
mem = get_memory_usage()
print("Memory usage: CPU: ", mem[0] - start_memory[0], "MB, GPU: ", mem[1] - start_memory[1], "MB")

# Create folder views
# out_folder = os.path.join(os.path.dirname(filepath), 'views_' + os.path.basename(filepath).split('.')[0])
#
# # Generate views
# generate_views(out_las, out_folder)
# print('Final time elapsed: ', time.time() - start_time, 's')
