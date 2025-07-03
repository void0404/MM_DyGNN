import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import os
from typing import Dict, Union, Optional
import traceback
import easytorch
from easytorch.utils import get_logger, set_visible_devices
from easytorch.config import init_cfg
from easytorch.device import set_device_type

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm # Import the colormap modul
def plot_active_regions_comparison_re_modified(graph, timepoint_p, timepoint_q, k=50, save_path=None):
    """
    Plot connectivity heatmaps of the k most active regions at two timepoints,
    with the color bar on the left and the third plot area blank.

    Parameters:
    - graph: Numpy array or Tensor with shape [timepoints, regions, regions] representing connectivity.
              If Tensor, requires torch to be imported.
    - timepoint_p: First timepoint index to visualize.
    - timepoint_q: Second timepoint index to visualize.
    - k: Number of most active regions to display.
    - save_path: Optional path to save the figure.

    Returns:
    - fig: The matplotlib figure.
    """
    # Check if input is a torch Tensor and convert if necessary
    # Make sure torch is imported if you expect tensors
    try:
        import torch
        if torch.is_tensor(graph):
            graph = graph.detach().cpu().numpy() # Convert PyTorch tensor to numpy array
    except ImportError:
        if not isinstance(graph, np.ndarray):
            raise TypeError("Input 'graph' must be a NumPy array or PyTorch Tensor.")

    adj_p = graph[timepoint_p]
    adj_q = graph[timepoint_q]

    strength_p = np.sum(adj_p, axis=1)
    strength_q = np.sum(adj_q, axis=1)
    combined_strength = strength_p + strength_q
    top_k_indices = np.argsort(combined_strength)[-k:]

    sub_adj_p = adj_p[np.ix_(top_k_indices, top_k_indices)]
    sub_adj_q = adj_q[np.ix_(top_k_indices, top_k_indices)]

    total_timepoints = graph.shape[0]
    time_p_hr = int(timepoint_p * 24 / total_timepoints)
    time_p_min = int((timepoint_p * 24 / total_timepoints % 1) * 60)
    time_q_hr = int(timepoint_q * 24 / total_timepoints)
    time_q_min = int((timepoint_q * 24 / total_timepoints % 1) * 60)
    time_p = f"{time_p_hr:02d}:{time_p_min:02d}"
    time_q = f"{time_q_hr:02d}:{time_q_min:02d}"

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 1, 1]})

    vmin = min(np.min(sub_adj_p), np.min(sub_adj_q))
    vmax = max(np.max(sub_adj_p), np.max(sub_adj_q))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd") # Get the colormap object

    region_labels = [f"R{idx}" for idx in top_k_indices]

    # Plot first timepoint heatmap
    sns.heatmap(sub_adj_p, ax=axes[0], cmap=cmap, norm=norm, cbar=False, # Pass cmap object
                      xticklabels=region_labels, yticklabels=region_labels)
    axes[0].set_title(f'Connectivity at {time_p}')
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('Region')
    if k > 15:
       plt.setp(axes[0].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
       plt.setp(axes[0].get_yticklabels(), rotation=0)

    # Plot second timepoint heatmap
    sns.heatmap(sub_adj_q, ax=axes[1], cmap=cmap, norm=norm, cbar=False, # Pass cmap object
                      xticklabels=region_labels, yticklabels=region_labels)
    axes[1].set_title(f'Connectivity at {time_q}')
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel('')
    axes[1].set_yticks([])
    if k > 15:
       plt.setp(axes[1].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Make the third axes blank
    axes[2].axis('off')

    # --- Corrected Colorbar Section ---
    # Create the axes for the color bar on the left
    cbar_ax = fig.add_axes([0.06, 0.15, 0.015, 0.7]) # Adjust position/size as needed

    # Create a ScalarMappable object with the same norm and cmap used in heatmaps
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create the colorbar using the ScalarMappable
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    # --- End Corrected Section ---

    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_label_position('left')

    # fig.suptitle(f'Comparison of the {k} Most Active Regions: {time_p} vs {time_q} and '
    #              f'Visualization of Real-World Region Mapping', fontsize=14,  y=0.98)
    plt.tight_layout(rect=[0.1, 0.03, 1.0, 0.93]) # Adjust layout

    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        except Exception as e:
            print(f"Error: Could not save figure to {save_path}. Error message: {e}")

    return fig

# config file path
cfg = '../MM_DyGNN/SZM.py'
# checkpoint path (replace with your model path) like：'checkpoints/MM_DyGNN/cd7bfe7de933fb5b483b5c1af91f4849/ckpt.pth'
ckpt_path = '../checkpoints/MM_DyGNN/{file_md5}/ckpt.pth'

cfg = init_cfg(cfg, save=False)
# initialize the configuration
cfg = init_cfg(cfg, save=False)

# set the device type (CPU, GPU, or MLU)
set_device_type(device_type)

# set the visible GPUs if the device type is not CPU
if device_type != 'cpu':
    set_visible_devices(gpus)

logger = get_logger('easytorch-launcher')
logger.info('Launching EasyTorch evaluation.')
logger = get_logger('easytorch-launcher')
logger.info(f"Initializing runner '{cfg['RUNNER']}'")
runner = cfg['RUNNER'](cfg)
# initialize the logger for the runner
runner.init_logger(logger_name='easytorch-evaluation', log_file_name='evaluation_log')
# get the model from the runner
model = runner.model

# get the dynamic graph for bus
# 参数细节
param_nodep1 = model.nodevec_p1
param_nodep2 = model.nodevec_p2
param_nodep3 = model.nodevec_p3
param_nodepk = model.nodevec_pk
def dgconstruct(time_embedding, source_embedding, target_embedding, core_embedding):
    adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding) # 288, 40, 40
    adp = torch.einsum('bj, ajk->abk', source_embedding, adp) # 491, 491, 40
    adp = torch.einsum('ck, abk->abc', target_embedding, adp) # 491,491,
    adp = F.softmax(adp, dim=2)
    return adp
adp_bus = dgconstruct(param_nodep1,param_nodep2,param_nodep3,param_nodepk)

# Select timepoints
timepoint1 = 0
timepoint2 = 8 * 6 # 48
# Call the modified function
fig = plot_active_regions_comparison_re_modified(
    adp_bus,
    timepoint1,
    timepoint2,
    k=50, # Use k_regions defined above
    save_path='figures/fig_5_bus.png' # Save the figure to a file
)


