# Multimodal Urban Travel Demand Forecasting via Adaptive Dynamic Graph Learning 

This repository contains the PyTorch implementation for the paper "Multimodal Urban Travel Demand Forecasting via Adaptive Dynamic Graph Learning". Our work proposes **MM-DyGNN**, a novel framework for urban travel demand forecasting. It addresses the challenge of capturing complex and dynamic spatial-temporal dependencies from multimodal data sources (bus,metro,taxi). MM-DyGNN adaptively learns graph structures and utilizes a dynamic graph neural network to effectively model the evolving relationships between different transportation systems.
![MM-DyGNN](./fig/fm_last.png)



## 💿Requirements

The code is built with BasicTS, you can easily install the requirements by (take Python 3.10 + PyTorch 2.3.1 + CUDA 12.1 as an example):

```bash
# Install Python
conda create -n BasicTS python=3.10
conda activate BasicTS
# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
```

More details can be found in [BasicTS](https://github.com/GestaltCogTeam/BasicTS).

## 📦 Data Preparation

1. **Data Availability**
   > The *Shenzhen Multimodal Transport Demand Dataset* is collected under a joint agreement with the Shenzhen Transportation Bureau.  
   > **Public redistribution is temporarily unavailable** while we finalise an English-language licence.  
   > - **Status:** _under licence review_  
   > - **Expected release:** processed (anonymised) files and loading scripts will be hosted here once the review is complete.  


3. **File Structure:** After downloading, please place the data files in the `dataset/` directory and ensure the structure is as follows:

   ```
   MM-DyGNN/
   ├── datasets/
   │   └── [SZM]/
   │       ├── data.dat
   │       └── desc.json
   ├── experiments/
   ...
   ```

   The `data.dat`  contain the multi-modal transport demand  data and `desc.json`  file contains metadata about the dataset, including: shape of the data、feature descriptions、number of time slices、ratios for training, validation, and test sets...

## 🎯 Training

Our model based [BasicTS ](https://github.com/GestaltCogTeam/BasicTS) platform.

You can modify the parameters in the `MM_DyGNN.py` file, such as: prediction time step, learning rate, minimum batch, etc.

```bash
python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '0'
```

For baseline models run

```bash
python experiments/train.py --cfg baselines/${MODEL}/SZM.py --gpus '0'
```

More details can be found in [BasicTS](https://github.com/GestaltCogTeam/BasicTS).



## 📈 Reproducing  Tables and Figures

This repository contains the code necessary to reproduce the results presented in the paper "Adaptive dynamic graph learning to forecast urban demand for multimodal travels." This document provides a step-by-step guide to reproduce all tables and figures in Section 5.

####  **Table 3: Main Performance Comparison** 

![image-20250703154316802](.\fig\performence_comp.png)

To generate the data for this table, you need to train and evaluate both the baseline models and our MM-DyGNN model.

1. Train and Evaluate Baseline Models: Execute the training script for each baseline model. You will need to replace `{model_name}` and `{config_name}` with the appropriate names for each baseline you wish to run. Replace `{gpuid}` with the ID of the GPU you want to use.

   ```
   # Example for a single baseline model
   python experiments/train.py --cfg baselines/{model_name}/{config_name}.py --gpus '{gpuid}'
   ```

2. Train the MM-DyGNN Model: Next, run the training script for our proposed MM-DyGNN model.

   ```
   python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
   ```

After running these scripts, the console output will display the MAE and RMSE values for each model, which you can use to populate Table 3.

#### Table 4 & Figure 5: Ablation Study and Case Study on Dynamic Graph

Table 4 evaluates the contribution of the dynamic graph constructor by comparing it to a static graph version. Figure 5 visualizes the learned bus connectivity at different times of the day to illustrate the dynamic nature of the graph.

To reproduce Table 4  the performance of a static graph version of our model, you need to modify the configuration file.

1. **Modify the Configuration:** Open the configuration file: `./MM_DyGNN/SZM.py`.

2. **Set to Static Mode:** Find the `days` parameter and change its value to `1`. This forces the model to use a static graph instead of a dynamic one.

3. **Run the Experiment:** Execute the training script with the modified configuration.

   ```
   # Run the ablation study for the dynamic vs. static graph、
   # change the parameter 'days' 
   python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
   ```

   The output will provide the performance metrics for the static graph version of MM-DyGNN.

   ![image-20250703190923902](G:\mypaper\publi_code\MM_DyGNN\fig\ablation_1.png)

To reproduce Figure 5：

Use the plotting script to visualize the learned graph for the bus mode. This script loads the dynamic graph model to generate the heatmaps.

``` bas
python ./visualization/fig_5.py
```

#### Table 5 & Figures 6-7: Ablation Study and Case Study on SCMI Module

**To reproduce Table 5**:This study analyzes the contribution of SCMI module. It compares the full model against two variants: one without top-k selection and another using simple summation instead of an attention mechanism.

You will need to run two separate experiments by modifying the configuration file.

1. **Variant 1: No Top-k Selection**

   - **Modify Config:** Open `./MM_DyGNN/SZM.py` and change the value of the `k` parameter to `None`.

   - **Run Experiment:**

     ```
     python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
     ```

     

2. **Variant 2: Simple Summation Fusion**

   - **Modify Config:** Open `./MM_DyGNN/SZM.py` and change the `fusion_way` parameter to use the simple summation method.

   - **Run Experiment:**

     ```
     python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
     ```

The results from these runs will allow you to populate the ablation study results in Table 5.

![image-20250703191020299](G:\mypaper\publi_code\MM_DyGNN\fig\ablation_2.png)



* **To reproduce Figure 7:** This process involves two steps: first, extracting the attention maps from the trained model using a hook function, and second, running the plotting script.

  * **Step 1: Extract Attention Maps with a Hook Function**

  To get the attention maps, you need to use a PyTorch hook to capture the intermediate outputs of the SCMI module during model inference. Add the following helper function to  `runner`  at  `./basicts/runners/runner_zoo`.

  <details>
    <summary>Usage of hook function to get Attention Map</summary>
    ```python
    def _create_hook(attention_maps, layer_name: str):
      """Creates a hook function to capture the attention map."""
      def hook(module, inputs, outputs):
          """
          The hook function itself.
  
          Args:
              module: The layer being hooked.
              inputs: The input to the layer.
              outputs: The output from the layer.
          """
          # --- Key: How to extract the attention map from the outputs ---
          # This depends on the return structure of your Attention layer's forward function.
          # Common cases:
          # 1. The output is the attention map itself.
          # 2. The output is a tuple (features, attention_map).
          # 3. The output is a dictionary {'features': ..., 'attention_map': ...}.
  
          # Example: Assume the Attention layer's forward function returns (features, attention_map)
          # We assume the attention map is the second element of the tuple.
          attn_map = outputs[1][:, 0, :, :]  # Select the attention map of the first head
  
          if attn_map is not None:
              # Move the attention map to the CPU and detach it from the computation graph to save memory.
              # Note: If you need to perform subsequent processing on the GPU, you can leave it there for now.
              attention_maps[layer_name].append(attn_map.detach().cpu())
      
      # Return the actual hook function
      return hook
  	Before running inference, register this hook to the SCMI module of your trained model:
       
      # 'model' is your loaded MM-DyGNN model
  	# 'scmi_layer' is the name of the Sparse Cross-Modal Interaction module in your model
  	attention_maps = {'scmi_layer': []}
  	hook_handle = model.scmi_layer.register_forward_hook(_create_hook(attention_maps, 'scmi_layer'))
  
  	# Run model inference/prediction here...
  	# The `attention_maps` dictionary will now be populated.
  
  	# After inference, save the maps to a file.
  	# import numpy as np
  	# np.savez('attention_maps.npz', scmi_maps=attention_maps['scmi_layer'])
  
  	# Don't forget to remove the hook when you're done
  	hook_handle.remove()
    ```
    
  </details>

  * **Step 2: Plot the Attention Maps**

    Once you have saved the attention maps, use the provided plotting script to generate the visualizations.

    ```bash
    # Generate spatial distribution maps of attention weights
    python ./visualization/fig_7.py
    ```



## Citation

A paper is in progress, and the citation will be added here.

