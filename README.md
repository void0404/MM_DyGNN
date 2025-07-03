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

Table 3 compares MM-DyGNN with eight baseline models across three prediction horizons. 

To reproduce this table: Run the main experiment script. This script will train and evaluate the proposed MM-DyGNN and all baseline models. The output will provide the  MAE and RMSE values needed to populate Table 3. 

``` bash
# Train and evaluate baseline models for the main comparison 
python experiments/train.py --cfg baselines/{model_name}/{config_name}.py --gpus '{gpuid}'
# Train and evaluate MM_DyGNN models for the main comparison
python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
```

#### Table 4 & Figure 5: Ablation Study and Case Study on Dynamic Graph

Table 4 evaluates the contribution of the dynamic graph constructor by comparing it to a static graph version. Figure 5 visualizes the learned bus connectivity at different times of the day to illustrate the dynamic nature of the graph.

To reproduce Table 4:

Change the parameter 'days' int the config file './MM_DyGNN/SZM.py' as 1 . This will execute MM-DyGNN with static graphs and report the performance metrics.

```bas
# Run the ablation study for the dynamic vs. static graph、
# change the parameter 'days' 
python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
```

To reproduce Figure 5：

Use the plotting script to visualize the learned graph for the bus mode. This script loads the dynamic graph model to generate the heatmaps.

``` bas
python ./visualization/fig_5.py
```

#### Table 5 & Figures 6-7: Ablation Study and Case Study on SCMI Module

To reproduce Table 5:

Run the ablation script for the SCMI module. This will test three variants: the full model, one without top-k selection, and one that replaces attention with simple summation. Change the parameter 'k' value to None in config file './MM_DyGNN/SZM.py' .For full simple summation variants change the parameter “fusion way“.and train these variants with:

```base
python experiments/train.py --cfg MM_DyGNN/SZM.py --gpus '{gpuid}'
```



## Citation

A paper is in progress, and the citation will be added here.

