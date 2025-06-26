import numpy as np
from torch import nn
import torch

region_mask = np.load('mymodel/loss/mask.npy')
region_mask = ~region_mask   #true表示需要计算第二个观测值的区域


# 不确定性平方损失
def uncertainty_weighted_loss_SE(prediction, target, log_sigma):
    """
    Compute the total loss with uncertainty weighting (using squared error SE).

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        log_sigma (torch.Tensor): Log standard deviation for each task, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")

    # Determine the shape of the input tensors
    if len(prediction.shape) == 4:
        B, L, N, C = prediction.shape
        prediction_flat = prediction.view(B * L * N, C)
        target_flat = target.view(B * L * N, C)
    elif len(prediction.shape) == 3:
        B, N, C = prediction.shape
        prediction_flat = prediction.view(B * N, C)
        target_flat = target.view(B * N, C)
    else:
        raise ValueError(f"Unsupported shape: {prediction.shape}")

    # Compute squared error per channel
    se_loss = (prediction_flat - target_flat) ** 2  # Shape: [B*L*N, C] or [B*N, C]
    se_per_channel = se_loss.sum(dim=0)  # Shape: [C]

    # Compute maximum target value per channel for normalization
    max_per_channel = target_flat.max(dim=0)[0]  # Shape: [C]

    # Prevent division by zero by adding a small epsilon
    epsilon = 1e-5
    max_per_channel = torch.where(max_per_channel == 0, torch.full_like(max_per_channel, epsilon), max_per_channel)

    # Normalize the squared error
    normalized_se = se_per_channel / max_per_channel  # Shape: [C]

    # Apply uncertainty weighting
    # Loss formula: sum_i (exp(-log_sigma_i) * normalized_se_i + log_sigma_i)
    loss = torch.exp(-log_sigma) * normalized_se + log_sigma  # Shape: [C]

    # Sum the losses across all channels to get the total loss
    total_loss = loss.sum()

    return total_loss
# 无归一化 不确定性损失
def se_loss_uncertainty(prediction, target, log_sigma):
    """
    Compute the total loss with uncertainty weighting (using squared error SE).

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        log_sigma (torch.Tensor): Log standard deviation for each task, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")

    # Determine the shape of the input tensors
    if len(prediction.shape) == 4:
        B, L, N, C = prediction.shape
        prediction_flat = prediction.view(B * L * N, C)
        target_flat = target.view(B * L * N, C)
    elif len(prediction.shape) == 3:
        B, N, C = prediction.shape
        prediction_flat = prediction.view(B * N, C)
        target_flat = target.view(B * N, C)
    else:
        raise ValueError(f"Unsupported shape: {prediction.shape}")

    # Compute squared error per channel
    se_loss = (prediction_flat - target_flat) ** 2  # Shape: [B*L*N, C] or [B*N, C]


    # Normalize the squared error


    # Apply uncertainty weighting
    # Loss formula: sum_i (exp(-log_sigma_i) * normalized_se_i + log_sigma_i)
    loss = torch.exp(-log_sigma) * se_loss + log_sigma  # Shape: [C]

    # Sum the losses across all channels to get the total loss
    total_loss = loss.sum()

    return total_loss
# 正则化归一损失
def uncertainty_weighted_loss_SE_standardized(prediction, target, log_sigma):
    """
    Compute the total loss with uncertainty weighting (using squared error SE) and standardization.

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        log_sigma (torch.Tensor): Log standard deviation for each task, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")

    # Determine the shape of the input tensors
    if len(prediction.shape) == 4:
        B, L, N, C = prediction.shape
        prediction_flat = prediction.view(B * L * N, C)
        target_flat = target.view(B * L * N, C)
    elif len(prediction.shape) == 3:
        B, N, C = prediction.shape
        prediction_flat = prediction.view(B * N, C)
        target_flat = target.view(B * N, C)
    else:
        raise ValueError(f"Unsupported shape: {prediction.shape}")

    # Compute squared error per sample per channel
    se_loss = (prediction_flat - target_flat) ** 2  # Shape: [B*L*N, C] or [B*N, C]

    # Compute mean and std of SE per channel for standardization
    se_mean = se_loss.mean(dim=0)  # Shape: [C]
    se_std = se_loss.std(dim=0)    # Shape: [C]

    # Prevent division by zero by adding a small epsilon
    epsilon = 1e-5
    se_std = torch.where(se_std == 0, torch.full_like(se_std, epsilon), se_std)

    # Standardize the squared error
    standardized_se = (se_loss - se_mean) / se_std  # Shape: [B*L*N, C] or [B*N, C]

    # Aggregate standardized SE per channel
    standardized_se_per_channel = standardized_se.sum(dim=0)  # Shape: [C]

    # Apply uncertainty weighting
    # Loss formula: sum_i (exp(-log_sigma_i) * standardized_se_i + log_sigma_i)
    loss = torch.exp(-log_sigma) * standardized_se_per_channel + log_sigma  # Shape: [C]

    # Sum the losses across all channels to get the total loss
    total_loss = loss.sum()

    return total_loss

def dma(prediction, target, weights):
    """
    Compute the total loss with uncertainty weighting (using squared error SE) and standardization.

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        log_sigma (torch.Tensor): Log standard deviation for each task, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")

    # Determine the shape of the input tensors
    if len(prediction.shape) == 4:
        B, L, N, C = prediction.shape
        prediction_flat = prediction.view(B * L * N, C)
        target_flat = target.view(B * L * N, C)
    elif len(prediction.shape) == 3:
        B, N, C = prediction.shape
        prediction_flat = prediction.view(B * N, C)
        target_flat = target.view(B * N, C)
    else:
        raise ValueError(f"Unsupported shape: {prediction.shape}")

    # Compute mae error  per channel
    mae_loss = torch.abs(prediction_flat - target_flat)  # Shape: [B*L*N, C] or [B*N, C]

    # weighted mae by weights

    weighted_mae = mae_loss * torch.tensor(weights, dtype=mae_loss.dtype, device=mae_loss.device)
    # Sum the losses across all channels to get the total loss
    total_loss = weighted_mae.sum()

    return total_loss,mae_loss.sum(dim=0)


def dma_se(prediction, target, weights):
    """
    Compute the total loss with uncertainty weighting (using squared error SE) and standardization.

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        weights (list or torch.Tensor): Weights for each channel, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")

    # Determine the shape of the input tensors
    if len(prediction.shape) == 4:
        B, L, N, C = prediction.shape
        prediction_flat = prediction.view(B * L * N, C)
        target_flat = target.view(B * L * N, C)
    elif len(prediction.shape) == 3:
        B, N, C = prediction.shape
        prediction_flat = prediction.view(B * N, C)
        target_flat = target.view(B * N, C)
    else:
        raise ValueError(f"Unsupported shape: {prediction.shape}")

    # Compute mean and std for each channel
    mean_pred = prediction_flat.mean(dim=0)
    std_pred = prediction_flat.std(dim=0)
    mean_target = target_flat.mean(dim=0)
    std_target = target_flat.std(dim=0)

    # Prevent division by zero by adding a small epsilon
    epsilon = 1e-5
    std_pred = torch.where(std_pred == 0, torch.full_like(std_pred, epsilon), std_pred)
    std_target = torch.where(std_target == 0, torch.full_like(std_target, epsilon), std_target)

    # Normalize prediction and target
    prediction_norm = (prediction_flat - mean_pred) / std_pred
    target_norm = (target_flat - mean_target) / std_target

    # Compute MAE loss per channel
    mae_loss = torch.abs(prediction_norm - target_norm)  # Shape: [B*L*N, C] or [B*N, C]

    # Apply weights to MAE loss
    weighted_mae = mae_loss * torch.tensor(weights, dtype=mae_loss.dtype, device=mae_loss.device)

    # Sum the losses across all channels to get the total loss
    total_loss = weighted_mae.sum()

    return total_loss, mae_loss.sum(dim=0)


# 不确定性平方损失
def uncertainty_MSE_region(prediction, target, log_sigma,region_mask=region_mask):
    """
    Compute the total loss with uncertainty weighting (using squared error SE).

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        log_sigma (torch.Tensor): Log standard deviation for each task, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    """
    Args:
        prediction (Tensor): [B, L, N, 3] 预测值
        target (Tensor):     [B, L, N, 3] 真实值
        region_mask (Tensor): [N] 布尔掩码，True表示需要计算第二个观测值的区域
    Returns:
        torch.Tensor: 组合后的损失值
    """
    # 确保mask在正确的设备上
    region_mask = region_mask.to(prediction.device)

    # 分割三个观测维度
    pred_obs1 = prediction[..., 0]  # [B, L, N]
    pred_obs2 = prediction[..., 1]
    pred_obs3 = prediction[..., 2]

    target_obs1 = target[..., 0]
    target_obs2 = target[..., 1]
    target_obs3 = target[..., 2]

    # 计算第一个观测值的MSE（全区域）
    mse_obs1 = torch.mean(torch.square(pred_obs1 - target_obs1))

    # 计算第二个观测值的MSE（带区域掩码）
    masked_pred_obs2 = pred_obs2[:, :, region_mask]  # [B, L, valid_N]
    masked_target_obs2 = target_obs2[:, :, region_mask]
    mse_obs2 = torch.tensor(0.0, device=prediction.device)
    if masked_pred_obs2.numel() > 0:  # 避免空张量情况
        mse_obs2 = torch.mean(torch.square(masked_pred_obs2 - masked_target_obs2))

    # 计算第三个观测值的MSE（全区域）
    mse_obs3 = torch.mean(torch.square(pred_obs3 - target_obs3))

    # 组合最终损失（可根据需求调整权重）
    # Apply uncertainty weighting to each observation
    loss_obs1 = torch.exp(-log_sigma[0]) * mse_obs1 + log_sigma[0]
    loss_obs2 = torch.exp(-log_sigma[1]) * mse_obs2 + log_sigma[1]
    loss_obs3 = torch.exp(-log_sigma[2]) * mse_obs3 + log_sigma[2]

    # Combine losses
    total_loss = loss_obs1 + loss_obs2 + loss_obs3

    return total_loss

# 不确定性平方损失
def uncertainty_MSE_region_norms(prediction, target, log_sigma,region_mask=region_mask):
    """
    Compute the total loss with uncertainty weighting (using squared error SE).

    Args:
        prediction (torch.Tensor): Predicted values, shape [B, L, N, C] or [B, N, C].
        target (torch.Tensor): Ground truth values, shape [B, L, N, C] or [B, N, C].
        log_sigma (torch.Tensor): Log standard deviation for each task, shape [C].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure prediction and target have the same shape
    """
    Args:
        prediction (Tensor): [B, L, N, 3] 预测值
        target (Tensor):     [B, L, N, 3] 真实值
        region_mask (Tensor): [N] 布尔掩码，True表示需要计算第二个观测值的区域
    Returns:
        torch.Tensor: 组合后的损失值
    """
    # 确保mask在正确的设备上
    region_mask = region_mask.to(prediction.device)

    # 分割三个观测维度
    pred_obs1 = prediction[..., 0]  # [B, L, N]
    pred_obs2 = prediction[..., 1]
    pred_obs3 = prediction[..., 2]

    target_obs1 = target[..., 0]
    target_obs2 = target[..., 1]
    target_obs3 = target[..., 2]

    # 计算第一个观测值的MSE（全区域）
    mse_obs1 = torch.mean(torch.square(pred_obs1 - target_obs1))

    # 计算第二个观测值的MSE（带区域掩码）
    masked_pred_obs2 = pred_obs2[:, :, region_mask]  # [B, L, valid_N]
    masked_target_obs2 = target_obs2[:, :, region_mask]
    mse_obs2 = torch.tensor(0.0, device=prediction.device)
    if masked_pred_obs2.numel() > 0:  # 避免空张量情况
        mse_obs2 = torch.mean(torch.square(masked_pred_obs2 - masked_target_obs2))

    # 计算第三个观测值的MSE（全区域）
    mse_obs3 = torch.mean(torch.square(pred_obs3 - target_obs3))

    # 组合最终损失（可根据需求调整权重）
    # Apply uncertainty weighting to each observation
    loss_obs1 = torch.exp(-log_sigma[0]) * mse_obs1 + log_sigma[0]
    loss_obs2 = torch.exp(-log_sigma[1]) * mse_obs2 + log_sigma[1]
    loss_obs3 = torch.exp(-log_sigma[2]) * mse_obs3 + log_sigma[2]

    # Combine losses
    total_loss = loss_obs1 + loss_obs2 + loss_obs3

    return total_loss


def uncertainty_MSE_region_norms(prediction, target, log_sigma, region_mask=region_mask):
    """
    Compute normalized uncertainty-weighted MSE loss for each observation.

    Args:
        prediction (Tensor): [B, L, N, 3] Predicted values
        target (Tensor): [B, L, N, 3] Target values
        log_sigma (Tensor): [3] Log standard deviation for each task
        region_mask (Tensor): [N] Boolean mask for second observation
    Returns:
        torch.Tensor: Combined normalized loss value
    """
    # Move mask to correct device
    region_mask = region_mask.to(prediction.device)

    # Split observations
    pred_obs1 = prediction[..., 0]  # [B, L, N]
    pred_obs2 = prediction[..., 1]
    pred_obs3 = prediction[..., 2]

    target_obs1 = target[..., 0]
    target_obs2 = target[..., 1]
    target_obs3 = target[..., 2]

    # Calculate MSE for first observation (full region)
    mse_obs1 = torch.mean(torch.square(pred_obs1 - target_obs1))
    # Normalize by maximum target value
    max_target1 = torch.max(torch.abs(target_obs1))
    norm_mse_obs1 = mse_obs1 / (max_target1 ** 2 + 1e-8)

    # Calculate MSE for second observation (masked region)
    masked_pred_obs2 = pred_obs2[:, :, region_mask]
    masked_target_obs2 = target_obs2[:, :, region_mask]
    if masked_pred_obs2.numel() > 0:
        mse_obs2 = torch.mean(torch.square(masked_pred_obs2 - masked_target_obs2))
        max_target2 = torch.max(torch.abs(masked_target_obs2))
        norm_mse_obs2 = mse_obs2 / (max_target2 ** 2 + 1e-8)
    else:
        norm_mse_obs2 = torch.tensor(0.0, device=prediction.device)

    # Calculate MSE for third observation (full region)
    mse_obs3 = torch.mean(torch.square(pred_obs3 - target_obs3))
    max_target3 = torch.max(torch.abs(target_obs3))
    norm_mse_obs3 = mse_obs3 / (max_target3 ** 2 + 1e-8)

    # Apply uncertainty weighting to normalized losses
    loss_obs1 = torch.exp(-log_sigma[0]) * norm_mse_obs1 + log_sigma[0]
    loss_obs2 = torch.exp(-log_sigma[1]) * norm_mse_obs2 + log_sigma[1]
    loss_obs3 = torch.exp(-log_sigma[2]) * norm_mse_obs3 + log_sigma[2]

    # Combine normalized losses
    total_loss = loss_obs1 + loss_obs2 + loss_obs3

    return total_loss

def uncertainty_SE_region_norms(prediction, target, log_sigma, region_mask=region_mask):
    """
    Compute normalized uncertainty-weighted MSE loss for each observation.

    Args:
        prediction (Tensor): [B, L, N, 3] Predicted values
        target (Tensor): [B, L, N, 3] Target values
        log_sigma (Tensor): [3] Log standard deviation for each task
        region_mask (Tensor): [N] Boolean mask for second observation
    Returns:
        torch.Tensor: Combined normalized loss value
    """
    # Move mask to correct device
    region_mask = region_mask.to(prediction.device)

    # Split observations
    pred_obs1 = prediction[..., 0]  # [B, L, N]
    pred_obs2 = prediction[..., 1]
    pred_obs3 = prediction[..., 2]

    target_obs1 = target[..., 0]
    target_obs2 = target[..., 1]
    target_obs3 = target[..., 2]

    # Calculate MSE for first observation (full region)
    mse_obs1 = torch.mean(torch.square(pred_obs1 - target_obs1))
    # Normalize by maximum target value
    max_target1 = torch.max(torch.abs(target_obs1))
    norm_mse_obs1 = mse_obs1 / (max_target1 ** 2 + 1e-8)

    # Calculate MSE for second observation (masked region)
    masked_pred_obs2 = pred_obs2[:, :, region_mask]
    masked_target_obs2 = target_obs2[:, :, region_mask]
    if masked_pred_obs2.numel() > 0:
        mse_obs2 = torch.mean(torch.square(masked_pred_obs2 - masked_target_obs2))
        max_target2 = torch.max(torch.abs(masked_target_obs2))
        norm_mse_obs2 = mse_obs2 / (max_target2 ** 2 + 1e-8)
    else:
        norm_mse_obs2 = torch.tensor(0.0, device=prediction.device)

    # Calculate MSE for third observation (full region)
    mse_obs3 = torch.mean(torch.square(pred_obs3 - target_obs3))
    max_target3 = torch.max(torch.abs(target_obs3))
    norm_mse_obs3 = mse_obs3 / (max_target3 ** 2 + 1e-8)

    # Apply uncertainty weighting to normalized losses
    loss_obs1 = torch.exp(-log_sigma[0]) * norm_mse_obs1 + log_sigma[0]
    loss_obs2 = torch.exp(-log_sigma[1]) * norm_mse_obs2 + log_sigma[1]
    loss_obs3 = torch.exp(-log_sigma[2]) * norm_mse_obs3 + log_sigma[2]

    # Combine normalized losses
    total_loss = loss_obs1 + loss_obs2 + loss_obs3

    return total_loss