import torch
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union
)

# Inspired by Grokking at the Edge of Numerical Stability (https://arxiv.org/abs/2501.04697)
def mse_loss(predictions, targets, eps=1e-37):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    squared_differences = differences ** 2

    # Add eps to address underflows due to squaring
    loss = squared_differences.add(eps)
    return loss

def smooth_l2_loss(predictions, targets, beta=1.0, eps=1e-37):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = ((differences**2) / differences.abs().add(beta)).pow(2)

    # Add eps to address underflows due to squaring
    loss = loss.add(eps)
    return loss

def pseudo_huber_loss(predictions, targets, delta=1.0, eps: float = 1e-37):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)

    # Compute the loss
    loss = delta**2 * (torch.sqrt(1 + (differences / delta)**2 + eps) - 1)
    return loss

def scaled_quadratic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
    eps: float = 1e-37,
) -> torch.Tensor:
    r = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = (r / delta)**2

    # Add eps to address underflows due to squaring
    loss = loss.add(eps)
    return loss

def smooth_l1_loss(predictions, targets, beta=1.0, eps=1e-37):
    diff = torch.abs(predictions - targets)
    condition = diff < beta
    
    # Where diff < beta, use quadratic form
    quadratic = 0.5 * diff.pow(2) / beta

    # Add eps to address underflows due to squaring
    loss = quadratic.add(eps)
    
    # Where diff >= beta, use linear form
    linear = diff - 0.5 * beta
    
    # Combine the two parts based on the condition
    loss = torch.where(condition, quadratic, linear)
    
    return loss


def huber_loss(predictions, targets, delta=1.0, eps=1e-37):
    diff = torch.abs(predictions - targets)
    abs_error = torch.abs(diff)
    
    # For small errors (≤ delta): use squared error (L2)
    quadratic = 0.5 * diff.pow(2) + eps
    
    # For large errors (> delta): use modified absolute error (L1)
    linear = delta * (abs_error - 0.5 * delta)
    
    # Combine both parts
    loss = torch.where(abs_error <= delta, quadratic, linear)

    return loss

def l1_loss(predictions, targets):
    loss = torch.abs(predictions - targets)
    return loss


def conditional_loss(
    model_pred: torch.Tensor, 
    target: torch.Tensor, 
    loss_type: str = "l2", 
    delta_beta: float = None,
):
    loss_type_lower = loss_type.lower()

    eps = torch.finfo(torch.float32).tiny

    # Make loss float64
    model_pred = model_pred.to(torch.float64)
    target = target.to(torch.float64)

    if loss_type_lower == "l2":
        loss = mse_loss(model_pred, target, eps=eps)
    elif loss_type == "l1":
        loss = l1_loss(model_pred, target, eps=eps)
    elif loss_type == "pseudo_huber":
        loss = pseudo_huber_loss(model_pred, target, delta=delta_beta, eps=eps)
    elif loss_type == "huber":
        loss = huber_loss(model_pred, target, delta=delta_beta, eps=eps)
    elif loss_type == "smooth_l1":
        loss = smooth_l1_loss(model_pred, target, beta=delta_beta, eps=eps)
    elif loss_type == "scaled_quadratic":
        loss = scaled_quadratic_loss(model_pred, target, delta=delta_beta, eps=eps)
    elif loss_type == "smooth_l2":
        loss = smooth_l2_loss(model_pred, target, delta=delta_beta, eps=eps)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss
