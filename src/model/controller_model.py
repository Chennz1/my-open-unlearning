"""
Controller Model Implementation - PEFT-style approach for adding adaptive controllers to LLaMA models.

This module provides a PEFT-like interface for adding lightweight adaptive controllers
to pre-trained language models, specifically designed for steering model behavior.
"""

import os
import json
import torch
from torch import nn
from typing import Optional, Dict, Any, Union
from transformers import PreTrainedModel, LlamaForCausalLM
from dataclasses import dataclass, asdict


def ortho_loss(matrix):
    """Compute orthogonal regularization loss for a matrix."""
    return torch.sum((matrix.T @ matrix - torch.eye(matrix.size(1), device=matrix.device))**2)


@dataclass
class ControllerConfig:
    """Configuration class for controller models."""
    controller_type: str = "mlp_adaptive"
    target_layer: int = 7
    hidden_size: Optional[int] = None  # Will be set from base model if None
    intermediate_size: int = 4
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class MLPAdaptiveController(nn.Module):
    """
    MLP-based adaptive controller that learns to modify hidden states.
    """
    
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Orthogonal rotation matrix for dimensionality reduction
        self.rotate_layer = nn.Parameter(
            torch.nn.init.orthogonal_(torch.randn(self.hidden_size, self.intermediate_size))
        )
        
        # Learned transformation
        self.learned_source = nn.Linear(self.hidden_size, self.intermediate_size)
        
        # Gate projection for controlling intervention strength
        self.gate_proj = nn.Linear(self.intermediate_size, 1, bias=False)
        
    def forward(self, hidden_states, intervention_mask=None):
        """
        Apply adaptive control to hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            intervention_mask: [batch_size, seq_len] - binary mask for selective intervention
            
        Returns:
            modified_hidden_states: [batch_size, seq_len, hidden_size]
            gate_values: scalar - average gate activation
        """
        # Project to intermediate space
        rotated_hidden = hidden_states @ self.rotate_layer  # [batch_size, seq_len, intermediate_size]
        
        # Compute gate values
        gate = torch.sigmoid(self.gate_proj(rotated_hidden))  # [batch_size, seq_len, 1]
        
        # Compute modification delta
        delta = torch.matmul(
            (self.learned_source(hidden_states) - rotated_hidden), 
            self.rotate_layer.T
        )  # [batch_size, seq_len, hidden_size]
        
        # Apply intervention mask if provided
        if intervention_mask is not None:
            mask = intervention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [batch_size, seq_len, 1]
            delta = delta * mask
            gate = gate * mask
        
        # Apply gated modification
        modified_hidden_states = hidden_states + gate * delta
        
        # Compute orthogonal loss for regularization
        self.ortho_loss = ortho_loss(self.rotate_layer)
        
        return modified_hidden_states, gate.mean()


class ControllerModelMixin:
    """
    Mixin class that adds controller functionality to any transformer model.
    """
    
    def __init_controller__(self, controller_config: ControllerConfig):
        """Initialize controller components."""
        self.controller_config = controller_config
        self.controller_enabled = True
        
        # Set hidden_size from base model if not specified
        if controller_config.hidden_size is None:
            controller_config.hidden_size = self.config.hidden_size
        
        # Create controller
        if controller_config.controller_type == "mlp_adaptive":
            self.controller = MLPAdaptiveController(controller_config)
        else:
            raise ValueError(f"Unsupported controller type: {controller_config.controller_type}")
        
        # Move to same device and dtype as base model
        if hasattr(self, 'device'):
            self.controller = self.controller.to(self.device)
        if hasattr(self, 'dtype'):
            self.controller = self.controller.to(self.dtype)
    
    def enable_controller(self):
        """Enable controller intervention."""
        self.controller_enabled = True
        return self
    
    def disable_controller(self):
        """Disable controller intervention."""
        self.controller_enabled = False
        return self
    
    def get_controller_parameters(self):
        """Get controller parameters for training."""
        if hasattr(self, 'controller'):
            return self.controller.parameters()
        return []
    
    def freeze_base_model(self):
        """Freeze base model parameters, only allow controller training."""
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.get_controller_parameters():
            param.requires_grad = True
        
        return self
    
    def save_controller(self, save_directory: str):
        """Save controller state and configuration."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save controller state dict
        torch.save(
            self.controller.state_dict(), 
            os.path.join(save_directory, "controller.pt")
        )
        
        # Save controller configuration
        config_dict = self.controller_config.to_dict()
        with open(os.path.join(save_directory, "controller_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Controller saved to {save_directory}")
    
    def load_controller(self, load_directory: str):
        """Load controller state and configuration."""
        # Load configuration
        config_path = os.path.join(load_directory, "controller_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Controller config not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        controller_config = ControllerConfig.from_dict(config_dict)
        
        # Initialize controller
        self.__init_controller__(controller_config)
        
        # Load state dict
        state_path = os.path.join(load_directory, "controller.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Controller state not found: {state_path}")
        
        state_dict = torch.load(state_path, map_location='cpu')
        self.controller.load_state_dict(state_dict)
        
        # Move to correct device and dtype
        if hasattr(self, 'device'):
            self.controller = self.controller.to(self.device)
        if hasattr(self, 'dtype'):
            self.controller = self.controller.to(self.dtype)
        
        print(f"Controller loaded from {load_directory}")


class ControllerLlamaForCausalLM(LlamaForCausalLM, ControllerModelMixin):
    """
    LlamaForCausalLM with controller intervention capability.
    """
    
    def __init__(self, config, controller_config: Optional[ControllerConfig] = None):
        super().__init__(config)
        
        if controller_config is not None:
            self.__init_controller__(controller_config)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        intervention_mask=None,
        **kwargs
    ):
        """Forward pass with optional controller intervention."""
        
        # Remove custom parameters from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['intervention_mask']}
        
        # If controller is disabled, use original forward
        if not hasattr(self, 'controller') or not self.controller_enabled:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **filtered_kwargs
            )
        
        # Store original layer forward
        target_layer = self.model.layers[self.controller_config.target_layer]
        original_forward = target_layer.forward
        
        # Define modified forward with controller intervention
        def controlled_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            hidden_states = outputs[0]
            
            # Apply controller intervention
            modified_hidden_states, gate_values = self.controller(
                hidden_states, intervention_mask
            )
            
            return (modified_hidden_states,) + outputs[1:] if len(outputs) > 1 else (modified_hidden_states,)
        
        # Temporarily replace layer forward
        target_layer.forward = controlled_forward
        
        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **filtered_kwargs
            )
        finally:
            # Restore original forward
            target_layer.forward = original_forward
        
        return outputs
    
    def generate(self, *args, intervention_mask=None, **kwargs):
        """Generate with controller intervention."""
        
        # Remove custom parameters from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['intervention_mask']}
        
        # If controller is disabled, use original generate
        if not hasattr(self, 'controller') or not self.controller_enabled:
            return super().generate(*args, **filtered_kwargs)
        
        # Store original layer forward
        target_layer = self.model.layers[self.controller_config.target_layer]
        original_forward = target_layer.forward
        
        # Define modified forward with controller intervention
        def controlled_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            hidden_states = outputs[0]
            
            # Only apply intervention during prompt processing (seq_len > 1)
            if hidden_states.shape[1] > 1:
                modified_hidden_states, gate_values = self.controller(
                    hidden_states, intervention_mask
                )
                return (modified_hidden_states,) + outputs[1:] if len(outputs) > 1 else (modified_hidden_states,)
            else:
                return outputs
        
        # Temporarily replace layer forward
        target_layer.forward = controlled_forward
        
        try:
            return super().generate(*args, **filtered_kwargs)
        finally:
            # Restore original forward
            target_layer.forward = original_forward


def get_controller_model(
    base_model: PreTrainedModel,
    controller_config: ControllerConfig,
    adapter_name: str = "default"
) -> PreTrainedModel:
    """
    Get a model with controller intervention capability.
    
    Args:
        base_model: The base pre-trained model
        controller_config: Configuration for the controller
        adapter_name: Name for the adapter (for future multi-adapter support)
    
    Returns:
        Model with controller intervention capability
    """
    
    # Check if base model is supported
    if not isinstance(base_model, LlamaForCausalLM):
        raise ValueError(f"Base model type {type(base_model)} is not supported. Only LlamaForCausalLM is supported.")
    
    # Create controller model
    controller_model = ControllerLlamaForCausalLM(
        config=base_model.config,
        controller_config=controller_config
    )
    
    # Copy weights from base model
    controller_model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Move to same device as base model
    controller_model = controller_model.to(base_model.device)
    
    # Set up for training (freeze base model by default)
    controller_model.freeze_base_model()
    
    print(f"Controller model created with {controller_config.controller_type} controller at layer {controller_config.target_layer}")
    
    return controller_model


def prepare_model_for_controller_training(model):
    """
    Prepare model for controller training by setting up parameter groups.
    
    Args:
        model: Controller model
        
    Returns:
        Parameter groups for optimizer
    """
    
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable controller parameters
    controller_params = []
    for param in model.get_controller_parameters():
        param.requires_grad = True
        controller_params.append(param)
    
    print(f"Prepared {len(controller_params)} controller parameters for training")
    
    return [{"params": controller_params}]
