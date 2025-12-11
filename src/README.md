# Controller Model Implementation

A PEFT-style implementation for adding adaptive controllers to pre-trained language models. This implementation allows you to add lightweight, trainable controllers to existing models without modifying the base model parameters.

## Features

- **PEFT-style API**: Similar to popular PEFT libraries like LoRA
- **Lightweight Controllers**: Only trains a small number of additional parameters
- **Flexible Intervention**: Support for selective intervention using masks
- **Easy Integration**: Works with Hugging Face Transformers and Trainer
- **Save/Load**: Easy checkpoint management for controllers

## Installation

```bash
# Install required dependencies
pip install torch transformers datasets accelerate
```

## Quick Start

### 1. Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.controller_model import ControllerConfig, get_controller_model

# Load base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create controller configuration
controller_config = ControllerConfig(
    controller_type="mlp_adaptive",
    target_layer=7,
    intermediate_size=4
)

# Get controller model
model = get_controller_model(base_model, controller_config)

# The model is now ready for training or inference!
```

### 2. Training

```bash
# Basic training
python src/training/train_controller.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --controller_type mlp_adaptive \
    --target_layer 7 \
    --intermediate_size 4 \
    --output_dir ./checkpoints/my_controller \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --controller_learning_rate 1e-3

# See more examples in src/scripts/train_controller_examples.sh
```

### 3. Inference

```bash
# Compare outputs with and without controller
python src/scripts/inference_controller.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --controller_path ./checkpoints/my_controller \
    --prompt "The future of artificial intelligence is" \
    --compare
```

## Architecture

### MLP Adaptive Controller

The MLP Adaptive Controller consists of:

- **Rotation Matrix**: Orthogonal matrix for dimensionality reduction
- **Learned Source**: Linear transformation for learning modifications
- **Gate Projection**: Controls the strength of intervention
- **Orthogonal Loss**: Regularization to maintain orthogonality

```
Input Hidden States → Rotation → Gate Computation
                   ↓
                   Learned Source → Delta Computation
                   ↓
Output = Input + Gate * Delta
```

## Configuration

### ControllerConfig

```python
@dataclass
class ControllerConfig:
    controller_type: str = "mlp_adaptive"  # Type of controller
    target_layer: int = 7                  # Layer to intervene on
    hidden_size: Optional[int] = None      # Hidden size (auto-detected)
    intermediate_size: int = 4             # Size of intermediate representation
```

### Training Arguments

Key training parameters:

- `controller_learning_rate`: Learning rate for controller parameters (default: 1e-3)
- `orthogonal_loss_weight`: Weight for orthogonal regularization (default: 1e-4)
- `intervention_probability`: Probability of applying intervention during training (default: 0.5)

## Advanced Usage

### Selective Intervention with Masks

```python
# Create intervention mask
batch_size, seq_len = inputs["input_ids"].shape
intervention_mask = torch.zeros(batch_size, seq_len)
intervention_mask[:, 10:20] = 1  # Only intervene on positions 10-20

# Forward pass with selective intervention
outputs = model(**inputs, intervention_mask=intervention_mask)
```

### Save and Load Controllers

```python
# Save controller
model.save_controller("./my_controller")

# Load controller
model.load_controller("./my_controller")
```

### Enable/Disable Controller

```python
# Disable controller (use base model)
model.disable_controller()

# Enable controller
model.enable_controller()
```

## File Structure

```
src/
├── model/
│   └── controller_model.py          # Main controller implementation
├── training/
│   └── train_controller.py          # Training script
├── scripts/
│   ├── train_controller_examples.sh # Training examples
│   └── inference_controller.py      # Inference script
└── examples/
    └── controller_usage_examples.py # Usage examples
```

## Examples

### Training Examples

```bash
# Run basic examples
bash src/scripts/train_controller_examples.sh

# Custom training
python src/training/train_controller.py \
    --model_name_or_path your-model \
    --output_dir ./checkpoints \
    --controller_type mlp_adaptive \
    --target_layer 12 \
    --intermediate_size 8 \
    --num_train_epochs 5
```

### Inference Examples

```bash
# Basic inference
python src/scripts/inference_controller.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --controller_path ./checkpoints/my_controller \
    --prompt "Your prompt here"

# Compare with and without controller
python src/scripts/inference_controller.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --controller_path ./checkpoints/my_controller \
    --prompt "Your prompt here" \
    --compare
```

## API Reference

### Main Functions

- `get_controller_model(base_model, controller_config)`: Create a controller model
- `prepare_model_for_controller_training(model)`: Prepare model for training

### Model Methods

- `model.enable_controller()`: Enable controller intervention
- `model.disable_controller()`: Disable controller intervention
- `model.save_controller(path)`: Save controller state and config
- `model.load_controller(path)`: Load controller state and config
- `model.freeze_base_model()`: Freeze base model parameters

### Training

- `ControllerTrainer`: Custom trainer with orthogonal loss support
- `ControllerTrainingArguments`: Extended training arguments

## Tips and Best Practices

1. **Layer Selection**: Middle layers (around layer 7-12 for most models) often work best
2. **Learning Rate**: Start with 1e-3 for controller parameters
3. **Intervention Probability**: Use 0.5 during training for balanced learning
4. **Orthogonal Loss**: Adjust weight based on model size (1e-4 to 1e-3)
5. **Batch Size**: Smaller batch sizes often work better for controller training

## Troubleshooting

### Common Issues

1. **CUDA Memory**: Reduce batch size or use gradient accumulation
2. **Convergence**: Try different target layers or learning rates
3. **Overfitting**: Increase orthogonal loss weight or reduce intervention probability

### Performance Tips

1. Use `torch.float16` for memory efficiency
2. Enable gradient checkpointing for large models
3. Use `device_map="auto"` for multi-GPU setups

## Contributing

To extend this implementation:

1. Add new controller types in `controller_model.py`
2. Extend `get_controller_model` to support new model types
3. Add custom loss functions in `ControllerTrainer`

## License

This implementation follows the same license as the base model being used.
