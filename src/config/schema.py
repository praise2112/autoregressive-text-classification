from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_name: str = Field(description="HuggingFace dataset name")
    text_column: str = Field(default="text", description="Column containing input text")
    label_column: str = Field(default="label", description="Column containing labels")
    train_split: str = Field(default="train", description="Training split name")
    test_split: str = Field(default="test", description="Test split name")
    validation_split: Optional[str] = Field(default=None, description="Validation split name")


class ModelConfig(BaseModel):
    model_name: str = Field(description="HuggingFace model name")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    torch_dtype: Optional[str] = Field(default=None, description="Model dtype")


class TrainingConfig(BaseModel):
    output_dir: str = Field(description="Output directory for checkpoints")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=8, description="Training batch size")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    warmup_steps: int = Field(default=100, description="Warmup steps")
    logging_steps: int = Field(default=50, description="Logging frequency")
    save_steps: int = Field(default=500, description="Save checkpoint frequency")
    eval_steps: int = Field(default=500, description="Evaluation frequency")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm")
    bf16: bool = Field(default=True, description="Use bfloat16")
    use_peft: bool = Field(default=False, description="Use PEFT/LoRA")


class TemplateConfig(BaseModel):
    template_format: str = Field(
        default="Classify the following text as one of {labels}: {text}\n{label_prefix}",
        description="Template format string"
    )
    label_prefix: str = Field(default="Emotion:", description="Prefix before label")
    use_chat_template: bool = Field(default=True, description="Use model's chat template")


class ExperimentConfig(BaseModel):
    name: str = Field(description="Experiment name")
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    seed: int = Field(default=42, description="Random seed")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")