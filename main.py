"""
Main training script for autoregressive supervised fine-tuning.

Usage:
    python main.py --config configs/emotion_full_finetune_fair.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.config.schema import ExperimentConfig
from src.training.trainer import AutoregressiveSFTTrainer


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return ExperimentConfig(**config_data)


def main():
    parser = argparse.ArgumentParser(description="Train autoregressive SFT for text classification")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    torch.manual_seed(config.seed)
    
    trainer = AutoregressiveSFTTrainer(config)
    
    if args.eval_only:
        print("Evaluation mode not implemented yet")
    else:
        trainer.train()
        
        output_path = Path(config.training.output_dir) / "final_model"
        trainer.save_model(str(output_path))
        print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()