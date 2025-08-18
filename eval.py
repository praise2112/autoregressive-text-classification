"""
Evaluation script for autoregressive classification models.

Usage:
    python eval.py --config configs/emotion_full_finetune_fair.yaml --model-path runs/emotion_full_finetune_fair/final_model
"""

import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config.schema import ExperimentConfig
from src.data.templates import ClassificationTemplate
from src.evaluation.metrics import GenerativeClassificationEvaluator


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return ExperimentConfig(**config_data)


def main():
    parser = argparse.ArgumentParser(description="Evaluate autoregressive classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    dataset = load_dataset(config.data.dataset_name)
    eval_data = dataset[args.split]
    
    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    
    if isinstance(eval_data[config.data.label_column][0], int):
        label_names = eval_data.features[config.data.label_column].names
        id2label = {i: label for i, label in enumerate(label_names)}
        true_labels = [id2label[label_id] for label_id in eval_data[config.data.label_column]]
    else:
        true_labels = eval_data[config.data.label_column]
        unique_labels = list(set(true_labels))
        id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
    
    labels = list(id2label.values())
    template = ClassificationTemplate(
        tokenizer=tokenizer,
        labels=labels,
        template_format=config.template.template_format,
        label_prefix=config.template.label_prefix,
        use_chat_template=config.template.use_chat_template
    )
    
    evaluator = GenerativeClassificationEvaluator(
        model=model,
        tokenizer=tokenizer,
        template=template
    )
    
    texts = eval_data[config.data.text_column]
    
    print(f"Evaluating {len(texts)} samples...")
    metrics, predicted_labels = evaluator.evaluate_dataset(texts, true_labels, batch_size=16)
    
    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Detailed Classification Report ===")
    print(evaluator.detailed_report(true_labels, predicted_labels))
    
    print(f"\nTarget F1 (DeBERTa baseline): 0.8989")
    print(f"Current F1: {metrics['f1']:.4f}")
    print(f"Improvement: {'+' if metrics['f1'] > 0.8989 else ''}{metrics['f1'] - 0.8989:.4f}")


if __name__ == "__main__":
    main()