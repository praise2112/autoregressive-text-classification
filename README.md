# Autoregressive Supervised Fine-Tuning for Text Classification

A research implementation exploring whether autoregressive supervised fine-tuning (SFT) can outperform traditional encoder-based approaches for text classification tasks.

## Key Results

Weighted F1 on the test split of [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (6 labels).

| Model | How it classifies | F1 |
|---|---|---|
| DeBERTa (baseline) | encoder with a classification head | 0.8989 |
| Gemma 3 270M | classification head on the same model | 0.8972 |
| **Gemma 3 270M, full fine-tuning** | **generates the label** | **0.8863** |
| Gemma 3 270M, LoRA (r=32) | generates the label | 0.8774 |

Generating the label with `google/gemma-3-270m-it` reaches 98.6% of the DeBERTa baseline's F1. Full fine-tuning beat LoRA on this task.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python main.py --config configs/emotion_full_finetune_fair.yaml
```

### Evaluation

```bash
python eval.py --config configs/emotion_full_finetune_fair.yaml --model-path runs/emotion_full_finetune_fair/final_model
```

## Project Structure

```
├── src/
│   ├── config/          # Configuration schema
│   ├── data/            # Data preprocessing and templates
│   ├── evaluation/      # Evaluation metrics
│   └── training/        # Training logic
├── configs/             # YAML configuration files
├── main.py              # Training script
├── eval.py              # Evaluation script
└── runs/                # Model outputs
```

## Configuration

Key configs:
- `emotion_full_finetune_fair.yaml` - Best performing configuration (F1: 0.8863)
- `emotion_full_finetune.yaml` - Alternative full fine-tuning setup
- `emotion_optimized_v1.yaml` - PEFT/LoRA configuration

## Research Notes

This implementation uses:
- TRL (Transformers Reinforcement Learning) for autoregressive SFT
- Chat template integration for instruction formatting
- Completion-only loss for efficient training
- Full fine-tuning outperforms PEFT for this classification task

## Limitations

- One dataset and one seed per configuration, so small F1 differences are within noise.
- Decoding a label costs more than one forward pass through an encoder with a classification head.

Run notes are in `PROGRESS.md`.