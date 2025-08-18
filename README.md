# Autoregressive Supervised Fine-Tuning for Text Classification

A research implementation exploring whether autoregressive supervised fine-tuning (SFT) can outperform traditional encoder-based approaches for text classification tasks.

## Key Results

- **F1 Score**: 0.8863 (98.6% of DeBERTa baseline: 0.8989)
- **Model**: google/gemma-3-270m-it with full fine-tuning
- **Approach**: Generative text classification using chat templates

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

For detailed progress and analysis, see `CLAUDE.md` and `PROGRESS.md`.