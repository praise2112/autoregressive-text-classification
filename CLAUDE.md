# Autoregressive Supervised Fine-Tuning for Text Classification

## Repository Purpose

This repository explores whether **autoregressive supervised fine-tuning (SFT)** can outperform traditional encoder-based approaches for text classification tasks. Instead of using encoder models with classification heads, we train autoregressive language models to generate classification labels as text tokens.

## Research Motivation

Current encoder-based approach (from encoder_playground):
- **DeBERTa**: F1 score of 0.8989 on emotion classification
- **Gemma-3-270m** (with classification head): F1 score of 0.89723

**Research Question**: Can generative autoregressive fine-tuning achieve better performance than encoder+head approaches?

## Key Design Decisions

### 1. Generative Text Classification
- Models generate label tokens directly (e.g., "anger", "joy") rather than using classification heads
- Uses instruction-based templates for consistent formatting
- Supports exact match evaluation for generated labels

### 2. Template Format
```
Classify the following text as one of [sadness, joy, love, anger, fear, surprise]: {input_text}
Emotion:
```

### 3. Chat Template Integration
- Leverages model's built-in chat templates when available (e.g., gemma-3-270m-it)
- Maintains consistent instruction format across different model architectures
- Supports both base and instruction-tuned models

### 4. Training Framework
- **TRL (Transformers Reinforcement Learning)** for autoregressive SFT
- Future support for importance-weighted SFT (iw-SFT)
- Compatible with HuggingFace ecosystem

## Target Models
- Primary: `google/gemma-3-270m-it`
- Secondary: Qwen, Llama, and other instruction-tuned models
- Architecture-agnostic design for broad compatibility

## Evaluation Metrics
- **Standard**: F1, accuracy, precision, recall
- **Generation-specific**: Exact match rate, valid label rate
- **Optional**: Text generation quality metrics (BLEU/ROUGE)

## Current Status

### ✅ **Completed Implementation**
- **End-to-end pipeline**: Data preprocessing → TRL training → Evaluation
- **Chat template integration**: Proper Gemma-3 chat format handling
- **Label extraction**: Handles both "sadness" and "Emotion: sadness" formats
- **TRL 0.21.0**: Correct API usage with completion-only loss

### 📊 **Performance Results (1 epoch)**
- **Model**: google/gemma-3-270m-it with PEFT/LoRA
- **F1 Score**: 0.6831 (target: 0.8989)
- **Accuracy**: 69.85%
- **Valid Label Rate**: 99.95%
- **Gap to baseline**: -0.22 F1 points

### 🔍 **Performance Gap Analysis**
The system works correctly but underperforms. Potential causes:
1. **Model capacity**: 270M parameters may be insufficient
2. **Training duration**: Only 1 epoch vs encoder fine-tuning
3. **Optimization target**: Language modeling loss vs direct F1 optimization
4. **Hyperparameters**: Learning rate/batch size not optimized for this task

## Future Enhancements
1. **Importance-weighted SFT**: Advanced optimization bridging SFT and RL
2. **Multi-label classification**: Support for multiple simultaneous labels
3. **CAALM integration**: Combining autoregressive and encoder approaches
4. **Half Fine-tuning**: Preserving general capabilities during specialization

---

*This project builds upon insights from the encoder_playground repository and incorporates latest 2024-2025 research in autoregressive fine-tuning.*