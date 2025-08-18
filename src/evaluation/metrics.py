import torch
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..data.templates import ClassificationTemplate


class GenerativeClassificationEvaluator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        template: ClassificationTemplate,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        self.device = device
        
    def generate_predictions(
        self,
        texts: List[str],
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        top_p: float = 0.9,
        batch_size: int = 8
    ) -> List[str]:
        """Generate predictions with batched inference for efficiency"""
        predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_prompts = []
            
            for text in batch_texts:
                prompt = self.template.format_prompt(text)
                
                if self.template.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt
                
                batch_prompts.append(formatted_prompt)
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Generate batch predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode batch outputs
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_text = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                )
                
                predicted_label = self.template.extract_label(generated_text)
                predictions.append(predicted_label)
        
        return predictions
    
    def calculate_metrics(
        self,
        true_labels: List[str],
        predicted_labels: List[str]
    ) -> Dict[str, float]:
        valid_predictions = []
        valid_true_labels = []
        invalid_count = 0
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            if pred_label is not None and self.template.is_valid_label(pred_label):
                valid_predictions.append(pred_label)
                valid_true_labels.append(true_label)
            else:
                invalid_count += 1
                valid_predictions.append(true_labels[0])  # Use first label as fallback
                valid_true_labels.append(true_label)
        
        exact_matches = sum(1 for true, pred in zip(true_labels, predicted_labels) 
                          if pred == true)
        exact_match_rate = exact_matches / len(true_labels)
        
        valid_label_rate = (len(true_labels) - invalid_count) / len(true_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_true_labels, valid_predictions, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_rate": exact_match_rate,
            "valid_label_rate": valid_label_rate,
            "invalid_predictions": invalid_count
        }
    
    def evaluate_dataset(
        self,
        texts: List[str],
        true_labels: List[str],
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        top_p: float = 0.9,
        batch_size: int = 8
    ) -> Tuple[Dict[str, float], List[str]]:
        predicted_labels = self.generate_predictions(
            texts, max_new_tokens, temperature, top_p, batch_size
        )
        
        metrics = self.calculate_metrics(true_labels, predicted_labels)
        
        return metrics, predicted_labels
    
    def detailed_report(
        self,
        true_labels: List[str],
        predicted_labels: List[str]
    ) -> str:
        valid_predictions = []
        valid_true_labels = []
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            if pred_label is not None and self.template.is_valid_label(pred_label):
                valid_predictions.append(pred_label)
                valid_true_labels.append(true_label)
            else:
                valid_predictions.append("INVALID")
                valid_true_labels.append(true_label)
        
        return classification_report(
            valid_true_labels, valid_predictions, 
            labels=self.template.labels + ["INVALID"]
        )