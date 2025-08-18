from typing import List, Optional
from transformers import AutoTokenizer


class ClassificationTemplate:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        labels: List[str],
        template_format: str = "Classify the following text as one of {labels}: {text}\n{label_prefix}",
        label_prefix: str = "Emotion:",
        use_chat_template: bool = True
    ):
        self.tokenizer = tokenizer
        self.labels = labels
        self.template_format = template_format
        self.label_prefix = label_prefix
        self.use_chat_template = use_chat_template
        self.label_set = set(labels)
    
    def format_prompt(self, text: str) -> str:
        labels_str = ", ".join(self.labels)
        return self.template_format.format(
            labels=f"[{labels_str}]",
            text=text,
            label_prefix=self.label_prefix
        )
    
    def format_example(self, text: str, label: str) -> str:
        prompt = self.format_prompt(text)
        
        if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label}
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            return f"{prompt} {label}"
    
    def extract_label(self, generated_text: str) -> Optional[str]:
        """Extract label from generated text - handles both 'sadness' and 'Emotion: sadness' formats"""
        generated_text = generated_text.strip()
        
        # First, look for text after label_prefix (traditional format)
        if self.label_prefix in generated_text:
            after_prefix = generated_text.split(self.label_prefix, 1)[1].strip()
            first_word = after_prefix.split()[0] if after_prefix.split() else ""
            clean_word = first_word.strip('.,!?;:"()[]{}').lower()
            for label in self.labels:
                if clean_word == label.lower():
                    return label
        
        # If no prefix found, check if the generated text itself is a label
        # This handles chat template format where model generates just "sadness"
        clean_generated = generated_text.strip('.,!?;:"()[]{}').lower()
        for label in self.labels:
            if clean_generated == label.lower():
                return label
        
        return None
    
    def is_valid_label(self, label: str) -> bool:
        return label in self.label_set