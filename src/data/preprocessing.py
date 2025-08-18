from typing import Dict
from datasets import Dataset, DatasetDict, load_dataset

from .templates import ClassificationTemplate


class DataPreprocessor:
    def __init__(self, template: ClassificationTemplate):
        self.template = template
    
    def load_dataset(
        self,
        dataset_name: str,
        text_column: str = "text",
        label_column: str = "label",
        train_split: str = "train",
        test_split: str = "test",
        validation_split: str = None
    ) -> tuple[DatasetDict, Dict]:
        dataset = load_dataset(dataset_name)
        
        if isinstance(dataset[train_split][label_column][0], int):
            label_names = dataset[train_split].features[label_column].names
            id2label = {i: label for i, label in enumerate(label_names)}
        else:
            unique_labels = list(set(dataset[train_split][label_column]))
            id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
        
        def map_labels(example):
            if isinstance(example[label_column], int):
                example["label_text"] = id2label[example[label_column]]
            else:
                example["label_text"] = example[label_column]
            return example
        
        processed_splits = {}
        for split_name in [train_split, test_split, validation_split]:
            if split_name and split_name in dataset:
                split_data = dataset[split_name].map(map_labels)
                processed_splits[split_name] = split_data
        
        return DatasetDict(processed_splits), id2label
    
    def format_examples(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        def format_fn(examples):
            formatted_texts = []
            for text, label in zip(examples[text_column], examples["label_text"]):
                # Format the full example (prompt + label)
                formatted_text = self.template.format_example(text, label)
                formatted_texts.append(formatted_text)
            
            return {"text": formatted_texts}  # Changed from formatted_text to text for TRL
        
        return dataset.map(format_fn, batched=True)
    
    def prepare_datasets(
        self,
        dataset_name: str,
        text_column: str = "text",
        label_column: str = "label",
        train_split: str = "train",
        test_split: str = "test",
        validation_split: str = None
    ) -> tuple[DatasetDict, Dict]:
        dataset_dict, id2label = self.load_dataset(
            dataset_name, text_column, label_column, 
            train_split, test_split, validation_split
        )
        
        processed_dict = {}
        for split_name, split_data in dataset_dict.items():
            processed_dict[split_name] = self.format_examples(split_data, text_column)
        
        return DatasetDict(processed_dict), id2label