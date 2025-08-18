import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
import wandb

from ..config.schema import ExperimentConfig
from ..data.templates import ClassificationTemplate
from ..data.preprocessing import DataPreprocessor


class AutoregressiveSFTTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.template = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": getattr(torch, self.config.model.torch_dtype) if self.config.model.torch_dtype else None,
            "device_map": "auto"
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            **{k: v for k, v in model_kwargs.items() if v is not None}
        )
        
        if self.config.training.use_peft:
            peft_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_config)
    
    def setup_template(self, labels):
        self.template = ClassificationTemplate(
            tokenizer=self.tokenizer,
            labels=labels,
            template_format=self.config.template.template_format,
            label_prefix=self.config.template.label_prefix,
            use_chat_template=self.config.template.use_chat_template
        )
    
    def prepare_data(self) -> DatasetDict:
        preprocessor = DataPreprocessor(template=self.template)
        
        dataset_dict, self.id2label = preprocessor.prepare_datasets(
            dataset_name=self.config.data.dataset_name,
            text_column=self.config.data.text_column,
            label_column=self.config.data.label_column,
            train_split=self.config.data.train_split,
            test_split=self.config.data.test_split,
            validation_split=self.config.data.validation_split
        )
        
        return dataset_dict
    
    def setup_trainer(self, train_dataset, eval_dataset=None):
        # Calculate warmup steps if warmup_ratio is provided
        total_steps = len(train_dataset) * self.config.training.num_epochs // (
            self.config.training.batch_size * self.config.training.gradient_accumulation_steps
        )
        
        warmup_steps = self.config.training.warmup_steps
        if hasattr(self.config.training, 'warmup_ratio') and self.config.training.warmup_ratio:
            warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        
        training_args = SFTConfig(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            eval_strategy="steps" if eval_dataset else "no",
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            bf16=self.config.training.bf16,
            report_to="wandb" if self.config.wandb_project else None,
            run_name=f"{self.config.name}",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            dataset_text_field="text",
            max_length=512,
            packing=False,
            completion_only_loss=True,
            lr_scheduler_type=getattr(self.config.training, 'lr_scheduler_type', 'cosine'),
            weight_decay=getattr(self.config.training, 'weight_decay', 0.01)
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
    
    def train(self):
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.name,
                config=self.config.model_dump()
            )
        
        self.setup_model_and_tokenizer()
        
        # Need to get labels first to setup template
        from datasets import load_dataset
        dataset = load_dataset(self.config.data.dataset_name)
        if isinstance(dataset[self.config.data.train_split][self.config.data.label_column][0], int):
            label_names = dataset[self.config.data.train_split].features[self.config.data.label_column].names
            labels = list(label_names)
        else:
            unique_labels = list(set(dataset[self.config.data.train_split][self.config.data.label_column]))
            labels = sorted(unique_labels)
        
        self.setup_template(labels)
        
        dataset_dict = self.prepare_data()
        train_dataset = dataset_dict[self.config.data.train_split]
        eval_dataset = dataset_dict.get(self.config.data.validation_split or self.config.data.test_split)
        
        self.setup_trainer(train_dataset, eval_dataset)
        
        self.trainer.train()
        
        if self.config.wandb_project:
            wandb.finish()
    
    def save_model(self, path: str):
        if self.trainer:
            self.trainer.save_model(path)
            self.tokenizer.save_pretrained(path)