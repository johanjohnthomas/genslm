import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from genslm.inference import GenSLM

PathLike = Union[str, Path]


class ClassificationHead(nn.Module):
    """Classification head with configurable architecture."""
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_sizes is None:
            # Simple linear layer
            self.classifier = nn.Linear(input_size, num_classes)
        else:
            # Multi-layer perceptron
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    self._get_activation(activation),
                    nn.Dropout(dropout)
                ])
                prev_size = hidden_size
                
            layers.append(nn.Linear(prev_size, num_classes))
            self.classifier = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class GenSLMClassifier(pl.LightningModule):
    """GenSLM-based classifier for genomic sequences."""
    
    def __init__(
        self,
        model_id: str = "genslm_25M_patric",
        num_classes: int = 2,
        model_cache_dir: PathLike = ".",
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        pooling_strategy: str = "mean",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        GenSLM-based classifier.
        
        Parameters
        ----------
        model_id : str
            GenSLM model ID (e.g., "genslm_25M_patric")
        num_classes : int
            Number of output classes
        model_cache_dir : PathLike
            Directory containing model weights
        hidden_sizes : Optional[List[int]]
            Hidden layer sizes for classification head. If None, uses single linear layer
        dropout : float
            Dropout rate for classification head
        freeze_backbone : bool
            Whether to freeze the GenSLM backbone during training
        pooling_strategy : str
            How to pool sequence embeddings ("mean", "max", "cls", "last")
        learning_rate : float
            Learning rate for training
        weight_decay : float
            Weight decay for optimizer
        class_weights : Optional[torch.Tensor]
            Class weights for imbalanced datasets
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained GenSLM model
        self.backbone = GenSLM(model_id, model_cache_dir)
        self.backbone.eval()  # Always in eval mode for inference
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get hidden size from the backbone model config
        config = AutoConfig.from_pretrained(self.backbone.model_info["config"])
        hidden_size = config.hidden_size
        
        # Create classification head
        self.classifier = ClassificationHead(
            input_size=hidden_size,
            num_classes=num_classes,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence embeddings based on pooling strategy."""
        # hidden_states: (batch_size, seq_length, hidden_size)
        # attention_mask: (batch_size, seq_length)
        
        if self.hparams.pooling_strategy == "mean":
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            return sum_embeddings / sum_mask
        
        elif self.hparams.pooling_strategy == "max":
            # Max pooling with masking
            hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1), -float('inf'))
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.hparams.pooling_strategy == "cls":
            # Use first token (CLS-like)
            return hidden_states[:, 0, :]
        
        elif self.hparams.pooling_strategy == "last":
            # Use last non-padded token
            batch_size, seq_length = attention_mask.shape
            last_indices = attention_mask.sum(dim=1) - 1
            return hidden_states[range(batch_size), last_indices]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.hparams.pooling_strategy}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get embeddings from backbone
        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Use last layer hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Pool embeddings
        pooled_embeddings = self._pool_embeddings(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_embeddings)
        
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        self.training_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": labels.detach()
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": labels.detach()
        })
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        
        self.test_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": labels.detach()
        })
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if self.training_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.training_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.training_step_outputs])
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted'
            )
            
            self.log("train/precision", precision)
            self.log("train/recall", recall)
            self.log("train/f1", f1)
            
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.validation_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted'
            )
            
            self.log("val/precision", precision)
            self.log("val/recall", recall)
            self.log("val/f1", f1)
            
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        if self.test_step_outputs:
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
            
            accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted'
            )
            
            self.log("test/acc", accuracy)
            self.log("test/precision", precision)
            self.log("test/recall", recall)
            self.log("test/f1", f1)
            
            # Print detailed results
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1: {f1:.4f}")
            
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Different learning rates for backbone and classifier
        if self.hparams.freeze_backbone:
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            # Lower learning rate for backbone
            param_groups = [
                {"params": self.backbone.parameters(), "lr": self.hparams.learning_rate * 0.1},
                {"params": self.classifier.parameters(), "lr": self.hparams.learning_rate}
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1
            }
        }
    
    def predict(self, sequences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Predict on a list of sequences."""
        from genslm.dataset import SequenceDataset
        from torch.utils.data import DataLoader
        
        self.eval()
        device = next(self.parameters()).device
        
        # Create dataset
        dataset = SequenceDataset(
            sequences, 
            self.backbone.seq_length, 
            self.backbone.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                logits = self(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu())
        
        return torch.cat(predictions, dim=0)