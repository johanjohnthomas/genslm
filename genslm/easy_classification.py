"""
High-level API for easy genomic sequence classification with GenSLM.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from genslm.classification import GenSLMClassifier
from genslm.dataset import ClassificationDataset
from genslm.inference import GenSLM

PathLike = Union[str, Path]


def train_classifier(
    data_path: PathLike,
    sequence_col: str = "BP",
    target_col: str = "Pathogens_To_Bees",
    model_id: str = "genslm_25M_patric",
    model_cache_dir: PathLike = ".",
    output_dir: PathLike = "./classification_results",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 8,
    max_epochs: int = 10,
    learning_rate: float = 1e-4,
    hidden_sizes: Optional[List[int]] = None,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    pooling_strategy: str = "mean",
    patience: int = 3,
    random_seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a GenSLM-based classifier on genomic sequence data.
    
    Parameters
    ----------
    data_path : PathLike
        Path to CSV file containing sequences and labels
    sequence_col : str
        Column name containing genomic sequences (default: "BP")
    target_col : str
        Column name containing labels (default: "Pathogens_To_Bees")
    model_id : str
        GenSLM model ID (default: "genslm_25M_patric")
    model_cache_dir : PathLike
        Directory containing pretrained model weights
    output_dir : PathLike
        Directory to save training results and checkpoints
    train_split : float
        Fraction of data for training (default: 0.7)
    val_split : float
        Fraction of data for validation (default: 0.15)
    test_split : float
        Fraction of data for testing (default: 0.15)
    batch_size : int
        Batch size for training (default: 8)
    max_epochs : int
        Maximum number of training epochs (default: 10)
    learning_rate : float
        Learning rate (default: 1e-4)
    hidden_sizes : Optional[List[int]]
        Hidden layer sizes for classification head. None for single linear layer
    dropout : float
        Dropout rate (default: 0.1)
    freeze_backbone : bool
        Whether to freeze GenSLM backbone (default: True)
    pooling_strategy : str
        Pooling strategy for sequence embeddings (default: "mean")
    patience : int
        Early stopping patience (default: 3)
    random_seed : int
        Random seed for reproducibility (default: 42)
    **kwargs
        Additional arguments passed to GenSLMClassifier
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing training results and model information
    """
    # Set seeds for reproducibility
    pl.seed_everything(random_seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🧬 Training GenSLM Classifier")
    print(f"📁 Data: {data_path}")
    print(f"📊 Sequence column: {sequence_col}")
    print(f"🎯 Target column: {target_col}")
    print(f"🤖 Model: {model_id}")
    print(f"💾 Output directory: {output_dir}")
    
    # Load a reference model to get tokenizer and sequence length
    print("🔄 Loading reference model...")
    ref_model = GenSLM(model_id, model_cache_dir)
    
    # Create datasets
    print("📚 Preparing datasets...")
    datasets = ClassificationDataset.from_csv(
        csv_path=data_path,
        sequence_col=sequence_col,
        label_col=target_col,
        seq_length=ref_model.seq_length,
        tokenizer=ref_model.tokenizer,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed,
        verbose=True
    )
    
    # Get number of classes
    num_classes = len(datasets["train"].label_encoder.classes_)
    print(f"🏷️  Number of classes: {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        datasets["train"], 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        datasets["val"], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        datasets["test"], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Calculate class weights for imbalanced datasets
    train_labels = [datasets["train"][i]["labels"].item() for i in range(len(datasets["train"]))]
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(len(class_counts) / class_counts)
    print(f"⚖️  Class weights: {class_weights.tolist()}")
    
    # Initialize model
    print("🏗️  Initializing classifier...")
    model = GenSLMClassifier(
        model_id=model_id,
        num_classes=num_classes,
        model_cache_dir=model_cache_dir,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        pooling_strategy=pooling_strategy,
        learning_rate=learning_rate,
        class_weights=class_weights,
        **kwargs
    )
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=patience,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stopping)
    
    # Model checkpointing
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="genslm-classifier-{epoch:02d}-{val/loss:.2f}",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        save_last=True
    )
    callbacks.append(checkpoint)
    
    # Logger
    csv_logger = CSVLogger(output_dir, name="training_logs")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=csv_logger,
        default_root_dir=str(output_dir),
        accelerator="auto",
        devices="auto",
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )
    
    # Train model
    print("🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    print("🧪 Testing model...")
    test_results = trainer.test(model, test_loader)
    
    # Save label encoder
    import joblib
    joblib.dump(
        datasets["train"].label_encoder, 
        output_dir / "label_encoder.pkl"
    )
    
    # Create results summary
    results = {
        "model_id": model_id,
        "num_classes": num_classes,
        "class_names": datasets["train"].label_encoder.classes_.tolist(),
        "best_model_path": checkpoint.best_model_path,
        "test_results": test_results[0] if test_results else {},
        "train_size": len(datasets["train"]),
        "val_size": len(datasets["val"]),
        "test_size": len(datasets["test"]),
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "freeze_backbone": freeze_backbone,
            "pooling_strategy": pooling_strategy,
        }
    }
    
    # Save results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Training completed!")
    print(f"📊 Best model: {checkpoint.best_model_path}")
    print(f"📈 Test results: {test_results[0] if test_results else 'N/A'}")
    
    return results


def load_classifier(
    checkpoint_path: PathLike,
    model_cache_dir: PathLike = "."
) -> GenSLMClassifier:
    """
    Load a trained GenSLM classifier from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : PathLike
        Path to saved model checkpoint
    model_cache_dir : PathLike
        Directory containing pretrained model weights
        
    Returns
    -------
    GenSLMClassifier
        Loaded classifier model
    """
    model = GenSLMClassifier.load_from_checkpoint(
        checkpoint_path,
        model_cache_dir=model_cache_dir
    )
    model.eval()
    return model


def predict_from_csv(
    model_path: PathLike,
    data_path: PathLike,
    sequence_col: str = "BP",
    output_path: Optional[PathLike] = None,
    batch_size: int = 32,
    model_cache_dir: PathLike = ".",
    label_encoder_path: Optional[PathLike] = None
) -> pd.DataFrame:
    """
    Make predictions on sequences from a CSV file.
    
    Parameters
    ----------
    model_path : PathLike
        Path to trained model checkpoint
    data_path : PathLike
        Path to CSV file containing sequences
    sequence_col : str
        Column name containing genomic sequences
    output_path : Optional[PathLike]
        Path to save predictions CSV. If None, doesn't save
    batch_size : int
        Batch size for inference
    model_cache_dir : PathLike
        Directory containing pretrained model weights
    label_encoder_path : Optional[PathLike]
        Path to saved label encoder. If None, looks in model directory
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus predictions
    """
    print(f"🔮 Making predictions...")
    print(f"🤖 Model: {model_path}")
    print(f"📁 Data: {data_path}")
    
    # Load model
    model = load_classifier(model_path, model_cache_dir)
    
    # Load label encoder
    if label_encoder_path is None:
        model_dir = Path(model_path).parent.parent  # Go up from checkpoints/
        label_encoder_path = model_dir / "label_encoder.pkl"
    
    import joblib
    if Path(label_encoder_path).exists():
        label_encoder = joblib.load(label_encoder_path)
    else:
        warnings.warn("Label encoder not found. Using numeric predictions.")
        label_encoder = None
    
    # Load data
    df = pd.read_csv(data_path)
    
    if sequence_col not in df.columns:
        raise ValueError(f"Sequence column '{sequence_col}' not found in CSV")
    
    sequences = df[sequence_col].tolist()
    
    # Make predictions
    with torch.no_grad():
        probabilities = model.predict(sequences, batch_size=batch_size)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Add predictions to dataframe
    df["predicted_class"] = predictions.cpu().numpy()
    df["prediction_confidence"] = torch.max(probabilities, dim=1)[0].cpu().numpy()
    
    # Add class probabilities
    for i, prob in enumerate(probabilities.T):
        df[f"prob_class_{i}"] = prob.cpu().numpy()
    
    # Convert to class names if label encoder available
    if label_encoder is not None:
        df["predicted_label"] = label_encoder.inverse_transform(df["predicted_class"])
        print(f"🏷️  Class names: {list(label_encoder.classes_)}")
    
    # Save if requested
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"💾 Predictions saved to: {output_path}")
    
    print(f"✅ Predictions completed for {len(df)} samples")
    
    return df


def evaluate_classifier(
    model_path: PathLike,
    data_path: PathLike,
    sequence_col: str = "BP",
    target_col: str = "Pathogens_To_Bees",
    model_cache_dir: PathLike = ".",
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate a trained classifier on test data.
    
    Parameters
    ----------
    model_path : PathLike
        Path to trained model checkpoint
    data_path : PathLike
        Path to CSV file containing test sequences and labels
    sequence_col : str
        Column name containing genomic sequences
    target_col : str
        Column name containing true labels
    model_cache_dir : PathLike
        Directory containing pretrained model weights
    batch_size : int
        Batch size for evaluation
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"📊 Evaluating classifier...")
    
    # Load model
    model = load_classifier(model_path, model_cache_dir)
    
    # Load data
    df = pd.read_csv(data_path)
    sequences = df[sequence_col].tolist()
    true_labels = df[target_col].tolist()
    
    # Make predictions
    with torch.no_grad():
        probabilities = model.predict(sequences, batch_size=batch_size)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
    
    # Calculate metrics
    report = classification_report(true_labels, predictions, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)
    
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }
    
    # Print summary
    print(f"✅ Evaluation completed!")
    print(f"🎯 Accuracy: {results['accuracy']:.4f}")
    print(f"📈 Macro F1: {results['macro_f1']:.4f}")
    print(f"⚖️  Weighted F1: {results['weighted_f1']:.4f}")
    
    return results


def quick_classify(
    sequences: List[str],
    model_id: str = "genslm_25M_patric",
    model_cache_dir: PathLike = ".",
    target: str = "pathogens_to_bees",
    data_path: Optional[PathLike] = None,
    sequence_col: str = "BP",
    target_col: str = "Pathogens_To_Bees"
) -> List[str]:
    """
    Quick classification of sequences using a pre-trained model or train on-the-fly.
    
    Parameters
    ----------
    sequences : List[str]
        List of genomic sequences to classify
    model_id : str
        GenSLM model ID for the backbone
    model_cache_dir : PathLike
        Directory containing pretrained model weights
    target : str
        Target classification task name
    data_path : Optional[PathLike]
        If provided, trains a new model using this data
    sequence_col : str
        Column name for sequences (if training)
    target_col : str
        Column name for labels (if training)
        
    Returns
    -------
    List[str]
        List of predicted class labels
    """
    if data_path is not None:
        # Train a quick model
        print("🚀 Quick training mode...")
        results = train_classifier(
            data_path=data_path,
            sequence_col=sequence_col,
            target_col=target_col,
            model_id=model_id,
            model_cache_dir=model_cache_dir,
            max_epochs=5,
            batch_size=16,
            output_dir=f"./quick_classify_{target}"
        )
        
        # Load trained model
        model = load_classifier(results["best_model_path"], model_cache_dir)
        
        # Load label encoder
        import joblib
        label_encoder = joblib.load(f"./quick_classify_{target}/label_encoder.pkl")
        
    else:
        raise NotImplementedError(
            "Pre-trained classification models not yet available. "
            "Please provide data_path to train a model."
        )
    
    # Make predictions
    with torch.no_grad():
        probabilities = model.predict(sequences, batch_size=32)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
    
    # Convert to class names
    predicted_labels = label_encoder.inverse_transform(predictions).tolist()
    
    return predicted_labels