#!/usr/bin/env python3
"""
Command-line interface for training GenSLM classifiers.
"""

import argparse
import sys
from pathlib import Path

from genslm.easy_classification import train_classifier


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train a GenSLM-based genomic sequence classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to CSV file containing sequences and labels"
    )
    
    # Data arguments
    parser.add_argument(
        "--sequence-col",
        type=str,
        default="BP",
        help="Column name containing genomic sequences"
    )
    
    parser.add_argument(
        "--target-col",
        type=str,
        default="Pathogens_To_Bees",
        help="Column name containing labels"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="genslm_25M_patric",
        choices=["genslm_25M_patric", "genslm_250M_patric", "genslm_2.5B_patric", "genslm_25B_patric"],
        help="GenSLM model ID"
    )
    
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=".",
        help="Directory containing pretrained model weights"
    )
    
    # Architecture arguments
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        help="Hidden layer sizes for classification head (e.g., --hidden-sizes 256 128)"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for classification head"
    )
    
    parser.add_argument(
        "--pooling-strategy",
        type=str,
        default="mean",
        choices=["mean", "max", "cls", "last"],
        help="Pooling strategy for sequence embeddings"
    )
    
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze GenSLM backbone during training"
    )
    
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze GenSLM backbone during training"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    # Data split arguments
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data for training"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation"
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of data for testing"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./classification_results",
        help="Directory to save training results and checkpoints"
    )
    
    # Utility arguments
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.data_path).exists():
        print(f"Error: Data file '{args.data_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        print("Error: Train, validation, and test splits must sum to 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Handle freeze/unfreeze backbone
    freeze_backbone = args.freeze_backbone and not args.unfreeze_backbone
    
    # Convert hidden sizes
    hidden_sizes = args.hidden_sizes if args.hidden_sizes else None
    
    try:
        # Train the classifier
        results = train_classifier(
            data_path=args.data_path,
            sequence_col=args.sequence_col,
            target_col=args.target_col,
            model_id=args.model_id,
            model_cache_dir=args.model_cache_dir,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
            freeze_backbone=freeze_backbone,
            pooling_strategy=args.pooling_strategy,
            patience=args.patience,
            random_seed=args.random_seed
        )
        
        if not args.quiet:
            print("\n🎉 Training completed successfully!")
            print(f"📂 Results saved to: {args.output_dir}")
            print(f"🏆 Best model: {results.get('best_model_path', 'N/A')}")
            if results.get('test_results'):
                test_acc = results['test_results'].get('test/acc', 'N/A')
                print(f"🎯 Test accuracy: {test_acc}")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())