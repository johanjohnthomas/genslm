#!/usr/bin/env python3
"""
Command-line interface for making predictions with trained GenSLM classifiers.
"""

import argparse
import sys
from pathlib import Path

from genslm.easy_classification import predict_from_csv, evaluate_classifier


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained GenSLM classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Make predictions on sequences"
    )
    
    predict_parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model checkpoint"
    )
    
    predict_parser.add_argument(
        "data_path",
        type=str,
        help="Path to CSV file containing sequences"
    )
    
    predict_parser.add_argument(
        "--sequence-col",
        type=str,
        default="BP",
        help="Column name containing genomic sequences"
    )
    
    predict_parser.add_argument(
        "--output-path",
        type=str,
        help="Path to save predictions CSV (default: input_predictions.csv)"
    )
    
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    
    predict_parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=".",
        help="Directory containing pretrained model weights"
    )
    
    predict_parser.add_argument(
        "--label-encoder-path",
        type=str,
        help="Path to saved label encoder (default: auto-detect)"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate classifier on test data"
    )
    
    eval_parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model checkpoint"
    )
    
    eval_parser.add_argument(
        "data_path",
        type=str,
        help="Path to CSV file containing test sequences and labels"
    )
    
    eval_parser.add_argument(
        "--sequence-col",
        type=str,
        default="BP",
        help="Column name containing genomic sequences"
    )
    
    eval_parser.add_argument(
        "--target-col",
        type=str,
        default="Pathogens_To_Bees",
        help="Column name containing true labels"
    )
    
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    eval_parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=".",
        help="Directory containing pretrained model weights"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Validate common arguments
    if not Path(args.model_path).exists():
        print(f"Error: Model file '{args.model_path}' not found", file=sys.stderr)
        return 1
    
    if not Path(args.data_path).exists():
        print(f"Error: Data file '{args.data_path}' not found", file=sys.stderr)
        return 1
    
    try:
        if args.command == "predict":
            # Set default output path if not provided
            if args.output_path is None:
                input_path = Path(args.data_path)
                args.output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
            
            # Make predictions
            df = predict_from_csv(
                model_path=args.model_path,
                data_path=args.data_path,
                sequence_col=args.sequence_col,
                output_path=args.output_path,
                batch_size=args.batch_size,
                model_cache_dir=args.model_cache_dir,
                label_encoder_path=args.label_encoder_path
            )
            
            print(f"\n🎉 Predictions completed!")
            print(f"📊 Processed {len(df)} samples")
            print(f"💾 Results saved to: {args.output_path}")
            
        elif args.command == "evaluate":
            # Evaluate model
            results = evaluate_classifier(
                model_path=args.model_path,
                data_path=args.data_path,
                sequence_col=args.sequence_col,
                target_col=args.target_col,
                model_cache_dir=args.model_cache_dir,
                batch_size=args.batch_size
            )
            
            print(f"\n🎉 Evaluation completed!")
            print(f"📊 Results:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Macro F1: {results['macro_f1']:.4f}")
            print(f"  Weighted F1: {results['weighted_f1']:.4f}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())