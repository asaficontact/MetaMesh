import os
from typing import Dict, Any, List
from src.agents.legal_contract_expert import LegalContractExpert
from src.cuad_executor import CUADExecutor
from collections import defaultdict
import argparse
import json

from src.utils import save_json

#TODO: The preprocessing data code should be made available as a part of the CUADEvaluator

## Constants
OPENAI_MODELS_LIST = ["gpt-4o-mini", "gpt-4o"]
OLLAMA_MODELS_LIST = ["phi3", "mistral", "llama2:13b"]
EVALUATION_SAVE_PATH = 'results/evals'
TOKEN_OUTPUT_DIR = 'results/QA_token_count'
TIME_OUTPUT_DIR = 'results/QA_processing_time'

class CUADEvaluator:
    """
    Evaluator class to assess model performance on CUAD dataset
    """
    def __init__(self, 
                 cuad_data_path: str, 
                 evaluation_save_path: str = EVALUATION_SAVE_PATH,
                 use_contract_doc: bool = True,
                 use_intermediate_rep: bool = False,
                 intermediate_rep_dir_path: str = None):
        """
        Initialize evaluator
        
        Args:
            cuad_data_path (str): Path to CUAD dataset
            evaluation_save_path (str): Path to save evaluation results
            use_contract_doc (bool): Whether to use original contract document
            use_intermediate_rep (bool): Whether to use intermediate representations
            intermediate_rep_dir_path (str): Directory containing intermediate representations
        """
        self.executor = CUADExecutor(
            cuad_data_path=cuad_data_path,
            use_contract_doc=use_contract_doc,
            use_intermediate_rep=use_intermediate_rep,
            intermediate_rep_dir_path=intermediate_rep_dir_path
        )
        self.openai_models = OPENAI_MODELS_LIST
        self.ollama_models = OLLAMA_MODELS_LIST
        self.evaluation_save_path = evaluation_save_path
        
        # Store configuration for logging
        self.config = {
            "use_contract_doc": use_contract_doc,
            "use_intermediate_rep": use_intermediate_rep,
            "intermediate_rep_dir_path": intermediate_rep_dir_path
        }
        
        # Create output directories
        for dir_path in [evaluation_save_path, TOKEN_OUTPUT_DIR, TIME_OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
    def _save_metrics(self, metrics: dict, output_dir: str, filename: str):
        """Save metrics to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def evaluate_model(self, model_name: str, debug: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single model on CUAD dataset
        
        Args:
            model_name (str): Name of model to evaluate
            debug (bool): Whether to enable debug mode
            verbose (bool): Verbosity level
            
        Returns:
            dict: Evaluation results for the model
        """
        # Determine provider
        if model_name in self.openai_models:
            provider = 'openai'
        elif model_name in self.ollama_models:
            provider = 'ollama'
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        # Update config with model info
        self.config.update({
            "model_name": model_name,
            "provider": provider
        })
        
        # Process dataset
        predictions = self.executor.process_dataset(
            model_name=model_name,
            provider=provider,
            debug=debug,
            verbose=verbose
        )
        
        if debug:
            print("\nDEBUG: Predictions structure:")
            for filename, file_data in predictions.items():
                print(f"\nFile: {filename}")
                print(f"Keys: {file_data.keys()}")
                if 'metrics' in file_data:
                    print("Has metrics")
                for category, category_data in file_data.items():
                    if category != 'metrics':
                        print(f"Category {category} keys: {category_data.keys()}")
        
        # Calculate metrics
        file_metrics = self._calculate_file_metrics(predictions, debug=debug)
        avg_metrics = self._calculate_average_metrics(file_metrics)
        
        # Aggregate token and time metrics by category
        category_metrics = defaultdict(lambda: {
            'token_counts': {'input_tokens': [], 'output_tokens': [], 'total_tokens': []},
            'processing_time': []
        })
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_processing_time = 0
        total_categories = 0
        
        # Collect metrics from predictions
        for filename, file_data in predictions.items():
            if filename == 'token_count':
                continue
            
            for category, category_data in file_data.items():
                if category == 'metrics':
                    continue
                
                if 'metrics' in category_data:
                    metrics = category_data['metrics']
                    token_count = metrics.get('token_count', {})
                    category_metrics[category]['token_counts']['input_tokens'].append(token_count.get('input_tokens', 0))
                    category_metrics[category]['token_counts']['output_tokens'].append(token_count.get('output_tokens', 0))
                    category_metrics[category]['token_counts']['total_tokens'].append(
                        token_count.get('input_tokens', 0) + token_count.get('output_tokens', 0)
                    )
                    category_metrics[category]['processing_time'].append(metrics.get('processing_time', 0))
                    
                    total_input_tokens += token_count.get('input_tokens', 0)
                    total_output_tokens += token_count.get('output_tokens', 0)
                    total_processing_time += metrics.get('processing_time', 0)
                    total_categories += 1
        
        # Calculate averages for each category
        category_averages = {}
        for category, metrics in category_metrics.items():
            category_averages[category] = {
                'token_counts': {
                    'avg_input_tokens': sum(metrics['token_counts']['input_tokens']) / len(metrics['token_counts']['input_tokens']),
                    'avg_output_tokens': sum(metrics['token_counts']['output_tokens']) / len(metrics['token_counts']['output_tokens']),
                    'avg_total_tokens': sum(metrics['token_counts']['total_tokens']) / len(metrics['token_counts']['total_tokens'])
                },
                'processing_time': sum(metrics['processing_time']) / len(metrics['processing_time'])
            }
        
        # Add comprehensive metrics to results
        results = {
            model_name: {
                "config": self.config,
                "file_level_metrics": file_metrics,
                "average_metrics": avg_metrics,
                "token_counts": {
                    "dataset_level": {
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    },
                    "category_averages": category_averages,
                    "overall_averages": {
                        "avg_input_tokens": total_input_tokens / total_categories,
                        "avg_output_tokens": total_output_tokens / total_categories,
                        "avg_total_tokens": (total_input_tokens + total_output_tokens) / total_categories
                    }
                },
                "processing_time": {
                    "dataset_level": {
                        "total_processing_time": total_processing_time
                    },
                    "category_averages": {cat: metrics['processing_time'] for cat, metrics in category_averages.items()},
                    "overall_average": total_processing_time / total_categories
                }
            }
        }
        
        # Create base config string
        config_str = "_".join([
            model_name,
            "rep" if self.config["use_intermediate_rep"] else "",
            "contract" if self.config["use_contract_doc"] else ""
        ]).strip("_")

        # Separate metrics into different dictionaries
        evaluation_results = {
            model_name: {
                "config": self.config,
                "file_level_metrics": file_metrics,
                "average_metrics": avg_metrics
            }
        }

        token_metrics = {
            model_name: {
                "config": self.config,
                "token_counts": results[model_name]["token_counts"]
            }
        }

        time_metrics = {
            model_name: {
                "config": self.config,
                "processing_time": results[model_name]["processing_time"]
            }
        }

        # Save each type of metrics to appropriate directory
        self._save_metrics(
            evaluation_results,
            self.evaluation_save_path,
            f"{config_str}_eval.json"
        )
        
        self._save_metrics(
            token_metrics,
            TOKEN_OUTPUT_DIR,
            f"{config_str}_token_count.json"
        )
        
        self._save_metrics(
            time_metrics,
            TIME_OUTPUT_DIR,
            f"{config_str}_processing_time.json"
        )

        return evaluation_results

    def evaluate_models(self, model_names: List[str], debug: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate multiple models on CUAD dataset
        
        Args:
            model_names (List[str]): List of model names to evaluate
            debug (bool): Whether to enable debug mode
            verbose (bool): Verbosity level
            
        Returns:
            dict: Evaluation results for each model
        """
        results = {}
        
        for model_name in model_names:
            results[model_name] = self.evaluate_model(model_name, debug=debug, verbose=verbose)
    
    
    def _calculate_file_metrics(self, predictions: Dict, debug: bool = False) -> Dict[str, Dict]:
        """Calculate metrics for each file"""
        file_metrics = {}
        
        for filename, file_data in predictions.items():
            if filename == 'token_count':  # Skip the metrics key
                continue
            
            if debug:
                print(f"\n{'='*50}")
                print(f"DEBUG: Processing file {filename}")
                print(f"{'='*50}")
            
            # Initialize counters
            total = 0
            correct = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            file_input_tokens = 0
            file_output_tokens = 0
            file_processing_time = 0
            
            # Process each category in the file
            for category, category_preds in file_data.items():
                if category == 'metrics':  # Skip the metrics key
                    continue
                
                try:
                    pred = category_preds.get('predicted_answer', '').lower().strip()
                    true = category_preds.get('true_value', '').lower().strip()
                    
                    if debug:
                        print(f"\nCategory: {category}")
                        print(f"Predicted: '{pred}'")
                        print(f"True: '{true}'")
                    
                    if pred and true:  # Only count if we have both values
                        total += 1
                        if pred == true:
                            correct += 1
                            if debug:
                                print("✓ Correct prediction")
                            
                        # Calculate TP, FP, FN
                        if pred == 'yes' and true == 'yes':
                            true_positives += 1
                            if debug:
                                print("✓ True positive")
                        elif pred == 'yes' and true == 'no':
                            false_positives += 1
                            if debug:
                                print("✗ False positive")
                        elif pred == 'no' and true == 'yes':
                            false_negatives += 1
                            if debug:
                                print("✗ False negative")
                            
                        if debug:
                            print(f"Running totals:")
                            print(f"- Total: {total}")
                            print(f"- Correct: {correct}")
                            print(f"- True Positives: {true_positives}")
                            print(f"- False Positives: {false_positives}")
                            print(f"- False Negatives: {false_negatives}")
                
                    # Collect token counts and processing time
                    if 'metrics' in category_preds:
                        metrics = category_preds['metrics']
                        token_count = metrics.get('token_count', {})
                        file_input_tokens += token_count.get('input_tokens', 0)
                        file_output_tokens += token_count.get('output_tokens', 0)
                        file_processing_time += metrics.get('processing_time', 0)
                        
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Error processing category {category}: {str(e)}")
                    continue
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if debug:
                print(f"\nFinal metrics for {filename}:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1_score:.4f}")
                print(f"Total questions: {total}")
                print(f"Correct answers: {correct}")
                print(f"True positives: {true_positives}")
                print(f"False positives: {false_positives}")
                print(f"False negatives: {false_negatives}")
            
            file_metrics[filename] = {
                'total_questions': total,
                'correct_answers': correct,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'token_counts': {
                    'input_tokens': file_input_tokens,
                    'output_tokens': file_output_tokens,
                    'total_tokens': file_input_tokens + file_output_tokens
                },
                'processing_time': file_processing_time
            }
        
        return file_metrics
    
    def _calculate_average_metrics(self, file_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across all files
        
        Args:
            file_metrics (dict): Metrics for each file
            
        Returns:
            dict: Average metrics across all files
        """
        # Initialize metric sums
        metric_sums = defaultdict(float)
        token_sums = defaultdict(float)
        processing_time_sum = 0
        total_files = len(file_metrics)
        
        if total_files == 0:
            return {
                'total_files_processed': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'token_counts': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0
                },
                'processing_time': 0
            }
        
        # Sum up metrics across files
        for metrics in file_metrics.values():
            # Handle basic metrics
            for metric_name, value in metrics.items():
                if metric_name not in ['token_counts', 'processing_time']:
                    metric_sums[metric_name] += value
            
            # Handle token counts
            if 'token_counts' in metrics:
                for token_type, count in metrics['token_counts'].items():
                    token_sums[token_type] += count
            
            # Handle processing time
            if 'processing_time' in metrics:
                processing_time_sum += metrics['processing_time']
        
        # Calculate averages
        avg_metrics = {
            metric: value / total_files 
            for metric, value in metric_sums.items()
            if metric not in ['token_counts', 'processing_time']
        }
        
        # Add token count averages
        avg_metrics['token_counts'] = {
            token_type: value / total_files
            for token_type, value in token_sums.items()
        }
        
        # Add processing time average
        avg_metrics['processing_time'] = processing_time_sum / total_files
        
        # Add total files processed
        avg_metrics['total_files_processed'] = total_files
        
        return avg_metrics

    def evaluate_model_for_files(self, model_name: str, file_list: List[str], debug: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model on a specific subset of files from CUAD dataset
        
        Args:
            model_name (str): Name of model to evaluate
            file_list (List[str]): List of filenames to evaluate
            debug (bool): Whether to enable debug mode
            verbose (bool): Verbosity level
            
        Returns:
            dict: Evaluation results for the model on specified files
        """
        # Determine provider
        if model_name in self.openai_models:
            provider = 'openai'
        elif model_name in self.ollama_models:
            provider = 'ollama'
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        # Update config with model info
        self.config.update({
            "model_name": model_name,
            "provider": provider
        })
        
        # Process files in parallel using the new method
        predictions = self.executor.process_files_list(
            file_list=file_list,
            model_name=model_name,
            provider=provider,
            debug=debug,
            verbose=verbose
        )
        
        # Calculate metrics
        file_metrics = self._calculate_file_metrics(predictions, debug=debug)
        avg_metrics = self._calculate_average_metrics(file_metrics)
        
        # Store results with configuration
        results = {
            model_name: {
                "config": self.config,
                "file_level_metrics": file_metrics,
                "average_metrics": avg_metrics
            }
        }

        # Save results with configuration in filename
        config_str = "_".join([
            model_name,
            "contract" if self.config["use_contract_doc"] else "",
            "rep" if self.config["use_intermediate_rep"] else ""
        ]).strip("_")
        
        save_path = os.path.join(
            self.evaluation_save_path, 
            f"{config_str}_subset_eval.json"
        )
        save_json(results, save_path)

        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate models on CUAD dataset')
    parser.add_argument('--cuad_data_path', type=str, required=True,
                      help='Path to CUAD dataset JSON file')
    parser.add_argument('--evaluation_save_path', type=str, default=EVALUATION_SAVE_PATH,
                      help='Path to save evaluation results')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to evaluate')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')

    args = parser.parse_args()

    evaluator = CUADEvaluator(
        cuad_data_path=args.cuad_data_path,
        evaluation_save_path=args.evaluation_save_path
    )
    
    results = evaluator.evaluate_model(
        model_name=args.model_name,
        debug=args.debug
    )
    
    print(f"Evaluation results for {args.model_name}:")
    print(results)

if __name__ == "__main__":
    main()