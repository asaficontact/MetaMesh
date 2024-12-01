import os
from typing import Dict, Any, List
from src.agents.legal_contract_expert import LegalContractExpert
from src.cuad_executor import CUADExecutor
from collections import defaultdict
import argparse

from src.utils import save_json

#TODO: The preprocessing data code should be made available as a part of the CUADEvaluator

## Constants
OPENAI_MODELS_LIST = ["gpt-4o-mini", "gpt-4o"]
OLLAMA_MODELS_LIST = ["phi3", "mistral", "llama2:13b"]
EVALUATION_SAVE_PATH = 'results/evals'

class CUADEvaluator:
    """
    Evaluator class to assess model performance on CUAD dataset
    """
    def __init__(self, cuad_data_path: str, evaluation_save_path: str = EVALUATION_SAVE_PATH):
        """
        Initialize evaluator
        
        Args:
            cuad_data_path (str): Path to CUAD dataset
        """
        self.executor = CUADExecutor(cuad_data_path)
        self.openai_models = OPENAI_MODELS_LIST
        self.ollama_models = OLLAMA_MODELS_LIST
        self.evaluation_save_path = evaluation_save_path
        
    def evaluate_model(self, model_name: str, debug: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single model on CUAD dataset
        
        Args:
            model_name (str): Name of model to evaluate
            debug (bool): Whether to enable debug mode
            
        Returns:
            dict: Evaluation results for the model
        """
        # Create agent for model
        if model_name in self.openai_models:
            agent = LegalContractExpert(provider='openai', model_name=model_name)
        elif model_name in self.ollama_models:
            agent = LegalContractExpert(provider='ollama', model_name=model_name)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        # Get predictions
        predictions = self.executor.process_dataset(agent, debug=debug)
        
        # Calculate metrics
        file_metrics = self._calculate_file_metrics(predictions)
        avg_metrics = self._calculate_average_metrics(file_metrics)
        
        # Store results
        results = {
            "file_level_metrics": file_metrics,
            "average_metrics": avg_metrics
        }

        # Save results
        save_path = os.path.join(self.evaluation_save_path, f"{model_name}_baseline.json")
        save_json({model_name: results}, save_path)

        return results

    def evaluate_models(self, model_names: List[str], debug: bool = False) -> Dict[str, Any]:
        """
        Evaluate multiple models on CUAD dataset
        
        Args:
            model_names (List[str]): List of model names to evaluate
            debug (bool): Whether to enable debug mode
            
        Returns:
            dict: Evaluation results for each model
        """
        results = {}
        
        for model_name in model_names:
            results[model_name] = self.evaluate_model(model_name, debug=debug)

        return results
    
    def _calculate_file_metrics(self, predictions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each file
        
        Args:
            predictions (dict): Model predictions for each file
            
        Returns:
            dict: Metrics for each file
        """
        file_metrics = {}
        
        for filename, file_preds in predictions.items():
            # Initialize counters
            correct = 0
            total = 0
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            # Calculate metrics for each category
            for category_preds in file_preds.values():
                pred = category_preds['predicted_answer'].lower()
                true = category_preds['true_value'].lower()
                
                total += 1
                if pred == true:
                    correct += 1
                    
                if true == 'yes':
                    if pred == 'yes':
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:  # true == 'no'
                    if pred == 'no':
                        true_negatives += 1
                    else:
                        false_positives += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            file_metrics[filename] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "correct_predictions": correct,
                "total_questions": total,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
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
        total_files = len(file_metrics)
        
        # Sum up metrics across files
        for metrics in file_metrics.values():
            for metric_name, value in metrics.items():
                metric_sums[metric_name] += value
        
        # Calculate averages
        avg_metrics = {
            metric: value / total_files 
            for metric, value in metric_sums.items()
        }
        
        return avg_metrics 

    def evaluate_model_for_files(self, model_name: str, file_list: List[str], debug: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model on a specific subset of files from CUAD dataset
        
        Args:
            model_name (str): Name of model to evaluate
            file_list (List[str]): List of filenames to evaluate
            debug (bool): Whether to enable debug mode
            
        Returns:
            dict: Evaluation results for the model on specified files
        """
        # Create agent for model
        if model_name in self.openai_models:
            agent = LegalContractExpert(provider='openai', model_name=model_name)
        elif model_name in self.ollama_models:
            agent = LegalContractExpert(provider='ollama', model_name=model_name)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        # Process files in parallel using the new method
        predictions = self.executor.process_files_list(
            file_list=file_list,
            agent=agent,
            debug=debug
        )
        
        # Calculate metrics
        file_metrics = self._calculate_file_metrics(predictions)
        avg_metrics = self._calculate_average_metrics(file_metrics)
        
        # Store results
        results = {
            model_name: {
                "file_level_metrics": file_metrics,
                "average_metrics": avg_metrics
            }
        }

        # Save results
        save_path = os.path.join(self.evaluation_save_path, f"{model_name}_subset_eval.json")
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