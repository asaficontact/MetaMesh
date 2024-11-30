import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, Any, Tuple, Union

def load_json(file_path: str) -> dict:
    """
    Load a JSON file from the given path.
    
    Args:
        file_path (str): Path to the JSON file to load
        
    Returns:
        dict: Loaded JSON data as a dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: dict, save_path: str) -> None:
    """
    Save a dictionary as a JSON file.
    
    Args:
        data (dict): Dictionary to save as JSON
        save_path (str): Path where to save the JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

def analyze_question_distributions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the distribution of yes/no answers across question categories.
    
    Args:
        data (dict): The CUAD dataset dictionary
        
    Returns:
        dict: Analysis results containing:
            - per_category: Distribution for each question category
            - overall: Overall distribution across all categories
            - per_file: Distribution for each file
    """
    # Initialize counters
    category_counts = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0})
    file_counts = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0})
    overall_counts = {'yes': 0, 'no': 0, 'total': 0}
    
    # Analyze each file
    for filename, file_data in data.items():
        for category, category_data in file_data.items():
            # Skip document_path
            if category == 'document_path':
                continue
                
            answer = category_data.get('answer', '').lower()
            if answer in ['yes', 'no']:
                # Update category counts
                category_counts[category][answer] += 1
                category_counts[category]['total'] += 1
                
                # Update file counts
                file_counts[filename][answer] += 1
                file_counts[filename]['total'] += 1
                
                # Update overall counts
                overall_counts[answer] += 1
                overall_counts['total'] += 1
    
    # Calculate percentages
    analysis = {
        'per_category': {},
        'overall': {},
        'per_file': {}
    }
    
    # Per category analysis
    for category, counts in category_counts.items():
        total = counts['total']
        analysis['per_category'][category] = {
            'yes_percent': (counts['yes'] / total * 100) if total > 0 else 0,
            'no_percent': (counts['no'] / total * 100) if total > 0 else 0,
            'total_questions': total
        }
    
    # Overall analysis
    total = overall_counts['total']
    analysis['overall'] = {
        'yes_percent': (overall_counts['yes'] / total * 100) if total > 0 else 0,
        'no_percent': (overall_counts['no'] / total * 100) if total > 0 else 0,
        'total_questions': total
    }
    
    # Per file analysis
    for filename, counts in file_counts.items():
        total = counts['total']
        analysis['per_file'][filename] = {
            'yes_percent': (counts['yes'] / total * 100) if total > 0 else 0,
            'no_percent': (counts['no'] / total * 100) if total > 0 else 0,
            'total_questions': total
        }
    
    return analysis

def plot_category_distributions(analysis: Dict[str, Any], save_path: str = None) -> None:
    """
    Create visualizations for the analysis results.
    
    Args:
        analysis (dict): Analysis results from analyze_question_distributions
        save_path (str, optional): Path to save the plots. If None, displays plots instead.
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot category distributions
    categories = list(analysis['per_category'].keys())
    yes_percentages = [analysis['per_category'][cat]['yes_percent'] for cat in categories]
    
    # Horizontal bar chart for category distributions
    y_pos = range(len(categories))
    ax1.barh(y_pos, yes_percentages)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories)
    ax1.set_xlabel('Percentage of "Yes" Answers')
    ax1.set_title('Distribution of "Yes" Answers by Category')
    
    # Overall distribution pie chart
    labels = ['Yes', 'No']
    sizes = [analysis['overall']['yes_percent'], 
             analysis['overall']['no_percent']]
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax2.set_title('Overall Distribution of Yes/No Answers')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def sort_eval_results(eval_data: dict, doc_stats: dict) -> list:
    """
    Sort evaluation results based on F1 score, precision, recall, and accuracy.
    Weighted score calculation:
    - F1 score: 50%
    - Precision: 17.5%
    - Recall: 17.5%
    - Accuracy: 15%
    
    Args:
        eval_data (dict): Evaluation results from llama2:13b_baseline.json
        doc_stats (dict): Document statistics from txt_doc_stats.json
        
    Returns:
        list: List of tuples containing (filename, metrics) sorted by performance
    """
    # Extract file-level metrics
    file_metrics = eval_data['llama2:13b']['file_level_metrics']
    
    # Create list of tuples with filename and metrics
    results = []
    for filename, metrics in file_metrics.items():
        # Calculate weighted score
        weighted_score = (
            0.50 * metrics['f1_score'] +      # F1 score: 50%
            0.175 * metrics['precision'] +     # Precision: 17.5%
            0.175 * metrics['recall'] +        # Recall: 17.5%
            0.15 * metrics['accuracy']         # Accuracy: 15%
        )
        
        results.append((
            filename,
            {
                'weighted_score': weighted_score,
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': metrics['accuracy'],
                'token_length': doc_stats.get(filename, {}).get('token_length', 0)
            }
        ))
    
    # Sort by weighted score (lowest first)
    return sorted(results, key=lambda x: x[1]['weighted_score'])

def get_worst_performing_files(sorted_results: list, n: int = 20, max_tokens: int = None) -> list:
    """
    Get the n worst performing files, optionally filtering by token count.
    
    Args:
        sorted_results (list): Sorted results from sort_eval_results()
        n (int): Number of files to return
        max_tokens (int, optional): Maximum token length filter
        
    Returns:
        list: List of n worst performing filenames
    """
    if max_tokens is None:
        return [filename for filename, _ in sorted_results[:n]]
    
    # Filter by token count and return worst n
    filtered_results = [
        (filename, metrics) 
        for filename, metrics in sorted_results
        if metrics['token_length'] <= max_tokens
    ]
    return [filename for filename, _ in filtered_results[:n]]

def save_file_subsets(worst_overall: list, worst_filtered: list, save_path: str, token_cutoff: int) -> None:
    """
    Save the two sets of worst performing files to a JSON file.
    
    Args:
        worst_overall (list): List of worst performing files overall
        worst_filtered (list): List of worst performing files under token limit
        save_path (str): Path to save the JSON file
    """
    subsets = {
        'worst_performing_overall': worst_overall,
        f'worst_performing_under_{str(token_cutoff)}_tokens': worst_filtered
    }
    save_json(subsets, save_path)

def analyze_subset_metrics(eval_data: dict, file_subset: Union[list, dict], print_results: bool = True) -> dict:
    """
    Calculate average metrics for a subset of files.
    
    Args:
        eval_data (dict): Evaluation results from llama2:13b_baseline.json
        file_subset (Union[list, dict]): List of filenames or dictionary containing performance categories
        print_results (bool): Whether to print the analysis results
        
    Returns:
        dict: Average metrics for each subset
    """
    metrics = eval_data['llama2:13b']['file_level_metrics']
    
    def calculate_avg_metrics(files: list) -> dict:
        subset_metrics = [metrics[filename] for filename in files]
        return {
            'avg_f1_score': sum(m['f1_score'] for m in subset_metrics) / len(subset_metrics),
            'avg_precision': sum(m['precision'] for m in subset_metrics) / len(subset_metrics),
            'avg_recall': sum(m['recall'] for m in subset_metrics) / len(subset_metrics),
            'avg_accuracy': sum(m['accuracy'] for m in subset_metrics) / len(subset_metrics),
            'num_files': len(subset_metrics)
        }
    
    # Handle both list and dict inputs
    if isinstance(file_subset, list):
        results = {'overall': calculate_avg_metrics(file_subset)}
        if print_results:
            print("\nMetrics for file subset:")
            print(f"Number of files: {results['overall']['num_files']}")
            print(f"Average F1 Score: {results['overall']['avg_f1_score']:.4f}")
            print(f"Average Precision: {results['overall']['avg_precision']:.4f}")
            print(f"Average Recall: {results['overall']['avg_recall']:.4f}")
            print(f"Average Accuracy: {results['overall']['avg_accuracy']:.4f}")
    else:
        results = {}
        for category, files in file_subset.items():
            results[category] = calculate_avg_metrics(files)
            
            if print_results:
                print(f"\nMetrics for {category}:")
                print(f"Number of files: {results[category]['num_files']}")
                print(f"Average F1 Score: {results[category]['avg_f1_score']:.4f}")
                print(f"Average Precision: {results[category]['avg_precision']:.4f}")
                print(f"Average Recall: {results[category]['avg_recall']:.4f}")
                print(f"Average Accuracy: {results[category]['avg_accuracy']:.4f}")
    
    return results

def get_performance_file_sets(sorted_results: list, 
                            n_worst: int = 20, 
                            n_avg: int = 20, 
                            n_75th: int = 20, 
                            max_tokens: int = None) -> dict:
    """
    Get sets of files based on different performance levels:
    - Worst performing files
    - Files with average performance
    - Files performing at 75th percentile
    
    Args:
        sorted_results (list): Sorted results from sort_eval_results()
        n_worst (int): Number of worst performing files to return
        n_avg (int): Number of average performing files to return
        n_75th (int): Number of 75th percentile performing files to return
        max_tokens (int, optional): Maximum token length filter
        
    Returns:
        dict: Dictionary containing lists of filenames for each performance category
    """
    # Filter by token count if specified
    if max_tokens is not None:
        results = [
            (filename, metrics) 
            for filename, metrics in sorted_results
            if metrics['token_length'] <= max_tokens
        ]
    else:
        results = sorted_results
        
    total_files = len(results)
    
    # Get worst performing files (from start of sorted list)
    worst_performing = [filename for filename, _ in results[:n_worst]]
    
    # Get average performing files (from middle of sorted list)
    mid_point = total_files // 2
    start_avg = mid_point - (n_avg // 2)
    end_avg = start_avg + n_avg
    average_performing = [filename for filename, _ in results[start_avg:end_avg]]
    
    # Get 75th percentile performing files
    percentile_75_idx = int(total_files * 0.75)
    start_75th = percentile_75_idx - (n_75th // 2)
    end_75th = start_75th + n_75th
    percentile_75_performing = [filename for filename, _ in results[start_75th:end_75th]]
    
    return {
        'worst_performance': worst_performing,
        'average_performance': average_performing,
        '75th_percentile_performance': percentile_75_performing
    }

def save_file_subsets(worst_overall: dict, worst_filtered: dict, save_path: str, token_cutoff: int) -> None:
    """
    Save the two sets of performance-based file groups to a JSON file.
    
    Args:
        worst_overall (dict): Dict containing lists of files at different performance levels
        worst_filtered (dict): Dict containing token-limited lists of files at different performance levels
        save_path (str): Path to save the JSON file
        token_cutoff (int): Token count cutoff used for filtering
    """
    subsets = {
        'overall_performance_sets': worst_overall,
        f'under_{str(token_cutoff)}_tokens_performance_sets': worst_filtered
    }
    save_json(subsets, save_path)
