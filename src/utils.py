"""
Utils Functions Index:

File Operations:
- load_json: Load JSON file into dictionary
- save_json: Save dictionary to JSON file

Performance Analysis:
- sort_eval_results: Sort files by weighted performance metrics (F1, precision, recall, accuracy)
- get_performance_file_sets: Get sets of files based on performance levels with yes/no distribution filtering
- analyze_subset_metrics: Calculate and print average metrics for file subsets
- analyze_true_value_distribution_of_subset: Analyze and print yes/no distribution stats for files

Data Processing:
- performance_subset_cuad_data: Create CUAD dataset subset from performance file sets
- plot_category_distributions: Visualize category distributions in analysis results

Model Analysis:
- analyze_model_results: Analyze model results and return analysis dictionary
- compare_models: Compare two models and plot comparison results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, Any, Tuple, Union, List
import tiktoken
import numpy as np
from rich.console import Console
from rich.table import Table
import tiktoken


console = Console()

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
                              cuad_data: Dict,
                            n_worst: int = 20, 
                            n_avg_plus: int = 20, 
                            max_tokens: int = None) -> dict:
    """
    Get sets of files based on different performance levels with yes/no distribution filtering
    """
    def check_yes_no_distribution(filename: str, print_info: bool = True) -> bool:
        """Check if file has acceptable yes/no distribution in true values"""
        if not cuad_data or filename not in cuad_data:
            if print_info:
                print(f"✗ {filename}: Not found in CUAD data")
            return False
            
        # Count yes/no answers in true values
        yes_count = sum(
            1 for category, data in cuad_data[filename].items()
            if category != 'document_path' and data['answer'].lower() == 'yes'
        )
        total_questions = sum(
            1 for category in cuad_data[filename] 
            if category != 'document_path'
        )
        
        # Calculate yes percentage
        yes_percentage = yes_count / total_questions * 100
        no_percentage = 100 - yes_percentage
        
        # Check if distribution is between 35-65%
        is_acceptable = 35 <= yes_percentage <= 65
        
        if print_info:
            status = "✓" if is_acceptable else "✗"
            print(f"{status} {filename}:")
            print(f"   Yes: {yes_percentage:.1f}% ({yes_count}/{total_questions})")
            print(f"   No:  {no_percentage:.1f}% ({total_questions-yes_count}/{total_questions})")
            print(f"   Status: {'Accepted' if is_acceptable else 'Rejected'}")
            print()
        
        return is_acceptable
    
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
    
    print("\nProcessing worst performing candidates (bottom 25%):")
    print("="*50)
    # Get worst performing files (from start of sorted list)
    worst_candidates = [
        filename for filename, _ in results[:int(total_files * 0.25)]  # Look in bottom 25%
        if check_yes_no_distribution(filename)
    ]
    worst_performing = worst_candidates[:n_worst]
    
    print("\nProcessing average or better performing candidates (top 50%):")
    print("="*50)
    # Get average or better performing files (from middle to end of sorted list)
    mid_point = total_files // 2
    avg_plus_candidates = [
        filename for filename, _ in results[mid_point:]  # Look in top 50%
        if check_yes_no_distribution(filename)
    ]
    average_plus_performing = avg_plus_candidates[:n_avg_plus]
    
    print("\nSummary:")
    print(f"Found {len(worst_performing)}/{n_worst} worst performing files with acceptable distribution")
    print(f"Found {len(average_plus_performing)}/{n_avg_plus} average+ performing files with acceptable distribution")
    
    return {
        'worst_performance': worst_performing,
        'average_plus_performance': average_plus_performing
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

def performance_subset_cuad_data(cuad_data: Dict[str, Any], performance_set: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Create a subset of CUAD data containing only files from the performance set.
    
    Args:
        cuad_data (Dict[str, Any]): Full CUAD dataset
        performance_set (Dict[str, List[str]]): Dictionary containing lists of filenames 
            categorized by performance level (worst, average, 75th percentile)
            
    Returns:
        Dict[str, Any]: Subset of CUAD data containing only the specified files
    """
    subset_data = {}
    missing_files = []
    
    # Process each performance category
    for category, file_list in performance_set.items():
        print(f"\nProcessing {category}:")
        
        for filename in file_list:
            if filename in cuad_data:
                subset_data[filename] = cuad_data[filename]
                print(f"✓ Added: {filename}")
            else:
                missing_files.append((category, filename))
                print(f"✗ Missing: {filename}")
    
    # Report any missing files
    if missing_files:
        print("\nWarning: The following files were not found in CUAD dataset:")
        for category, filename in missing_files:
            print(f"Category: {category}, File: {filename}")
    
    print(f"\nCreated subset with {len(subset_data)} files out of {len(cuad_data)} total files")
    
    return subset_data

def analyze_true_value_distribution_of_subset(subset_data: Dict[str, List[str]], 
                                            cuad_data: Dict[str, Any],
                                            doc_stats: Dict[str, Any] = None) -> None:
    """
    Analyze and print yes/no distribution and length statistics for each file in the performance subsets
    
    Args:
        subset_data (Dict[str, List[str]]): Dictionary containing lists of filenames by performance category
        cuad_data (Dict[str, Any]): Full CUAD dataset
        doc_stats (Dict[str, Any]): Document statistics containing word and token counts
    """
    # Initialize GPT-4 tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4o")
    
    def print_file_distribution(filename: str) -> None:
        """Helper function to print distribution and length stats for a single file"""
        if filename not in cuad_data:
            print(f"✗ {filename}: Not found in CUAD data\n")
            return
            
        # Count yes/no answers in true values
        yes_count = sum(
            1 for category, data in cuad_data[filename].items()
            if category != 'document_path' and data['answer'].lower() == 'yes'
        )
        total_questions = sum(
            1 for category in cuad_data[filename] 
            if category != 'document_path'
        )
        
        # Calculate percentages
        yes_percentage = yes_count / total_questions * 100
        no_percentage = 100 - yes_percentage
        
        # Get length statistics
        stats = doc_stats.get(filename, {}) if doc_stats else {}
        word_count = stats.get('word_length', 'N/A')
        
        # Get token count using tiktoken
        if 'document_path' in cuad_data[filename]:
            try:
                with open(cuad_data[filename]['document_path'], 'r') as f:
                    text = f.read()
                token_count = len(encoding.encode(text))
            except:
                token_count = 'N/A (file not found)'
        else:
            token_count = 'N/A (no path)'
        
        print(f"► {filename}:")
        print(f"   Yes: {yes_percentage:.1f}% ({yes_count}/{total_questions})")
        print(f"   No:  {no_percentage:.1f}% ({total_questions-yes_count}/{total_questions})")
        print(f"   Words: {word_count}")
        print(f"   GPT-4o Tokens: {token_count}")
        print()
    
    # Process each category in the subset
    for category, file_list in subset_data.items():
        print(f"\n{category.upper()} FILES")
        print("="*50)
        print(f"Total files in category: {len(file_list)}\n")
        
        for filename in file_list:
            print_file_distribution(filename)

def analyze_model_results(results_path: str, model_name: str = None, save_plots: bool = False) -> Dict[str, Any]:
    """
    Analyze and visualize results from a single model evaluation
    
    Args:
        results_path (str): Path to model results JSON file
        model_name (str, optional): Name of model to analyze. If None, uses first key in results
        save_plots (bool): Whether to save plots instead of displaying them
        
    Returns:
        dict: Analysis results
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get model name if not provided
    if model_name is None:
        model_name = list(results.keys())[0]
    
    model_results = results[model_name]
    file_metrics = model_results['file_level_metrics']
    avg_metrics = model_results['average_metrics']

    # Print analysis
    console.print(f"\n[bold blue]Analysis Results for {model_name}[/bold blue]")
    
    # Create table for average metrics
    avg_table = Table(title="Average Metrics")
    avg_table.add_column("Metric", style="cyan")
    avg_table.add_column("Value", style="magenta")
    
    for metric, value in avg_metrics.items():
        if isinstance(value, (int, float)):
            avg_table.add_row(metric, f"{value:.4f}")
        elif isinstance(value, dict) and metric == 'token_counts':
            for token_type, count in value.items():
                avg_table.add_row(f"Average {token_type}", f"{count:.4f}")
    
    console.print(avg_table)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Performance Analysis for {model_name}', fontsize=16)
    
    # Convert data to DataFrame for easier plotting
    df_data = {}
    for filename, metrics in file_metrics.items():
        file_data = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'input_tokens': metrics['token_counts']['input_tokens'],
            'output_tokens': metrics['token_counts']['output_tokens'],
            'total_tokens': metrics['token_counts']['total_tokens'],
            'processing_time': metrics['processing_time']
        }
        df_data[filename] = file_data
    
    df = pd.DataFrame.from_dict(df_data, orient='index').reset_index()
    df.index = range(len(df))  # Reset index to numeric values
    
    # 1. Distribution of metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    df[metrics].boxplot(ax=ax1)
    ax1.set_title('Distribution of Metrics')
    ax1.set_ylabel('Score')
    
    # 2. Input tokens vs F1 Score
    ax2.scatter(df['input_tokens'], df['f1_score'])
    ax2.set_xlabel('Input Tokens')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Input Tokens vs F1 Score')
    
    # 3. Performance metrics across files
    df[['accuracy', 'f1_score', 'precision', 'recall']].plot(ax=ax3, marker='o')
    ax3.set_title('Performance Metrics Across Files')
    ax3.set_xlabel('File Index')
    ax3.set_ylabel('Score')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. Individual metric trends
    metrics_df = df[['accuracy', 'f1_score', 'precision', 'recall']]
    metrics_df.plot(ax=ax4, kind='bar', width=0.8)
    ax4.set_title('Performance Metrics by File')
    ax4.set_xlabel('File Index')
    ax4.set_ylabel('Score')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'results/analysis/{model_name}_analysis.png')
    else:
        plt.show()
        
    return {
        'metrics': avg_metrics, 
        'file_metrics': file_metrics,
        'dataframe': df  # Return DataFrame for additional analysis if needed
    }

def compare_models(model1_path: str, model2_path: str, 
                  model1_name: str = None, model2_name: str = None,
                  save_plots: bool = False) -> None:
    """
    Compare and visualize results from two different models
    
    Args:
        model1_path (str): Path to first model's results
        model2_path (str): Path to second model's results
        model1_name (str, optional): Name of first model
        model2_name (str, optional): Name of second model
        save_plots (bool): Whether to save plots instead of displaying them
    """
    # Get analysis for both models
    analysis1 = analyze_model_results(model1_path, model1_name, save_plots=True)
    analysis2 = analyze_model_results(model2_path, model2_name, save_plots=True)
    
    if model1_name is None:
        model1_name = "Model 1"
    if model2_name is None:
        model2_name = "Model 2"
    
    # Print comparison
    console.print(f"\n[bold green]Model Comparison: {model1_name} vs {model2_name}[/bold green]")
    
    # Create comparison table
    comp_table = Table(title="Metric Comparison")
    comp_table.add_column("Metric", style="cyan")
    comp_table.add_column(model1_name, style="magenta")
    comp_table.add_column(model2_name, style="magenta")
    comp_table.add_column("Difference", style="yellow")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        val1 = analysis1['metrics'][metric]
        val2 = analysis2['metrics'][metric]
        diff = val1 - val2
        comp_table.add_row(
            metric,
            f"{val1:.4f}",
            f"{val2:.4f}",
            f"{diff:+.4f}"
        )
    
    console.print(comp_table)
    
    # Create comparison plots - fix the subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))  # Changed from (2, 3) to (2, 2)
    fig.suptitle(f'Model Comparison: {model1_name} vs {model2_name}', fontsize=16)
    
    # Convert data to DataFrames with numeric indices
    df1 = pd.DataFrame.from_dict(analysis1['file_metrics'], orient='index').reset_index()
    df2 = pd.DataFrame.from_dict(analysis2['file_metrics'], orient='index').reset_index()
    df1.index = range(len(df1))
    df2.index = range(len(df2))
    
    # 1. Box plot comparison of metrics
    data_to_plot = {
        f"{model1_name} Accuracy": df1['accuracy'],
        f"{model2_name} Accuracy": df2['accuracy'],
        f"{model1_name} F1": df1['f1_score'],
        f"{model2_name} F1": df2['f1_score']
    }
    pd.DataFrame(data_to_plot).boxplot(ax=ax1)
    ax1.set_title('Distribution of Metrics')
    ax1.set_ylabel('Score')
    
    # 2. Scatter plot of F1 scores
    ax2.scatter(df1['f1_score'], df2['f1_score'])
    ax2.plot([0, 1], [0, 1], 'r--')  # diagonal line
    ax2.set_xlabel(f'{model1_name} F1 Score')
    ax2.set_ylabel(f'{model2_name} F1 Score')
    ax2.set_title('F1 Score Comparison')
    
    # 3. Metric differences across files
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    diff_df = pd.DataFrame()
    for metric in metrics:
        diff_df[metric] = df1[metric] - df2[metric]
    
    diff_df.plot(ax=ax3, marker='o')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title(f'Metric Differences Across Files\n({model1_name} - {model2_name})')
    ax3.set_xlabel('File Index')
    ax3.set_ylabel('Score Difference')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. Bar plot of differences
    diff_df.plot(kind='bar', ax=ax4, width=0.8)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title(f'Metric Differences by File\n({model1_name} - {model2_name})')
    ax4.set_xlabel('File Index')
    ax4.set_ylabel('Score Difference')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'results/analysis/model_comparison.png')
    else:
        plt.show()



def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to cl100k_base encoding if model-specific encoding not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)) 