import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, Any, Tuple

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
