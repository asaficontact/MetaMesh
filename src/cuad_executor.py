import os
from typing import Dict, Any
from src.utils import load_json, save_json
from src.agents.legal_contract_expert import LegalContractExpert
import json
import tempfile
from rich.console import Console
from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, as_completed

## Constants
DATASET_PREDICTION_SAVE_PATH = 'results/predictions'
MAX_WORKERS = 3

# Initialize rich console
console = Console()

class CUADExecutor:
    """
    Executor class to evaluate LegalContractExpert against CUAD dataset
    """
    def __init__(self, 
                 cuad_data_path: str, 
                 dataset_save_path: str = DATASET_PREDICTION_SAVE_PATH):
        """
        Initialize with path to CUAD dataset
        
        Args:
            cuad_data_path (str): Path to CUAD dataset JSON file
            dataset_save_path (str): Path to save dataset predictions
        """
        console.print("[bold blue]Initializing CUAD Executor[/bold blue]")
        self.cuad_data = load_json(cuad_data_path)
        self.dataset_save_path = dataset_save_path
        console.print(f"Loaded dataset with {len(self.cuad_data)} files")
        
    def _create_category_questions_file(self, file_data: Dict) -> str:
        """
        Create temporary JSON file with category to question mapping
        
        Args:
            file_data (dict): File data containing questions for each category
            
        Returns:
            str: Path to temporary questions file
        """
        category_to_question = {}
        for category, category_data in file_data.items():
            if category == 'document_path':
                continue
            category_to_question[category] = category_data['question']
            
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(category_to_question, temp_file)
        temp_file.close()
        
        return temp_file.name
        
    def process_file(self, 
                     filename: str, 
                     agent: LegalContractExpert, 
                     debug: bool = False,
                     verbose: bool = False) -> Dict[str, Any]:
        """
        Process all category questions for a given file
        
        Args:
            filename (str): Name of file to process
            agent (LegalContractExpert): LLM agent to use for predictions
            debug (bool): Whether to enable debug mode
            verbose (bool): Whether to enable verbose mode
        Returns:
            dict: Predictions for each category question
        """
        if filename not in self.cuad_data:
            raise ValueError(f"File {filename} not found in CUAD dataset")
            
        file_data = self.cuad_data[filename]
        document_path = file_data['document_path']
        
        # Create temporary questions file
        questions_file = self._create_category_questions_file(file_data)
        
        try:
            # Get predictions using answer_questions_list
            predictions = agent.answer_questions_list(
                contract_path=document_path,
                category_to_question_path=questions_file,
                debug=debug,
                verbose=verbose
            )
            
            # Format predictions to include true values
            formatted_predictions = {}
            for category, pred_data in predictions.items():
                formatted_predictions[category] = {
                    'question': file_data[category]['question'],
                    'predicted_answer': pred_data['answer'],
                    'true_value': file_data[category]['answer'],
                    'raw_response': pred_data['raw_response']
                }
                
        finally:
            # Clean up temporary file
            os.unlink(questions_file)
            
        return formatted_predictions

    def _process_condition_file(self, args):
        """Helper function for parallel condition processing"""
        filename, file_data, condition, agent, debug, verbose = args
        if condition not in file_data:
            return filename, None
            
        document_path = file_data['document_path']
        question = file_data[condition]['question']
        true_answer = file_data[condition]['answer']
        
        prediction = agent.answer_contract_question(
            contract_path=document_path,
            question=question,
            debug=debug,
            verbose=verbose
        )
        
        return filename, {
            'question': question,
            'predicted_answer': prediction['answer'],
            'true_value': true_answer,
            'raw_response': prediction['raw_response']
        }
    
    def process_condition(self, 
                          condition: str, 
                          agent: LegalContractExpert, 
                          debug: bool = False, 
                          verbose: bool = False) -> Dict[str, Any]:
        """
        Process a specific condition across all files
        
        Args:
            condition (str): Name of condition to process
            agent (LegalContractExpert): LLM agent to use for predictions
            debug (bool): Whether to enable debug mode
            
        Returns:
            dict: Predictions for condition across all files
        """
        console.print(f"[bold green]Processing condition: {condition}[/bold green]")
        predictions = {}

        # Create arguments for parallel processing
        process_args = [
            (filename, file_data, condition, agent, debug, verbose)
            for filename, file_data in self.cuad_data.items()
        ]
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(self._process_condition_file, args) for args in process_args]
            
            for future in track(as_completed(futures), total=len(futures), description="Processing files"):
                filename, result = future.result()
                if result is not None:
                    predictions[filename] = result
                    
        return predictions
    
    def process_dataset(self, 
                        agent: LegalContractExpert, 
                        debug: bool = False, 
                        verbose: bool = False) -> Dict[str, Any]:
        """
        Process entire CUAD dataset
        
        Args:
            agent (LegalContractExpert): LLM agent to use for predictions
            debug (bool): Whether to enable debug mode
            verbose (bool): Whether to enable verbose mode
        Returns:
            dict: Predictions for all files and conditions
        """
        console.print("[bold blue]Starting dataset processing[/bold blue]")
        predictions = {}
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.process_file, filename, agent, debug, verbose): filename 
                for filename in self.cuad_data.keys()
            }
            
            for future in track(as_completed(futures), total=len(futures), description="Processing files"):
                filename = futures[future]
                try:
                    predictions[filename] = future.result()
                except Exception as e:
                    console.print(f"[bold red]Error processing {filename}: {str(e)}[/bold red]")
        
        # Save predictions
        save_path = os.path.join(self.dataset_save_path, f'{agent.model_name}_baseline.json')
        save_json(predictions, save_path)
        console.print("[bold green]Dataset processing complete![/bold green]")
            
        return predictions 