import os
from typing import Dict, Any
from src.utils import load_json, save_json
from src.agents.legal_contract_expert import LegalContractExpert
import json
import tempfile
from rich.console import Console
from rich.progress import track

## Constants
DATASET_PREDICTION_SAVE_PATH = 'results/predictions'

# Initialize rich console
console = Console()

class CUADExecutor:
    """
    Executor class to evaluate LegalContractExpert against CUAD dataset
    """
    def __init__(self, cuad_data_path: str, dataset_save_path: str = DATASET_PREDICTION_SAVE_PATH):
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
        
    def process_file(self, filename: str, agent: LegalContractExpert, debug: bool = False) -> Dict[str, Any]:
        """
        Process all category questions for a given file
        
        Args:
            filename (str): Name of file to process
            agent (LegalContractExpert): LLM agent to use for predictions
            debug (bool): Whether to enable debug mode
            
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
                debug=debug
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
    
    def process_condition(self, condition: str, agent: LegalContractExpert, debug: bool = False) -> Dict[str, Any]: #TODO: The answer_questions_list function will not work here as the questions list must be for the same contract file and not across. 
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
        
        for filename, file_data in track(self.cuad_data.items(), description="Processing files"):
            if condition not in file_data:
                continue
                
            document_path = file_data['document_path']
            
            # Create category to question mapping for single condition
            category_to_question = {condition: file_data[condition]['question']}
            
            # Create temporary questions file
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(category_to_question, temp_file)
            temp_file.close()
            
            try:
                # Get predictions
                pred = agent.answer_questions_list(
                    contract_path=document_path,
                    category_to_question_path=temp_file.name,
                    debug=debug
                )
                
                # Format prediction
                predictions[filename] = {
                    'question': file_data[condition]['question'],
                    'predicted_answer': pred[condition]['answer'],
                    'true_value': file_data[condition]['answer'],
                    'raw_response': pred[condition]['raw_response']
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
                
        return predictions
    
    def process_dataset(self, agent: LegalContractExpert, debug: bool = False) -> Dict[str, Any]:
        """
        Process entire CUAD dataset
        
        Args:
            agent (LegalContractExpert): LLM agent to use for predictions
            debug (bool): Whether to enable debug mode
            
        Returns:
            dict: Predictions for all files and conditions
        """
        console.print("[bold blue]Starting dataset processing[/bold blue]")
        predictions = {}
        
        for filename in track(self.cuad_data.keys(), description="Processing files"):
            predictions[filename] = self.process_file(filename, agent, debug=debug)
        
        # Save predictions
        save_path = os.path.join(self.dataset_save_path, f'{agent.model_name}_baseline.json')
        save_json(predictions, save_path)
        console.print("[bold green]Dataset processing complete![/bold green]")
            
        return predictions 