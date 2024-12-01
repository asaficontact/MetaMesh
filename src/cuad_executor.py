import os
from typing import Dict, Any, List, Tuple
from src.utils import load_json, save_json
from src.agents.legal_contract_expert import LegalContractExpert
import json
import tempfile
from rich.console import Console
from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, as_completed

## Constants
DATASET_PREDICTION_SAVE_PATH = 'results/predictions'
MAX_WORKERS = 8

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
        
    def process_file(self, filename: str, agent: LegalContractExpert, debug: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Process all category questions for a given file"""
        if filename not in self.cuad_data:
            raise ValueError(f"File {filename} not found in CUAD dataset")
        
        file_data = self.cuad_data[filename]
        document_path = file_data['document_path']
        
        # Verify file exists and is readable
        if not os.path.exists(document_path):
            raise ValueError(f"Contract file not found at path: {document_path}")
        
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
            
            if not predictions:
                raise ValueError(f"No predictions returned for file {filename}")
            
            # Check to make sure we got prediction for all 31 questions

            
            # Format predictions to include true values
            formatted_predictions = {}
            for category, pred_data in predictions.items():
                if category == 'token_count':
                    formatted_predictions['token_count'] = pred_data
                    continue
                    
                formatted_predictions[category] = {
                    'question': file_data[category]['question'],
                    'predicted_answer': pred_data['answer'],
                    'true_value': file_data[category]['answer'],
                    'raw_response': pred_data['raw_response']
                }
                
            if len(formatted_predictions) <= 1:  # Only token_count present
                raise ValueError(f"No valid predictions for file {filename}")
            
            return formatted_predictions
            
        except Exception as e:
            if debug:
                print(f"Error processing file {filename}: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            os.unlink(questions_file)
        
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
        
        result = {
            'question': question,
            'predicted_answer': prediction['answer'],
            'true_value': true_answer,
            'raw_response': prediction['raw_response']
        }
        
        # Include token count if available
        if 'token_count' in prediction:
            result['token_count'] = prediction['token_count']
        
        return filename, result
    
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
                        model_name: str,
                        provider: str,
                        debug: bool = False, 
                        verbose: bool = False) -> Dict[str, Any]:
        """Process entire CUAD dataset"""
        console.print("[bold blue]Starting dataset processing[/bold blue]")
        predictions = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_files = 0
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(self._process_single_file, 
                              filename, 
                              model_name, 
                              provider, 
                              debug, 
                              verbose) 
                for filename in self.cuad_data.keys()
            ]
            
            for future in track(as_completed(futures), total=len(futures), description="Processing files"):
                try:
                    filename, result = future.result()
                    if result is not None:
                        predictions[filename] = result
                        # Aggregate raw token counts, not averages
                        if 'token_count' in result:
                            total_input_tokens += result['token_count'].get('input_tokens', 0)
                            total_output_tokens += result['token_count'].get('output_tokens', 0)
                            total_files += 1
                except Exception as e:
                    console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        
        # Add dataset-level token count averages
        if total_files > 0:
            predictions['token_count'] = {
                'average_input_tokens': total_input_tokens,  # Also store averages
                'average_output_tokens': total_output_tokens,
                'total_files_processed': total_files
            }
        else:
            predictions['token_count'] = {
                'average_input_tokens': -1,
                'average_output_tokens': -1,
                'total_files_processed': 0
            }
        
        # Save predictions
        save_path = os.path.join(self.dataset_save_path, f'{model_name}_baseline.json')
        save_json(predictions, save_path)
        console.print("[bold green]Dataset processing complete![/bold green]")
            
        return predictions
    
    def process_files_list(self, 
                            file_list: List[str],
                            model_name: str,
                            provider: str,
                            debug: bool = False,
                            verbose: bool = False) -> Dict[str, Any]:
        """Process a specific list of files from CUAD dataset in parallel"""
        console.print("[bold blue]Starting processing of specified files[/bold blue]")
        predictions = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_files = 0
        
        # Validate files exist in dataset
        valid_files = [f for f in file_list if f in self.cuad_data]
        if len(valid_files) != len(file_list):
            missing_files = set(file_list) - set(valid_files)
            console.print(f"[bold yellow]Warning: {len(missing_files)} files not found in dataset[/bold yellow]")
            for f in missing_files:
                console.print(f"[yellow]Missing file: {f}[/yellow]")
        
        if not valid_files:
            raise ValueError("No valid files to process")
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(self._process_single_file, 
                              filename, 
                              model_name, 
                              provider, 
                              debug, 
                              verbose) 
                for filename in valid_files
            ]
            
            for future in track(as_completed(futures), 
                              total=len(futures), 
                              description="Processing files"):
                try:
                    filename, result = future.result()
                    if result is not None:
                        predictions[filename] = result
                        # Aggregate token counts
                        if 'token_count' in result:
                            total_input_tokens += result['token_count'].get('average_input_tokens', 0)
                            total_output_tokens += result['token_count'].get('average_output_tokens', 0)
                            total_files += 1
                except Exception as e:
                    console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        
        # Add file list-level token count averages
        if total_files > 0:
            predictions['token_count'] = {
                'average_input_tokens': total_input_tokens / total_files,
                'average_output_tokens': total_output_tokens / total_files,
                'total_files_processed': total_files
            }
        else:
            predictions['token_count'] = {
                'average_input_tokens': -1,
                'average_output_tokens': -1,
                'total_files_processed': 0
            }
        
        console.print("[bold green]Completed processing specified files![/bold green]")
        return predictions
    
    def _process_single_file(self, filename: str, model_name: str, provider: str, debug: bool = False, verbose: bool = False) -> Tuple[str, Dict]:
        """Helper function to process a single file with a new agent instance"""
        # Create agent inside the worker process
        agent = LegalContractExpert(provider=provider, model_name=model_name)
        try:
            if debug:
                print(f"\nDEBUG: Processing file {filename}")
            result = self.process_file(filename, agent, debug, verbose)
            if debug:
                print(f"DEBUG: Result structure for {filename}:")
                print(f"DEBUG: Keys in result: {result.keys() if result else 'None'}")
                if result:
                    for category, data in result.items():
                        print(f"DEBUG: Category {category} data structure: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                if len(result) <= 31: # This is to ensure we get all answers for 31 questions in each file
                    raise ValueError(f"Expected 31 categories, got {len(result)}")
            return filename, result
        except Exception as e:
            if debug:
                print(f"DEBUG: Exception in _process_single_file for {filename}: {str(e)}")
                import traceback
                print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
            return filename, None