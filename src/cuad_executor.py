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
MAX_WORKERS = 11

# Initialize rich console
console = Console()

class CUADExecutor:
    """
    Executor class to evaluate LegalContractExpert against CUAD dataset
    """
    def __init__(self, 
                 cuad_data_path: str, 
                 dataset_save_path: str = DATASET_PREDICTION_SAVE_PATH,
                 use_contract_doc: bool = True,
                 use_intermediate_rep: bool = False,
                 intermediate_rep_dir_path: str = None):
        """
        Initialize with path to CUAD dataset
        
        Args:
            cuad_data_path (str): Path to CUAD dataset JSON file
            dataset_save_path (str): Path to save dataset predictions
            use_contract_doc (bool): Whether to use original contract document
            use_intermediate_rep (bool): Whether to use intermediate representations
            intermediate_rep_dir_path (str): Directory containing intermediate representations
        """
        console.print("[bold blue]Initializing CUAD Executor[/bold blue]")
        
        # Validate parameters
        if use_intermediate_rep and not intermediate_rep_dir_path:
            raise ValueError("intermediate_rep_dir_path must be provided when use_intermediate_rep is True")
            
        if not use_contract_doc and not use_intermediate_rep:
            raise ValueError("At least one of use_contract_doc or use_intermediate_rep must be True")
            
        self.cuad_data = load_json(cuad_data_path)
        self.dataset_save_path = dataset_save_path
        self.use_contract_doc = use_contract_doc
        self.use_intermediate_rep = use_intermediate_rep
        self.intermediate_rep_dir_path = intermediate_rep_dir_path
        
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
        
    def _get_intermediate_rep_path(self, filename: str) -> str:
        """Get path to intermediate representation file for given contract"""
        if not self.intermediate_rep_dir_path:
            return None
            
        # Remove any file extension from filename
        base_filename = os.path.splitext(filename)[0]
            
        # Convert filename to intermediate rep filename by adding _extraction.json
        rep_filename = f"{base_filename}_extraction.json"
        rep_path = os.path.join(self.intermediate_rep_dir_path, rep_filename)
        
        if not os.path.exists(rep_path):
            raise ValueError(f"Intermediate representation not found at: {rep_path}")
            
        return rep_path
        
    def process_file(self, filename: str, agent: LegalContractExpert, debug: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Process all category questions for a given file"""
        if filename not in self.cuad_data:
            raise ValueError(f"File {filename} not found in CUAD dataset")
        
        file_data = self.cuad_data[filename]
        document_path = file_data['document_path'] if self.use_contract_doc else None
        
        # Get intermediate rep path if needed
        intermediate_rep_path = None
        if self.use_intermediate_rep:
            intermediate_rep_path = self._get_intermediate_rep_path(filename)
            print(f"Intermediate rep path: {intermediate_rep_path}")
        
        # Verify contract file exists if using it
        if document_path and not os.path.exists(document_path):
            raise ValueError(f"Contract file not found at path: {document_path}")
        
        # Create temporary questions file
        questions_file = self._create_category_questions_file(file_data)
        
        try:
            # Choose appropriate method based on configuration
            if self.use_intermediate_rep and self.use_contract_doc:
                # Use both sources
                predictions = agent.answer_questions_list_with_intermediate_rep(
                    intermediate_rep_path=intermediate_rep_path,
                    category_to_question_path=questions_file,
                    contract_path=document_path,
                    debug=debug,
                    verbose=verbose
                )
            elif self.use_intermediate_rep:
                # Use only intermediate representation
                predictions = agent.answer_questions_list_with_intermediate_rep(
                    intermediate_rep_path=intermediate_rep_path,
                    category_to_question_path=questions_file,
                    debug=debug,
                    verbose=verbose
                )
            else:
                # Use only contract document (original behavior)
                predictions = agent.answer_questions_list(
                    contract_path=document_path,
                    category_to_question_path=questions_file,
                    debug=debug,
                    verbose=verbose
                )
            
            if not predictions:
                raise ValueError(f"No predictions returned for file {filename}")
            
            # Format predictions and collect metrics
            formatted_predictions = {}
            total_input_tokens = 0
            total_output_tokens = 0
            total_processing_time = 0
            metrics_by_category = {}
            
            for category, pred_data in predictions.items():
                if category == 'metrics':  # Skip the overall metrics
                    continue
                    
                formatted_predictions[category] = {
                    'question': file_data[category]['question'],
                    'predicted_answer': pred_data['answer'],
                    'true_value': file_data[category]['answer'],
                    'raw_response': pred_data['raw_response']
                }
                
                # Safely get metrics
                if 'metrics' in pred_data:
                    formatted_predictions[category]['metrics'] = pred_data['metrics']
                    metrics = pred_data['metrics']
                    token_count = metrics.get('token_count', {})
                    total_input_tokens += token_count.get('input_tokens', 0)
                    total_output_tokens += token_count.get('output_tokens', 0)
                    total_processing_time += metrics.get('processing_time', 0)
                    metrics_by_category[category] = metrics
            
            # Calculate averages
            num_categories = len(formatted_predictions)
            if num_categories > 0:
                formatted_predictions['metrics'] = {
                    'by_category': metrics_by_category,
                    'averages': {
                        'average_input_tokens': total_input_tokens / num_categories,
                        'average_output_tokens': total_output_tokens / num_categories,
                        'average_processing_time': total_processing_time / num_categories,
                        'total_categories': num_categories
                    }
                }
            
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
            
        document_path = file_data['document_path'] if self.use_contract_doc else None
        question = file_data[condition]['question']
        true_answer = file_data[condition]['answer']
        
        try:
            # Get intermediate rep path if needed
            intermediate_rep_path = None
            if self.use_intermediate_rep:
                intermediate_rep_path = self._get_intermediate_rep_path(filename)
            
            # Choose appropriate method based on configuration
            if self.use_intermediate_rep and self.use_contract_doc:
                # Use both sources
                prediction = agent.answer_contract_question_with_intermediate_rep(
                    intermediate_rep_path=intermediate_rep_path,
                    question=question,
                    contract_path=document_path,
                    debug=debug,
                    verbose=verbose
                )
            elif self.use_intermediate_rep:
                # Use only intermediate representation
                prediction = agent.answer_contract_question_with_intermediate_rep(
                    intermediate_rep_path=intermediate_rep_path,
                    question=question,
                    debug=debug,
                    verbose=verbose
                )
            else:
                # Use only contract document (original behavior)
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
            
        except Exception as e:
            if debug:
                print(f"Error processing condition for {filename}: {str(e)}")
            return filename, None
    
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
    
    def process_dataset(self, model_name: str, provider: str, debug: bool = False, verbose: bool = False) -> Dict[str, Any]:
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
                        total_files += 1
                        
                        # Aggregate metrics from each file
                        if 'metrics' in result:
                            file_metrics = result['metrics']
                            if 'averages' in file_metrics:
                                total_input_tokens += file_metrics['averages'].get('average_input_tokens', 0)
                                total_output_tokens += file_metrics['averages'].get('average_output_tokens', 0)
                                
                except Exception as e:
                    console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        
        # Add dataset-level token count averages
        if total_files > 0:
            predictions['token_count'] = {
                'average_input_tokens': total_input_tokens / total_files,
                'average_output_tokens': total_output_tokens / total_files,
                'total_files_processed': total_files
            }
        else:
            predictions['token_count'] = {
                'average_input_tokens': 0,
                'average_output_tokens': 0,
                'total_files_processed': 0
            }
        
        # Save predictions
        # Determine file suffix based on configuration
        if self.use_contract_doc and self.use_intermediate_rep:
            suffix = 'rep_contract_pred.json'
        elif self.use_contract_doc:
            suffix = 'contract_pred.json'
        elif self.use_intermediate_rep:
            suffix = 'rep_pred.json'
        else:
            suffix = 'pred.json'
            
        save_path = os.path.join(self.dataset_save_path, f'{model_name}_{suffix}')
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