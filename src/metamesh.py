import os
import json
from src.agents.planner import Planner
from src.agents.plan_executor import PlanExecutor
from datetime import datetime
from multiprocessing import Pool, cpu_count

class MetaMesh:
    # Base output directories
    BASE_DIR = '/Users/tawab/Desktop/columbia/Courses/Fall2024/Practical Deep Learning Systems/Project/MetaMesh/results'
    PLAN_DIR = 'plans'
    EXTRACTION_DIR = 'reps'
    TOKEN_COUNT_DIR = 'token_counts'
    TIME_DIR = 'time'
    
    # Parallel processing settings
    MAX_WORKERS = cpu_count() - 1  
    
    def __init__(self, 
                 planner_model='gpt-4o',
                 executor_model='gpt-4o', 
                 planner_temp=0.7,
                 executor_temp=0.7,
                 debug=False):
        
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.planner_temp = planner_temp
        self.executor_temp = executor_temp
        self.debug = debug
        self.token_counts = {}
        self.processing_times = {}
        
        # Create timestamped model directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_timestamp_dir = f"{self.planner_model}_{self.executor_model}_{timestamp}"
        
        # Set up output directories with model and timestamp
        self.PLAN_OUTPUT_DIR = os.path.join(self.BASE_DIR, self.PLAN_DIR, self.model_timestamp_dir)
        self.EXTRACTION_OUTPUT_DIR = os.path.join(self.BASE_DIR, self.EXTRACTION_DIR, self.model_timestamp_dir)
        self.TOKEN_COUNT_OUTPUT_DIR = os.path.join(self.BASE_DIR, self.TOKEN_COUNT_DIR, self.model_timestamp_dir)
        self.TIME_OUTPUT_DIR = os.path.join(self.BASE_DIR, self.TIME_DIR, self.model_timestamp_dir)
        
        # Create all output directories
        for dir_path in [self.PLAN_OUTPUT_DIR, self.EXTRACTION_OUTPUT_DIR, 
                        self.TOKEN_COUNT_OUTPUT_DIR, self.TIME_OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            
    def _save_metrics(self, metrics: dict, output_dir: str, filename: str):
        """Save metrics to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_path}")
        
    def process_contract(self, contract_path):
        # Get the contract filename without extension
        contract_filename = os.path.basename(contract_path)
        contract_name, _ = os.path.splitext(contract_filename)

        # Paths for the plan and extraction results
        plan_output_path = os.path.join(self.PLAN_OUTPUT_DIR, f"{contract_name}_plan.json")
        extraction_output_path = os.path.join(self.EXTRACTION_OUTPUT_DIR, f"{contract_name}_extraction.json")

        # Step 1: Create a plan using the Planner with planner model and temp
        planner = Planner(model=self.planner_model, 
                         temp=self.planner_temp, 
                         debug=self.debug)
        plan = planner.process_contract(contract_path)
        planner.save_plan(plan_output_path)

        # Step 2: Use the PlanExecutor with executor model and temp
        executor = PlanExecutor(plan_path=plan_output_path,
                              contract_path=contract_path,
                              model=self.executor_model,
                              temp=self.executor_temp,
                              debug=self.debug)
        results = executor.execute_plan()
        executor.save_results(extraction_output_path)

        # Collect and save token counts
        token_counts = {
            "planner_input_token_count": planner.token_counts["input_tokens"],
            "planner_output_token_count": planner.token_counts["output_tokens"],
            "plan_executor_input_token_count": executor.token_counts["input_tokens"],
            "plan_executor_output_token_count": executor.token_counts["output_tokens"],
            "planner_model": self.planner_model,
            "executor_model": self.executor_model
        }
        self._save_metrics(token_counts, self.TOKEN_COUNT_OUTPUT_DIR, f"{contract_name}_token_counts.json")
        
        # Collect and save processing times
        processing_times = {
            "planner_processing_time": planner.llm_processing_time,
            "planner_execution_processing_time": sum(executor.processing_times.values()),
            "planner_execution_time_breakdown": executor.processing_times,
            "total_llm_processing_time": planner.llm_processing_time + sum(executor.processing_times.values()),
            "planner_model": self.planner_model,
            "executor_model": self.executor_model
        }
        self._save_metrics(processing_times, self.TIME_OUTPUT_DIR, f"{contract_name}_processing_times.json")

        return results
    
    def _process_single_contract(self, contract_info):
        """Process a single contract - used for both serial and parallel processing."""
        try:
            document_path = contract_info.get('document_path')
            if not document_path or not os.path.exists(document_path):
                return None
            
            document_path = os.path.abspath(document_path)
            print(f"Processing document: {document_path}")
            
            # Process the contract
            results = self.process_contract(document_path)
            
            # Get metrics
            contract_name = os.path.splitext(os.path.basename(document_path))[0]
            
            # Load time metrics
            time_file_path = os.path.join(self.TIME_OUTPUT_DIR, f"{contract_name}_processing_times.json")
            with open(time_file_path, 'r') as f:
                times = json.load(f)
            
            # Load token metrics
            token_file_path = os.path.join(self.TOKEN_COUNT_OUTPUT_DIR, f"{contract_name}_token_counts.json")
            with open(token_file_path, 'r') as f:
                tokens = json.load(f)
            
            return {
                'times': times,
                'tokens': tokens,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing {document_path}: {e}")
            return None
    
    def process_dataset(self, cuad_dataset_path, run_parallel=False):
        """Process a dataset of contracts with optional parallel processing."""
        # Load the CUAD dataset JSON file
        with open(cuad_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Initialize counters
        total_planner_time = 0
        total_executor_time = 0
        total_planner_input_tokens = 0
        total_planner_output_tokens = 0
        total_executor_input_tokens = 0
        total_executor_output_tokens = 0
        processed_count = 0
        
        if run_parallel:
            # Prepare contract info list for parallel processing
            contract_info_list = [
                {'document_path': info.get('document_path')} 
                for info in dataset.values()
            ]
            
            # Process contracts in parallel
            with Pool(processes=self.MAX_WORKERS) as pool:
                results = pool.map(self._process_single_contract, contract_info_list)
                
            # Aggregate results from parallel processing
            for result in results:
                if result is not None:
                    times = result['times']
                    tokens = result['tokens']
                    
                    # Aggregate times
                    total_planner_time += times["planner_processing_time"]
                    total_executor_time += times["planner_execution_processing_time"]
                    
                    # Aggregate tokens
                    total_planner_input_tokens += tokens["planner_input_token_count"]
                    total_planner_output_tokens += tokens["planner_output_token_count"]
                    total_executor_input_tokens += tokens["plan_executor_input_token_count"]
                    total_executor_output_tokens += tokens["plan_executor_output_token_count"]
                    
                    processed_count += 1
        else:
            # Serial processing
            for file_info in dataset.values():
                result = self._process_single_contract(file_info)
                if result is not None:
                    times = result['times']
                    tokens = result['tokens']
                    
                    # Aggregate times
                    total_planner_time += times["planner_processing_time"]
                    total_executor_time += times["planner_execution_processing_time"]
                    
                    # Aggregate tokens
                    total_planner_input_tokens += tokens["planner_input_token_count"]
                    total_planner_output_tokens += tokens["planner_output_token_count"]
                    total_executor_input_tokens += tokens["plan_executor_input_token_count"]
                    total_executor_output_tokens += tokens["plan_executor_output_token_count"]
                    
                    processed_count += 1

        # Calculate and save averages if any files were processed
        if processed_count > 0:
            # Calculate average times
            average_times = {
                "average_planner_processing_time": total_planner_time / processed_count,
                "average_executor_processing_time": total_executor_time / processed_count,
                "average_total_processing_time": (total_planner_time + total_executor_time) / processed_count,
                "total_files_processed": processed_count,
                "parallel_processing": run_parallel,
                "num_workers": self.MAX_WORKERS if run_parallel else 1
            }
            
            # Calculate average token counts
            average_tokens = {
                "average_planner_input_tokens": total_planner_input_tokens / processed_count,
                "average_planner_output_tokens": total_planner_output_tokens / processed_count,
                "average_executor_input_tokens": total_executor_input_tokens / processed_count,
                "average_executor_output_tokens": total_executor_output_tokens / processed_count,
                "average_total_tokens": (total_planner_input_tokens + total_planner_output_tokens + 
                                       total_executor_input_tokens + total_executor_output_tokens) / processed_count,
                "total_files_processed": processed_count,
                "parallel_processing": run_parallel,
                "num_workers": self.MAX_WORKERS if run_parallel else 1
            }
            
            # Save the averages
            self._save_metrics(average_times, self.TIME_OUTPUT_DIR, "average_time_calculations.json")
            self._save_metrics(average_tokens, self.TOKEN_COUNT_OUTPUT_DIR, "average_token_counts.json")