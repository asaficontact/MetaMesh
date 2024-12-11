
import os
import json
from src.agents.planner import Planner
from src.agents.plan_executor import PlanExecutor

class MetaMesh:
    # Constants for output directories
    PLAN_OUTPUT_DIR = '/Users/tawab/Desktop/columbia/Courses/Fall2024/Practical Deep Learning Systems/Project/MetaMesh/results/plans/gpt-4o'              # Directory to save generated plans
    EXTRACTION_OUTPUT_DIR = '/Users/tawab/Desktop/columbia/Courses/Fall2024/Practical Deep Learning Systems/Project/MetaMesh/results/reps/gpt-4o'  # Directory to save extraction results

    def __init__(self, 
                 model='gpt-4o', 
                 temp=0.7, 
                 debug=False):
        
        self.model = model
        self.temp = temp
        self.debug = debug

    def process_contract(self, contract_path):
        # Ensure that the output directories exist
        os.makedirs(self.PLAN_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.EXTRACTION_OUTPUT_DIR, exist_ok=True)

        # Get the contract filename without extension
        contract_filename = os.path.basename(contract_path)
        contract_name, _ = os.path.splitext(contract_filename)

        # Paths for the plan and extraction results
        plan_output_path = os.path.join(self.PLAN_OUTPUT_DIR, f"{contract_name}_plan.json")
        extraction_output_path = os.path.join(self.EXTRACTION_OUTPUT_DIR, f"{contract_name}_extraction.json")

        # Step 1: Create a plan using the Planner
        planner = Planner(model=self.model, temp=self.temp)
        plan = planner.process_contract(contract_path)
        # Save the plan
        planner.save_plan(plan_output_path)

        # Step 2: Use the PlanExecutor to extract information using the plan
        executor = PlanExecutor(plan_path=plan_output_path,
                                contract_path=contract_path,
                                model=self.model,
                                temp=self.temp,
                                debug=self.debug)
        # Execute the plan
        results = executor.execute_plan()
        # Save the extraction results
        executor.save_results(extraction_output_path)

        # Return the results if needed
        return results
    
    def process_dataset(self, cuad_dataset_path):
        # Load the CUAD dataset JSON file
        with open(cuad_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Ensure that the output directories exist
        os.makedirs(self.PLAN_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.EXTRACTION_OUTPUT_DIR, exist_ok=True)

        # Iterate over all the files in the dataset
        for file_key in dataset.keys():
            file_info = dataset[file_key]
            document_path = file_info.get('document_path')

            if document_path:
                # Resolve the full path to the document
                document_path = os.path.abspath(document_path)

                if os.path.exists(document_path):
                    try:
                        print(f"Processing document: {document_path}")
                        self.process_contract(document_path)
                    except Exception as e:
                        print(f"Error processing {document_path}: {e}")
                else:
                    print(f"Document not found: {document_path}")
            else:
                print(f"No 'document_path' found for key: {file_key}")