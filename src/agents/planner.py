# planner.py

from pydantic import Field
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import re
import time
from src.utils import count_tokens

load_dotenv()

# Set the OpenAI API key
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY

def _load_prompt_from_file(prompt_path: str) -> str:
    """
    Load prompt text from a file.
    
    Args:
        prompt_path (str): Path to the prompt text file
        
    Returns:
        str: Content of the prompt file
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
    except Exception as e:
        raise Exception(f"Error loading prompt file: {str(e)}")



PLANNER_SYSTEM_PROMPT = _load_prompt_from_file('src/agents/prompts/planner.txt')


class DataPoint(BaseModel):
    data_point_name: str = Field(description="The name of the data point that the agent will collect.")
    data_point_description: str = Field(description="A description of the data point for the agent so it knows what information to collect for the data point.")
    data_point_questions: List[str] = Field(description="A list of questions that will help guide agents when collecting information about the data point.")
    data_point_instructions: str = Field(description="Instructions for the agent on how to collect the data point.")

class AgentInfo(BaseModel):
    agent_name: str = Field(description="The name of the agent that will collect data points for a section.")
    agent_instructions: str = Field(description="Detailed instructions for the agent on how to collect data points for a section. This should include the agent's role, the section it is collecting data points for, information about the data points and how to collect them, the thought process the agent should follow, and any other relevant information.")

class TemplateSection(BaseModel):
    section_name: str = Field(description="The name of the section that the agent will collect data points for.")
    section_description: str = Field(description="A description of the section for the agent so it knows what information is expected to be collected in the section.")
    data_points: List[DataPoint] = Field(description="A list of data points that the agent will collect for the section.")
    section_agent: AgentInfo = Field(description="The agents that will collect data points for the section.")

class Plan(BaseModel):
    introduction: str = Field(description="The introduction to the template that the agent will use to understand the overall structure of the template.")
    sections: List[TemplateSection] = Field(description="A list of sections and the agents that will collect data points for the sections.")

def clean_response_text(response_text):
    # Remove leading and trailing whitespace
    response_text = response_text.strip()
    
    # Check for leading code fences
    if response_text.startswith('```json'):
        response_text = response_text[len('```json'):].strip()
    elif response_text.startswith('```'):
        response_text = response_text[len('```'):].strip()
    
    # Remove trailing code fences if present
    if response_text.endswith('```'):
        response_text = response_text[:-len('```')].strip()
    
    return response_text

# OpenAI API call functions
def oai_call(system_prompt, user_prompt, model="gpt-4o", temp=0.7):
    if 'gpt' in model:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            response_format = Plan,
        )
    
        return response.choices[0].message.parsed
    
    elif 'o1' in model:
        prompt = system_prompt + '\n\n' + user_prompt
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        response_text = clean_response_text(response_text)
        try: 
            parsed_response = json.loads(response_text)
            return parsed_response
        except Exception as e:
            raise Exception(f"Error parsing response: {e}")


class Planner: 
    def __init__(self, 
                 model: str = None, 
                 temp: float = 0.7,
                 debug: bool = False,
                 api_key: str = OPENAI_API_KEY):
        
        self.model = model
        self.temp = temp
        self.debug = debug
        self.api_key = api_key
        self.plan: Optional[Plan] = None
        self.token_counts = {
            "input_tokens": 0,
            "output_tokens": 0
        }
        self.llm_processing_time = 0
    
    def _load_file(self, file_path: str) -> str:
        """Load text content from a file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            print(f"Successfully loaded content from {file_path}")
            return content
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {e}")

    def _count_tokens(self, system_prompt: str, user_prompt: str) -> dict:
        """Count tokens for input prompts."""
        input_tokens = count_tokens(system_prompt + user_prompt, self.model)
        return input_tokens
    
    def _count_output_tokens(self, response: str) -> int:
        """Count tokens in the response."""
        return count_tokens(response, self.model)
    
    def generate_plan(self, contract_text: str):
        """Generate the plan by analyzing the contract and making an OpenAI API call."""
        user_prompt = f"""
            Please analyze the following contract and perform the tasks as per your instructions.

            ## Contract Document

            {contract_text}
        """

        if self.debug:
            print("\nGenerating Plan:")
            print("\nSystem Prompt:")
            print(PLANNER_SYSTEM_PROMPT)
            print("\nUser Prompt:")
            print(user_prompt)
        else:
            print("Generating plan...")
        
        # Count input tokens
        input_tokens = self._count_tokens(PLANNER_SYSTEM_PROMPT, user_prompt)
        self.token_counts["input_tokens"] = input_tokens
        
        if self.debug:
            print(f"\nInput token count: {input_tokens}")
        
        # Time the LLM call
        start_time = time.time()
        response = oai_call(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            temp=self.temp
        )
        end_time = time.time()
        
        # Record processing time
        self.llm_processing_time = end_time - start_time
        
        if self.debug:
            print(f"\nLLM Processing time: {self.llm_processing_time:.2f} seconds")
        
        # Count output tokens
        if isinstance(response, str):
            output_tokens = self._count_output_tokens(response)
        elif isinstance(response, dict):
            output_tokens = self._count_output_tokens(json.dumps(response))
        elif isinstance(response, Plan):
            output_tokens = self._count_output_tokens(json.dumps(response.dict()))
        else: 
            raise ValueError(f"Unsupported response type: {type(response)}")
        
        self.token_counts["output_tokens"] = output_tokens
        
        if self.debug:
            print(f"Output token count: {output_tokens}")
            print("\nPlanner Response:")
            if isinstance(response, str): 
                print(response)
            elif isinstance(response, dict):
                print(json.dumps(response, indent=2))
            elif isinstance(response, Plan):
                print(json.dumps(response.dict(), indent=2))
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")
        
        self.plan = response
        return self.plan
    

    def process_contract(self, contract_path: str):
        """Process the contract and generate the plan."""
        contract_text = self._load_file(contract_path)
        return self.generate_plan(contract_text)
    
    def save_plan(self, output_file_path: str):
        """Save the generated plan to a JSON file."""
        if self.plan is None:
            raise ValueError("Plan not generated yet. Run `process_contract` first.")

        try:
            with open(output_file_path, 'w') as f:
                if isinstance(self.plan, Plan):
                    json.dump(self.plan.dict(), f, indent=2)
                elif isinstance(self.plan, dict):
                    json.dump(self.plan, f, indent=2)
                else:
                    raise ValueError(f"Unsupported plan type: {type(self.plan)}")
                    
            if self.debug:
                print(f"\nPlan saved successfully to: {output_file_path}")
            else:
                print(f"Plan saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving plan to file: {e}")

