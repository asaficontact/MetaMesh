# planner.py

from pydantic import Field
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import re

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


def _parse_response(response_text: str) -> Optional[Plan]:
    """Parse the assistant's response into a Plan object."""
    try:
        # Use regex to extract JSON object
        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            plan_dict = json.loads(json_str)
            # Validate and parse using Pydantic
            plan = Plan(**plan_dict)
            return plan
        else:
            print("Failed to extract JSON from the assistant's response.")
            return None
    except Exception as e:
        print(f"Error parsing assistant's response: {e}")
        return None


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
        parsed_response = _parse_response(response_text)
        return response_text


class Planner: 
    def __init__(self, 
                 model: str = None, 
                 temp: float = 0.7,
                 api_key: str = OPENAI_API_KEY):
        
        self.model = model
        self.temp = temp
        self.api_key = api_key
        self.plan: Optional[Plan] = None
    
    def _load_file(self, file_path: str) -> str:
        """Load text content from a file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            print(f"Successfully loaded content from {file_path}")
            return content
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {e}")

    
    def generate_plan(self, contract_text: str):
        """Generate the plan by analyzing the contract and making an OpenAI API call."""

        # Prepare the user prompt with the contract text
        user_prompt = f"""
            Please analyze the following contract and perform the tasks as per your instructions.

            ## Contract Document

            {contract_text}
        """ 

        print("Generating plan...")

        # Make the OpenAI API call
        response = oai_call(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            temp=self.temp
        )

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
                json.dump(self.plan.dict(), f, indent=2)
            print(f"Plan saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving plan to file: {e}")


    

        
        