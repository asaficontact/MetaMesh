

from pydantic import Field
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import json
import re

load_dotenv()

# Set the OpenAI API key
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY


class DataPoint(BaseModel):
    data_point_name: str = Field(description="The name of the data point for which the agent collected the information. Should be copied verbatim from the data point name in the plan.")
    data_point_description: str = Field(description="A description of what is contained in the data point. Should be copied verbatim from the data point description in the plan.")
    data_point_questions: List[str] = Field(description="A list of questions that the agent used to collect the data point. Should be copied verbatim from the data point questions in the plan.")
    data_point_overview: str = Field(description="A summary of all the information that was collected for the datapoint by the agent.")
    data_point_answers: str = Field(description="The answer to the data point questions. The agent should include all the answers for the data point questions in the data point answers. Write each answer in a new paragraph.")
    data_point_verbatim: str = Field(description="The verbatim text from the contract that the agent used to collect the data point. Should be copied verbatim from the contract.")


class ExtractedData(BaseModel):
    data_points: List[DataPoint] = Field(description="A list of data points that were extracted from the contract by the agent.")


class AgentInfo(BaseModel):
    agent_name: str
    agent_instructions: str

class TemplateSection(BaseModel):
    section_name: str
    section_description: str
    data_points: List[Dict]  # Will be converted to DataPoint objects when necessary
    section_agent: AgentInfo

class Plan(BaseModel):
    introduction: str
    sections: List[TemplateSection]

class SectionResult(BaseModel):
    section_name: str
    section_description: str
    data_points: List[DataPoint]


def _parse_response(response_text: str) -> Optional[ExtractedData]:
    """Parse the assistant's response into a Plan object."""
    try:
        # Use regex to extract JSON object
        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            plan_dict = json.loads(json_str)
            # Validate and parse using Pydantic
            plan = ExtractedData(**plan_dict)
            return plan
        else:
            print("Failed to extract JSON from the assistant's response.")
            return None
    except Exception as e:
        print(f"Error parsing assistant's response: {e}")
        return None

def oai_call(system_prompt, user_prompt, model="gpt-4o", temp=0.7):
    if 'gpt' in model:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            response_format = ExtractedData,
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
    
class PlanExecutor:
    def __init__(self, 
                 plan_path, 
                 contract_path, 
                 model='gpt-4', 
                 temp=0.7,
                 debug=False):
        
        self.results = {
            "introduction": "",
            "sections": []
        }
        
        self.plan_path = plan_path
        self._load_plan()
        self.contract_path = contract_path
        self._load_contract()
        self.model = model
        self.temp = temp
        self.debug = debug
        
    
    def _load_plan(self):
        """Load the plan JSON file."""
        with open(self.plan_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
            self.plan = Plan(**plan_data)
        self.results["introduction"] = self.plan.introduction
        print(f"Loaded plan from {self.plan_path}")
    
    def _load_contract(self):
        """Load the contract text file."""
        with open(self.contract_path, 'r', encoding='utf-8') as f:
            self.contract_text = f.read()
        print(f"Loaded contract from {self.contract_path}")
    
    def execute_plan(self):
        """Execute the plan by processing each section."""
        for section in self.plan.sections:
            print(f"Processing section: {section.section_name}")
            section_result = self.process_section(section)
            self.results["sections"].append(section_result.dict())
        
        # Check if all sections from the plan are in the results
        plan_section_names = {section.section_name for section in self.plan.sections}
        result_section_names = {section["section_name"] for section in self.results["sections"]}
        
        missing_sections = plan_section_names - result_section_names
        if missing_sections:
            raise ValueError(f"Missing sections in response: {', '.join(missing_sections)}")
            
        return self.results

    def process_section(self, section: TemplateSection):
        """Process a single section by generating prompts and extracting data."""
        # Generate the system and user prompts
        system_prompt = self.generate_agent_system_prompt(section)
        user_prompt = self.generate_agent_user_prompt()

        if self.debug:
            print(f"\nAgent: {section.section_agent.agent_name}")
            print("\nSystem Prompt:")
            print(system_prompt)

        # Call the OpenAI API
        response_text = oai_call(system_prompt, user_prompt, model=self.model, temp=self.temp)

        if self.debug:
            print("\nAgent Response:")
            print(response_text)

        if 'o1' in self.model:
            response_json = json.loads(response_text)
            extracted_data_points = response_json['data_points']
        elif 'gpt' in self.model:
            extracted_data_points = response_text.data_points

        # Create SectionResult
        section_result = SectionResult(
            section_name=section.section_name,
            section_description=section.section_description,
            data_points=extracted_data_points
        )

        return section_result
    

    def generate_agent_system_prompt(self, section: TemplateSection):
        """Generate the system prompt for the agent."""
        agent_info = section.section_agent
        system_prompt = f"""
            # Agent Role
            You are {agent_info.agent_name}. Your role is to extract information for the section '{section.section_name}' of the contract.

            # Agent Instructions
            {agent_info.agent_instructions}

            # Section Description

            {section.section_description}

            # Data Points to Extract
        """
        for dp in section.data_points:
            system_prompt += f"""
                ## Data Point Name: {dp['data_point_name']}
                - Description: {dp['data_point_description']}
                - Questions: {', '.join(dp['data_point_questions'])}
                - Instructions: {dp['data_point_instructions']}
            """
        # Instructions for the agent's output format
        system_prompt += """

            # Additional Important Instructions:

            - Carefully read the contract provided in the user prompt.
            - Sequentially process each data point listed above.
            - For each data point, provide the following information:
            - **data_point_name**: Copy verbatim from the data point name provided.
            - **data_point_description**: Copy verbatim from the data point description provided.
            - **data_point_questions**: Copy verbatim from the data point questions provided.
            - **data_point_overview**: Summarize all the information collected for the data point.
            - **data_point_answers**: Answer each of the data point questions in separate paragraphs.
            - **data_point_verbatim**: Provide the exact text from the contract that was used to collect the data point.

            - Provide the extracted data points in the following JSON format:

            {
                "data_points": [
                    {
                        "data_point_name": "...",
                        "data_point_description": "...",
                        "data_point_questions": ["...", "..."],
                        "data_point_overview": "...",
                        "data_point_answers": "...",
                        "data_point_verbatim": "..."
                    }
                    // Include additional data points as needed
                ]
            }

            # Output Format
            ```json
                            {
                "$defs": {
                    "DataPoint": {
                    "properties": {
                        "data_point_name": {
                        "description": "The name of the data point for which the agent collected the information. Should be copied verbatim from the data point name in the plan.",
                        "title": "Data Point Name",
                        "type": "string"
                        },
                        "data_point_description": {
                        "description": "A description of what is contained in the data point. Should be copied verbatim from the data point description in the plan.",
                        "title": "Data Point Description",
                        "type": "string"
                        },
                        "data_point_questions": {
                        "description": "A list of questions that the agent used to collect the data point. Should be copied verbatim from the data point questions in the plan.",
                        "items": {
                            "type": "string"
                        },
                        "title": "Data Point Questions",
                        "type": "array"
                        },
                        "data_point_overview": {
                        "description": "A summary of all the information that was collected for the datapoint by the agent.",
                        "title": "Data Point Overview",
                        "type": "string"
                        },
                        "data_point_answers": {
                        "description": "The answer to the data point questions. The agent should include all the answers for the data point questions in the data point answers. Write each answer in a new paragraph.",
                        "title": "Data Point Answers",
                        "type": "string"
                        },
                        "data_point_verbatim": {
                        "description": "The verbatim text from the contract that the agent used to collect the data point. Should be copied verbatim from the contract.",
                        "title": "Data Point Verbatim",
                        "type": "string"
                        }
                    },
                    "required": [
                        "data_point_name",
                        "data_point_description",
                        "data_point_questions",
                        "data_point_overview",
                        "data_point_answers",
                        "data_point_verbatim"
                    ],
                    "title": "DataPoint",
                    "type": "object"
                    }
                },
                "properties": {
                    "data_points": {
                    "description": "A list of data points that were extracted from the contract by the agent.",
                    "items": {
                        "$ref": "#/$defs/DataPoint"
                    },
                    "title": "Data Points",
                    "type": "array"
                    }
                },
                "required": [
                    "data_points"
                ],
                "title": "ExtractedData",
                "type": "object"
                }
                
            ```

            - Ensure that your response is valid JSON and follows the structure exactly.
            - Do not include any text outside the JSON object.
        """
        return system_prompt.strip()
    
    def generate_agent_user_prompt(self):
        """Generate the user prompt for the agent."""
        user_prompt = f"""
            Please analyze the following contract and extract the required data points.

            # Contract Document

            {self.contract_text}
        """
        return user_prompt.strip()
    
    def save_results(self, output_path):
        """Save the extracted results to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {output_path}")
