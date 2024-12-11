from typing import Callable, List, Optional, Union
from src.swarm.swarm.core import Swarm
from src.swarm.swarm.types import Agent
from openai import BaseModel, OpenAI
import os
from dotenv import load_dotenv
import tiktoken
import json
import sys
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize rich console
console = Console()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Thoughts
# - A checking mechanism can be implemented on top of the legal contract expert agent to make sure the response is a Yes or No answer and not a maybe or something else (in which case the question is asked again).
# - Maintaining conversation history might be useful to reduce the token usage and might lead to improved performance - at the moment no conversation history is maintained.

## Constants

MAX_EXPERT_RETRIES = 3  # Maximum number of retries for expert agent

OLLAMA_CLIENT = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
OPENAI_CLIENT = OpenAI(
    base_url="https://api.openai.com/v1/",
    api_key=OPENAI_API_KEY
)
# Ollama Model Options: 
# - llama2:13b
# - mistral
# - phi3

# OpenAI Model Options:
# - gpt-4o-mini
# - gpt-4o

## Agent Class

class LegalContractExpert:
    def __init__(self, 
                 provider=None, # 'ollama' or 'openai'
                 model_name=None,
                 ):
        if provider == 'ollama':
            self.client = Swarm(client=OLLAMA_CLIENT)
        elif provider == 'openai':
            self.client = Swarm(client=OPENAI_CLIENT)
        else: 
            raise ValueError("Invalid provider")
        
        if model_name is None:
            raise ValueError("Model name must be provided")
        
        self.model_name = model_name
        self.provider = provider
        self.verbose = 0
        if self.provider == 'openai':
            self.encoding = tiktoken.encoding_for_model(model_name)
    
    def _log(self, message: str):
        """Helper method to handle logging with rich output"""
        if self.verbose:
            console.print(f"[cyan]►[/cyan] {message}")
    
    def _print_agent_message(self, agent_name: str, message: str):
        """Helper method to print agent messages in chat-style format"""
        if self.verbose:
            console.print(Panel(
                Text(message, style="white"),
                title=agent_name,
                border_style="green"
            ))

    def _print_user_message(self, message: str):
        """Helper method to print user messages in chat-style format"""
        if self.verbose: 
            console.print(Panel(
                Text(message, style="white"),
                title="User",
                border_style="blue"
            ))

    def _get_question_answer_instructions(self, contract_text: str) -> str:
        return f"""
            **Role:**

            You are an **expert legal contract interpreter**. Your task is to answer questions about the provided contract text. For each question asked by the user, you must analyze the contract and provide a clear "Yes" or "No" answer, followed by a brief explanation of your reasoning.

            ---

            **Contract Text:**

            {contract_text}

            ---

            **Instructions:**

            When answering questions, please adhere strictly to the following guidelines:

            1. **Answer Format:**
            - **First Line:** Start your response with either **"Yes"** or **"No"** on its own line. This line should contain only the word "Yes" or "No" without any additional words or punctuation.
            - **Second Line:** Provide a brief and concise explanation of your reasoning, based solely on the content of the contract.
            - **Length:** The explanation should be **1-3 sentences**.

            3. **Determination:**
            - **Definite Answers Only:** You must make a clear determination; responses like "Maybe", "Possibly", "It depends", or "I don't know" are **not acceptable**.
            - **Handling Ambiguity:** If the contract is ambiguous or does not directly address the question, make your best-informed determination based on the available information and explain your reasoning.

            4. **Scope of Analysis:**
            - **Only Use Provided Text:** Base your answers **solely** on the provided contract text. Do not assume any information that is not present in the contract.
            - **No External Information:** Do not use external knowledge or assumptions beyond the contract text.

            5. **Language and Tone:**
            - **Professional Tone:** Maintain a professional and objective tone throughout your response.
            - **First Person Singular:** Do not use phrases like "We can see that" or "It seems that". Instead, state facts directly.

            ---

            **Response Format Examples:**

            **Example 1:**

            Yes  
            Section 3.1 explicitly states that "The party may engage in...", which permits this action.

            **Example 2:**

            No  
            According to Clause 4.2, "The party shall not...", which prohibits this activity.

            **Example 3 (Handling Ambiguity):**

            Yes  
            While the contract does not mention this directly, Section 2.3 implies permission by stating "The party is allowed to...".

            **Example 4:**

            No  
            The contract specifies in Section 5.4 that "Termination requires a 30-day notice," indicating that immediate termination is not allowed.

            ---

            **Important Notes:**

            - **Consistency:** Always follow the exact format shown in the examples.
            - **Answer Placement:** Ensure that your Yes/No answer is on the first line by itself.
            - **Avoid Uncertainty:** Do not express uncertainty or provide non-committal answers.
            - **Do Not Deviate:** Do not include any additional information beyond what is specified.

        """

    def _extract_yes_no_answer(self, response: str) -> Optional[str]:
        """Helper function to extract Yes/No answer using regex"""
        has_yes = bool(re.search(r'(?:^|\W)\*{0,2}(?i:Yes)\*{0,2}(?:$|\W)', response))
        has_no = bool(re.search(r'(?:^|\W)\*{0,2}(?i:No)\*{0,2}(?:$|\W)', response))
        
        if has_yes and not has_no:
            return "Yes"
        elif has_no and not has_yes:
            return "No"
        else:
            return None

    def _get_expert_response(self, question: str, expert_agent: Agent, debug: bool = False) -> str:
        """Helper function to get response from expert agent with retry logic"""
        self._log("Getting response from expert agent")

        retry_count = 0
        while retry_count < MAX_EXPERT_RETRIES:
            if retry_count > 0:
                self._log(f"Retrying expert response (attempt {retry_count + 1}/{MAX_EXPERT_RETRIES}) - Previous response did not have a clear Yes or No answer")
                question_message = f"""
                    ## Question
                    Based on your analysis of the contract text, answer the following question with a clear Yes or No answer:
                    Question: {question}
                    Your previous response did not have a clear Yes or No answer. Please start your response with either 'Yes' or 'No' followed by your explanation.
                    Your previous response was probably a bit too verbose. Keep your response short and concise and clearly state your Yes or No answer.
                """
            else:
                question_message = f"""
                    ## Question
                    Based on your analysis of the contract text, answer the following question with a Yes or No answer:
                    Question: {question}
                """

            self._print_user_message(question_message)
            expert_response = self.client.run(
                agent=expert_agent,
                messages=[{"role": "user", "content": question_message}],
                debug=debug
            ).messages[-1]['content']
            self._print_agent_message("Legal Contract Expert", expert_response)

            answer = self._extract_yes_no_answer(expert_response)
            if answer is not None:
                return expert_response

            retry_count += 1

        # If we get here, we've exhausted retries
        self._log("Failed to get clear Yes/No response after retries")
        return expert_response  # Return last response anyway

    def _load_contract_file(self, contract_path: str) -> str:
        """Helper function to load and validate contract file"""
        self._log(f"Loading contract file from: {contract_path}")
            
        # Normalize path and check if file exists
        contract_path = os.path.normpath(contract_path)
        if not os.path.exists(contract_path):
            # Try looking in project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            full_path = os.path.join(project_root, contract_path)
            if not os.path.exists(full_path):
                raise ValueError(f"Contract file not found at path: {contract_path} or {full_path}")
            contract_path = full_path

        try:
            with open(contract_path, 'r') as file:
                contract_text = file.read()
        except Exception as e:
            raise ValueError(f"Error reading contract file: {str(e)}")

        if not contract_text:
            raise ValueError("Contract file is empty.")
            
        self._log("Contract file loaded successfully")
            
        return contract_text

    def _process_category(self, category_question_pair: tuple, expert_agent: Agent, debug: bool = False) -> tuple:
        """Helper function to process individual category questions"""
        category, question = category_question_pair
        self._log(f"Processing category: {category}")
        
        expert_response = self._get_expert_response(question, expert_agent, debug)
        answer = self._extract_yes_no_answer(expert_response)
        if not answer:
            answer = "N/A"
        
        # Handle token count consistently
        if self.provider == 'openai':
            # Calculate input tokens including system prompt
            system_prompt = expert_agent.instructions
            input_tokens = len(self.encoding.encode(question)) + \
                         len(self.encoding.encode(system_prompt))
            token_count = {
                "input_tokens": input_tokens,
                "output_tokens": len(self.encoding.encode(expert_response))
            }
        else:
            # For non-OpenAI models, use -1 to indicate token counting not available
            token_count = {
                "input_tokens": -1,
                "output_tokens": -1
            }
        
        return category, {
            "answer": answer,
            "raw_response": expert_response,
            "token_count": token_count
        }

    def answer_questions_list(self, 
                            contract_path: str, 
                            category_to_question_path: str, 
                            debug: bool = False,
                            verbose: int = 0):
        """
        Answer questions about the contract based on categories defined in a JSON file
        This function initializes a single instance of LegalContractExpert agent for given contract and uses it to answer questions.
        """
        self.verbose = verbose
        self._log("Starting contract analysis for multiple questions")
            
        contract_text = self._load_contract_file(contract_path)
        
        # Load and validate category to question mapping
        self._log(f"Loading questions from: {category_to_question_path}")
            
        try:
            with open(category_to_question_path, 'r') as f:
                category_to_question = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading category to question JSON file: {str(e)}")
            
        if not category_to_question:
            raise ValueError("Category to question mapping is empty")
        
        # Generate Expert Agent once
        self._log("Initializing Legal Contract Expert agent")
            
        expert_instructions = self._get_question_answer_instructions(contract_text)
        expert_agent = Agent(
            name="Legal Contract Expert",
            model=self.model_name,
            instructions=expert_instructions
        )

        results = {}
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Process categories sequentially with rich output
        for category, question in category_to_question.items():
            try:
                self._print_user_message(f"Question: {question}")
                category, result = self._process_category(
                    (category, question), 
                    expert_agent, 
                    debug=debug
                )
                results[category] = result
                if self.provider == 'openai':
                    total_input_tokens += result['token_count']['input_tokens']
                    total_output_tokens += result['token_count']['output_tokens']
            except Exception as e:
                self._log(f"Error processing category: {str(e)}")

        self._log("Completed processing all categories")
        
        # Always include token_count in results
        if self.provider == 'openai':
            num_questions = len(category_to_question)
            results['token_count'] = {
                "average_input_tokens": total_input_tokens / num_questions,
                "average_output_tokens": total_output_tokens / num_questions
            }
        else:
            # For non-OpenAI models, use -1 to indicate token counting not available
            results['token_count'] = {
                "average_input_tokens": -1,
                "average_output_tokens": -1
            }
            
        return results

    def answer_contract_question(self, contract_path: str, question: str, debug: bool = False, verbose: int = 0):
        """
        Answer specific questions about the contract
        """
        self.verbose = verbose
        self._log("Starting contract analysis for single question")
        
        contract_text = self._load_contract_file(contract_path)
        
        # Generate Expert Agent
        self._log("Initializing Legal Contract Expert agent")
        
        expert_instructions = self._get_question_answer_instructions(contract_text)
        expert_agent = Agent(
            name="Legal Contract Expert",
            model=self.model_name,
            instructions=expert_instructions
        )

        # Get expert response with retry logic
        expert_response = self._get_expert_response(question, expert_agent, debug)

        # Extract Yes/No answer
        answer = self._extract_yes_no_answer(expert_response)
        if not answer:
            answer = "N/A"

        self._log("Completed contract analysis")
        
        # Handle token count consistently with answer_questions_list
        if self.provider == 'openai':
            # Calculate input tokens including system prompt
            system_prompt = expert_agent.instructions
            input_tokens = len(self.encoding.encode(question)) + \
                         len(self.encoding.encode(system_prompt))
            token_count = {
                "input_tokens": input_tokens,
                "output_tokens": len(self.encoding.encode(expert_response))
            }
        else:
            # For non-OpenAI models, use -1 to indicate token counting not available
            token_count = {
                "input_tokens": -1,
                "output_tokens": -1
            }
            
        return {
            "answer": answer,
            "raw_response": expert_response,
            "token_count": token_count
        }