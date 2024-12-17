from typing import Callable, List, Optional, Union, Tuple, Dict
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
import time

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

    def _get_question_answer_instructions_for_intermediate_rep(self, intermediate_rep: dict) -> str:
        """Helper function to create system prompt for answering questions using intermediate representation"""
        return f"""
            **Role:**

            You are an **expert legal contract interpreter**. Your task is to answer questions about a contract using its structured intermediate representation. For each question asked by the user, you must analyze the provided representation and provide a clear "Yes" or "No" answer, followed by a brief explanation of your reasoning.

            ---

            **Contract Representation:**

            The provided structured representation contains detailed information extracted from the original contract, organized into sections covering various aspects like:
            - Parties and Agreement Overview
            - Intellectual Property Rights
            - Non-Compete Clauses
            - Financial Provisions
            - Liability and Indemnification
            - And other relevant sections

            Each section contains data points with specific information, verbatim quotes, and analysis.

            ---

            **Instructions:**

            When answering questions, please adhere strictly to the following guidelines:

            1. **Answer Format:**
            - **First Line:** Start your response with either **"Yes"** or **"No"** on its own line
            - **Second Line:** Provide a brief and concise explanation of your reasoning, based on the information in the representation
            - **Length:** The explanation should be **1-3 sentences**

            2. **Using the Representation:**
            - Base your answers solely on the information provided in the structured representation
            - Pay special attention to verbatim quotes and explicit data points
            - Consider both the overview sections and specific data points

            3. **Determination:**
            - Make clear determinations; avoid responses like "Maybe" or "It depends"
            - If the representation is ambiguous, make your best-informed determination based on available information

            4. **Language and Tone:**
            - Maintain a professional and objective tone
            - State facts directly, citing specific sections or data points from the representation

            ---

            **Example Response Format:**

            Yes
            According to the Financial Provisions section, the agreement explicitly includes revenue sharing with a royalty rate of [X]% on gross margins.

            ---

            **Important Notes:**

            - **Consistency:** Always follow the exact format shown in the examples.
            - **Answer Placement:** Ensure that your Yes/No answer is on the first line by itself.
            - **Avoid Uncertainty:** Do not express uncertainty or provide non-committal answers.
            - **Do Not Deviate:** Do not include any additional information beyond what is specified.

        """

    def _get_question_answer_instructions_with_both(self, intermediate_rep: dict, contract_text: str) -> str:
        """Helper function to create system prompt for answering questions using both intermediate representation and contract"""
        return f"""
            **Role:**

            You are an **expert legal contract interpreter**. Your task is to answer questions about a contract using both its structured intermediate representation and the original contract text. For each question, you must provide a clear "Yes" or "No" answer, followed by a brief explanation of your reasoning.

            ---

            **Available Resources:**

            1. **Structured Representation:**
            The intermediate representation contains pre-extracted information organized into sections like:
            - Parties and Agreement Overview
            - Intellectual Property Rights
            - Non-Compete Clauses
            - Financial Provisions
            - Liability and Indemnification
            - And other relevant sections
            Each section contains data points with specific information, verbatim quotes, and analysis.

            2. **Original Contract Text:**
            The full text of the original contract is also available for reference.

            ---

            **Analysis Process:**

            1. **Primary Reference:**
            - Always start by consulting the structured representation
            - Look for relevant sections and data points that address the question
            - Pay special attention to verbatim quotes and explicit analysis

            2. **Secondary Verification:**
            - If information in the representation is unclear or incomplete, refer to the original contract
            - Use the contract text to verify and supplement your understanding
            - Ensure consistency between both sources

            ---

            **Instructions:**

            When answering questions, please adhere strictly to the following guidelines:

            1. **Answer Format:**
            - **First Line:** Start your response with either **"Yes"** or **"No"** on its own line
            - **Second Line:** Provide a brief and concise explanation of your reasoning
            - **Length:** The explanation should be **1-3 sentences**

            2. **Source Usage:**
            - Primarily cite information from the structured representation
            - Reference the contract text only when needed for clarification or completeness
            - Ensure your answer is supported by both sources when possible

            3. **Determination:**
            - Make clear determinations; avoid responses like "Maybe" or "It depends"
            - If both sources are ambiguous, make your best-informed determination based on available information

            4. **Language and Tone:**
            - Maintain a professional and objective tone
            - State facts directly, citing specific sections or data points

            ---

            **Example Response Format:**

            Yes
            According to the Financial Provisions section of the representation and verified in Section 7.3 of the contract, the agreement includes revenue sharing with a royalty rate of [X]% on gross margins.

            ---

            **Important Notes:**

            - **Consistency:** Always follow the exact format shown in the examples
            - **Answer Placement:** Ensure that your Yes/No answer is on the first line by itself
            - **Avoid Uncertainty:** Do not express uncertainty or provide non-committal answers
            - **Do Not Deviate:** Do not include any additional information beyond what is specified

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

    def _get_expert_response(self, question: str, expert_agent: Agent, debug: bool = False) -> dict:
        """Helper function to get response from expert agent with retry logic"""
        self._log("Getting response from expert agent")
        
        # Initialize metrics
        metrics = {
            "token_count": {"input_tokens": 0, "output_tokens": 0},
            "processing_time": 0
        }

        retry_count = 0
        while retry_count < MAX_EXPERT_RETRIES:
            if retry_count > 0:
                self._log(f"Retrying expert response (attempt {retry_count + 1}/{MAX_EXPERT_RETRIES}) - Previous response did not have a clear Yes or No answer")
                question_message = f"""
                    ## Question
                    Based on your analysis, answer the following question with a clear Yes or No answer:
                    Question: {question}
                    Your previous response did not have a clear Yes or No answer. Please start your response with either 'Yes' or 'No' followed by your explanation.
                    Your previous response was probably a bit too verbose. Keep your response short and concise and clearly state your Yes or No answer.
                """
            else:
                question_message = f"""
                    ## Question
                    Based on your analysis, answer the following question with a Yes or No answer:
                    Question: {question}
                """

            self._print_user_message(question_message)
            
            # Calculate input tokens
            if self.provider == 'openai':
                system_prompt = expert_agent.instructions
                input_tokens = len(self.encoding.encode(question_message)) + \
                             len(self.encoding.encode(system_prompt))
                metrics["token_count"]["input_tokens"] = input_tokens
            
            # Time the LLM call
            start_time = time.time()
            response = self.client.run(
                agent=expert_agent,
                messages=[{"role": "user", "content": question_message}],
                debug=debug
            ).messages[-1]['content']
            end_time = time.time()
            
            # Update metrics
            metrics["processing_time"] = end_time - start_time
            if self.provider == 'openai':
                metrics["token_count"]["output_tokens"] = len(self.encoding.encode(response))
            
            self._print_agent_message("Legal Contract Expert", response)

            answer = self._extract_yes_no_answer(response)
            if answer is not None:
                return {
                    "answer": answer,
                    "response": response,
                    "metrics": metrics
                }

            retry_count += 1

        # If we get here, we've exhausted retries
        self._log("Failed to get clear Yes/No response after retries")
        return {
            "answer": "N/A",
            "response": response,
            "metrics": metrics
        }

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

    def _load_intermediate_rep(self, intermediate_rep_path: str) -> dict:
        """Helper function to load and validate intermediate representation file"""
        self._log(f"Loading intermediate representation from: {intermediate_rep_path}")
            
        try:
            with open(intermediate_rep_path, 'r') as f:
                intermediate_rep = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading intermediate representation file: {str(e)}")
            
        if not intermediate_rep:
            raise ValueError("Intermediate representation is empty")
        
        self._log("Intermediate representation loaded successfully")
        return intermediate_rep

    def _process_category(self, category_question_tuple: Tuple[str, str], expert_agent: Agent, debug: bool = False) -> Tuple[str, Dict]:
        """Process a single category question"""
        category, question = category_question_tuple
        
        # Get expert response with metrics
        response_data = self._get_expert_response(question, expert_agent, debug)
        
        # Format result consistently regardless of mode
        result = {
            'answer': response_data.get('answer', 'N/A'),
            'raw_response': response_data.get('response', ''),
            'metrics': {
                'token_count': {
                    'input_tokens': response_data.get('metrics', {}).get('token_count', {}).get('input_tokens', 0),
                    'output_tokens': response_data.get('metrics', {}).get('token_count', {}).get('output_tokens', 0)
                },
                'processing_time': response_data.get('metrics', {}).get('processing_time', 0)
            }
        }
        
        return category, result

    def answer_questions_list(self, 
                            contract_path: str,
                            category_to_question_path: str,
                            debug: bool = False,
                            verbose: int = 0):
        """Answer questions about the contract"""
        self.verbose = verbose
        self._log("Starting contract analysis")
        
        # Load contract
        contract_text = self._load_contract_file(contract_path)
        
        # Load and validate category to question mapping
        try:
            with open(category_to_question_path, 'r') as f:
                category_to_question = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading category to question JSON file: {str(e)}")
        
        if not category_to_question:
            raise ValueError("Category to question mapping is empty")
        
        # Generate Expert Agent
        expert_instructions = self._get_question_answer_instructions(contract_text)
        expert_agent = Agent(
            name="Legal Contract Expert",
            model=self.model_name,
            instructions=expert_instructions
        )

        results = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_processing_time = 0
        
        # Process categories sequentially
        for category, question in category_to_question.items():
            try:
                self._print_user_message(f"Question: {question}")
                response_data = self._get_expert_response(question, expert_agent, debug)
                
                # Store results with consistent metrics format
                results[category] = {
                    'answer': response_data['answer'],
                    'raw_response': response_data['response'],
                    'metrics': {
                        'token_count': {
                            'input_tokens': response_data['metrics']['token_count']['input_tokens'],
                            'output_tokens': response_data['metrics']['token_count']['output_tokens']
                        },
                        'processing_time': response_data['metrics']['processing_time']
                    }
                }
                
                # Aggregate metrics
                total_input_tokens += response_data['metrics']['token_count']['input_tokens']
                total_output_tokens += response_data['metrics']['token_count']['output_tokens']
                total_processing_time += response_data['metrics']['processing_time']
                
            except Exception as e:
                self._log(f"Error processing category: {str(e)}")
                continue
        
        # Add overall metrics in consistent format
        results['metrics'] = {
            'token_count': {
                'average_input_tokens': total_input_tokens / len(category_to_question) if category_to_question else 0,
                'average_output_tokens': total_output_tokens / len(category_to_question) if category_to_question else 0
            },
            'average_processing_time': total_processing_time / len(category_to_question) if category_to_question else 0,
            'total_questions': len(category_to_question)
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

    def answer_questions_list_with_intermediate_rep(self, 
                                                  intermediate_rep_path: str,
                                                  category_to_question_path: str,
                                                  contract_path: str = None,
                                                  debug: bool = False,
                                                  verbose: int = 0):
        """
        Answer questions about the contract using its intermediate representation and optionally the original contract
        
        Args:
            intermediate_rep_path (str): Path to the intermediate representation JSON file
            category_to_question_path (str): Path to JSON file mapping categories to questions
            contract_path (str, optional): Path to original contract file. If provided, both sources will be used
            debug (bool): Enable debug mode
            verbose (int): Verbosity level
            
        Returns:
            dict: Results containing answers and metadata for each category
        """
        self.verbose = verbose
        self._log("Starting contract analysis using intermediate representation")
            
        intermediate_rep = self._load_intermediate_rep(intermediate_rep_path)
        
        # Load contract if path provided
        contract_text = None
        if contract_path:
            self._log("Loading original contract for supplementary reference")
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
        
        # Generate Expert Agent with appropriate instructions
        self._log("Initializing Legal Contract Expert agent")
            
        if contract_text:
            self._log("Using both intermediate representation and contract text")
            expert_instructions = self._get_question_answer_instructions_with_both(
                intermediate_rep, 
                contract_text
            )
        else:
            self._log("Using only intermediate representation")
            expert_instructions = self._get_question_answer_instructions_for_intermediate_rep(
                intermediate_rep
            )
            
        expert_agent = Agent(
            name="Legal Contract Expert",
            model=self.model_name,
            instructions=expert_instructions
        )

        results = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_processing_time = 0
        
        # Process categories sequentially
        for category, question in category_to_question.items():
            try:
                self._print_user_message(f"Question: {question}")
                category, result = self._process_category(
                    (category, question), 
                    expert_agent, 
                    debug=debug
                )
                
                # Ensure result has all required fields
                result = {
                    'answer': result.get('answer', 'N/A'),
                    'raw_response': result.get('raw_response', ''),
                    'metrics': {
                        'token_count': {
                            'input_tokens': result.get('metrics', {}).get('token_count', {}).get('input_tokens', 0),
                            'output_tokens': result.get('metrics', {}).get('token_count', {}).get('output_tokens', 0)
                        },
                        'processing_time': result.get('metrics', {}).get('processing_time', 0)
                    }
                }
                
                results[category] = result
                
                # Aggregate metrics
                total_input_tokens += result['metrics']['token_count']['input_tokens']
                total_output_tokens += result['metrics']['token_count']['output_tokens']
                total_processing_time += result['metrics']['processing_time']
                
            except Exception as e:
                self._log(f"Error processing category: {str(e)}")
                continue

        # Add overall metrics
        results['metrics'] = {
            'token_count': {
                'average_input_tokens': total_input_tokens / len(category_to_question) if category_to_question else 0,
                'average_output_tokens': total_output_tokens / len(category_to_question) if category_to_question else 0
            },
            'average_processing_time': total_processing_time / len(category_to_question) if category_to_question else 0,
            'total_questions': len(category_to_question)
        }
            
        return results

    def answer_contract_question_with_intermediate_rep(self, 
                                                     intermediate_rep_path: str, 
                                                     question: str,
                                                     contract_path: str = None,
                                                     debug: bool = False, 
                                                     verbose: int = 0):
        """
        Answer specific question about the contract using its intermediate representation and optionally the original contract
        
        Args:
            intermediate_rep_path (str): Path to the intermediate representation JSON file
            question (str): Question to answer
            contract_path (str, optional): Path to original contract file. If provided, both sources will be used
            debug (bool): Enable debug mode
            verbose (int): Verbosity level
            
        Returns:
            dict: Result containing answer and metadata
        """
        self.verbose = verbose
        self._log("Starting contract analysis using intermediate representation")
        
        intermediate_rep = self._load_intermediate_rep(intermediate_rep_path)
        
        # Load contract if path provided
        contract_text = None
        if contract_path:
            self._log("Loading original contract for supplementary reference")
            contract_text = self._load_contract_file(contract_path)
        
        # Generate Expert Agent with appropriate instructions
        self._log("Initializing Legal Contract Expert agent")
        
        if contract_text:
            self._log("Using both intermediate representation and contract text")
            expert_instructions = self._get_question_answer_instructions_with_both(
                intermediate_rep, 
                contract_text
            )
        else:
            self._log("Using only intermediate representation")
            expert_instructions = self._get_question_answer_instructions_for_intermediate_rep(
                intermediate_rep
            )
            
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
        
        # Handle token count
        if self.provider == 'openai':
            system_prompt = expert_agent.instructions
            input_tokens = len(self.encoding.encode(question)) + \
                         len(self.encoding.encode(system_prompt))
            token_count = {
                "input_tokens": input_tokens,
                "output_tokens": len(self.encoding.encode(expert_response))
            }
        else:
            token_count = {
                "input_tokens": -1,
                "output_tokens": -1
            }
            
        return {
            "answer": answer,
            "raw_response": expert_response,
            "token_count": token_count
        }