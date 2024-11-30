# MetaMesh
Meta-Prompting using Dynamic MultiAgent Collaboration for Legal Contract Analysis. The focus on CUAD dataset to show the performance of the system, this can easily be extended to other datasets.


## Project Structure
```txt
metamesh/
├── src/                  # Core project source code
│   ├── agents/          # Agent implementations for legal contract analysis
│   │   ├── legal_contract_expert.py    # Main agent for contract Q&A
│   │   └── __init__.py  # Agent exports
│   ├── cuad_evaluator.py    # Evaluation metrics calculation
│   ├── cuad_executor.py     # CUAD dataset processing pipeline
│   ├── utils.py            # Helper functions for data loading/saving
│   └── __init__.py        # Package exports
├── results/              # Evaluation results and model predictions
│   ├── evals/           # Model evaluation metrics
│   └── predictions/     # Raw model predictions
├── .env                 # Environment variables and API keys
├── .gitignore          # Git ignore patterns for data, env files etc
└── README.md           # Project documentation
```

**NOTE:** I couldn't include the data and result files in the repo due to size contraints, but feel free to reach out to me to get them.

## Implementation and Experiment Log

### Baseline Evaluation System

**Objective:** Establish baseline performance metrics for local and openai models directly processing legal contracts.

**Implementation:**
1. Created core components:
   - LegalContractExpert agent with OpenAI/Ollama model support
   - CUADExecutor for dataset processing pipeline
   - CUADEvaluator for metrics calculation (accuracy, precision, recall, F1)
   - Preprocessing scripts for CUAD dataset formatting

2. Key Features:
   - Single/multi question contract analysis
   - File-level and category-level evaluation
   - Support for multiple model comparisons
   - Metrics tracking and result storage

**Current Status:**
- Baseline evaluation system operational + evaluation for a model can also be run through command line.
- Direct contract-to-model approach established as performance baseline
- Results stored in `results/evals/` and `results/predictions/`

**Key Insights:**
1. Direct contract processing shows room for improvement
2. Legal contracts need structured preprocessing
3. Multi-agent collaboration could enhance understanding
4. Systematic evaluation framework essential for comparing approaches

**Next Steps:**
1. Design multi-agent collaboration system:
   - Develop specialized agents for contract analysis
   - Create structured format for contract information
   - Implement inter-agent communication protocols

2. Implement iterative prompt improvement:
   - Use subset of CUAD for prompt engineering
   - Create feedback loop for prompt refinement
   - Track prompt evolution and performance gains

3. Evaluation targets:
   - Compare preprocessed vs baseline performance
   - Analyze impact of different preprocessing strategies
   - Measure efficiency of multi-agent collaboration

**Future Experiments:**
- Multi-agent preprocessing strategies
- Structured information extraction methods
- Prompt engineering optimization
- Cross-model performance analysis
