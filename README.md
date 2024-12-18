# MetaMesh: Multi-Agent Meta-Prompting for Legal Contract Analysis

**MetaMesh** is a multi-agent framework designed to improve document comprehension—particularly for long, domain-specific contracts—by generating **structured intermediate representations**. Leveraging dynamic agent collaboration and meta-prompting, MetaMesh transforms raw contracts into JSON-like schemas, enabling smaller or specialized models to more reliably answer yes/no questions and potentially reduce inference overhead.

## Overview
Despite recent advances in Large Language Models (LLMs), directly parsing raw legal contracts often proves inefficient and can miss crucial nuances. **MetaMesh** addresses these gaps by:
- **Generating Structured Representations**: Dynamically creating a JSON-like breakdown of contract clauses and key data points.
- **Multi-Agent Collaboration**: Coordinating specialized agents (Planner Agent, Plan Executor, and dedicated “section” agents) to extract and annotate contract content.
- **Offline Preprocessing**: Allowing the structured representation to be reused for multiple queries, saving computational costs and token usage over time.

The system has been tested on a subset of the **Contract Understanding Atticus Dataset (CUAD)** to compare raw contract Q&A (baseline) vs. using structured representations (MetaMesh).

## Key Insights from Evaluation
- **Accuracy & F1**: Directly processing raw contracts (baseline) currently outperforms MetaMesh in overall accuracy and F1 scores on the sampled CUAD subset, indicating room for improvement in capturing contract nuances.
- **Token Usage**: Despite the additional overhead of generating an intermediate representation, MetaMesh **significantly reduces overall token usage** when handling repeated queries or multiple questions.
- **Processing Time**: While the multi-agent approach initially increases latency (due to plan generation), **offline reuse** of the structured representation can speed up subsequent queries.
- **Future Directions**: Introducing self/collaborative evaluation mechanisms and refining the multi-agent orchestration loop may improve accuracy and F1. Exploring local LLM integration and alternative representation formats (e.g., knowledge graphs) are promising next steps.

---

## Project Structure

```txt
metamesh/
├── src/
│   ├── agents/
│   │   ├── prompts/                      # Prompts for the agents
│   │   ├── legal_contract_expert.py       # Main agent for contract Q&A
│   │   ├── plan_executor.py              # Agent that executes on the plan to create the structured representation
│   │   ├── planner.py                    # Agent that creates the representation template and agent instructions
│   ├── cuad_evaluator.py                  # Evaluation metrics calculation
│   ├── cuad_executor.py                   # CUAD dataset processing pipeline
│   ├── metamesh.py                        # Main class for the MetaMesh system
│   ├── utils.py                           # Helper functions for data loading/saving
│   └── plot.py                            # Plotting functions for analysis
├── results/                               # Evaluation results and model predictions --> uploaded on google drive linked below
├── data/                                  # CUAD raw and processed data --> uploaded on google drive linked below
├── project_presentation.pdf               # Project presentation
├── project_report.pdf                     # Project report
├── .env                                   # Environment variables and API keys
├── .gitignore                             # Git ignore patterns for data, env files etc.
└── README.md                              # Project documentation
└── Notebooks                              # All notebooks are expected to be run from the root directory
```

## Tutorial Notebooks --> The tutorialnotebooks are in the 'tutorial_notebooks' folder in the google drive linked below. Couldn't upload them here due to file size constraints.
- `baseline_analysis.ipynb`: The notebook that you will need to run to recreate all the evaluation metrics and plots.
- `metamesh_main.ipynb`: The notebook that contains examples on how to use the MetaMesh system.
- `agent_testing_and_tuning.ipynb`: The notebook used to run the agents and tune the prompts.
- `initial_analysis.ipynb`: The notebook used for initial analysis of the dataset. Some of it is also in the baseline analysis notebook.


### Google Drive Link
[MetaMesh Google Drive](https://drive.google.com/drive/folders/1i1a6puhNSNd3OPe4D333U5ytOu4yVVXn?usp=sharing)
