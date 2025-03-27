# LLM Annotator

## Overview
LLM Annotator is a powerful framework for **automated data annotation** using state-of-the-art **Large Language Models (LLMs)**. This tool allows users to leverage **Claude 3-7 Sonnet, ChatGPT-4o, and DeepSeek** for labeling and processing textual data efficiently. 

## Features
- Supports **multiple LLMs** for flexible annotation: Claude 3-7 Sonnet and ChatGPT-4o.
- **Modular pipeline architecture**, allowing customizable annotation workflows.
- **Batch Processing**.

## Setup
### 1. Clone the Repository
```bash
git clone https://github.com/KingArthur0205/LLM-Annotator.git
cd LLM-Annotator
```

### 2. Install Dependencies
Install the package in editable mode:
```bash
pip install -e .
```

### 3. Set Up API Keys
LLM Annotator requires API keys for the LLMs. Create a `.env` file in the root directory by running ```cd LLM-Annotator```, ```touch .env```, and ```open .env``` with the following format:
```ini
OPENAI_API_KEY=your-chatgpt-4o-key
ANTHROPIC_API_KEY=your-claude-3-7-sonnet-key
```
Replace `your-chatgpt-4o-key` and `your-claude-3-7-sonnet-key`with the respective API keys.

### 4. Handling Data Sources
- **Google Colab**: Provide the **Google Sheet ID** for data loading.
- **Local Execution**: Place the required files inside the `data` folder.

## Usage
### 1. Running the Annotation Pipeline
Once set up, you can use the annotator in Python:
```python
from llm_annotator.main import annotate

annotate(
    model_list=["gpt-4o", "claude-3-7"],
    obs_list=["146", "170"],
    feature="Mathcompetent",
    transcript_path="data/alltranscripts_423_clean_segmented.csv",
    sheet_source="data/MOL Roles Features.xlsx",
    n_uttr=30, # No. of utterances to include in one LLM request
    if_wait=True # This keeps the program running until the annotations are generated.
)
```

### 2. Fetch
To fetch the result and generate annotations, we can use the ```fetch()``` function:
```python
from llm_annotator.main import fetch

fetch(feature="Mathcompetent", batch_dir="path_to_batch")
```
Note: The ```batch_dir``` parameter can be ignored. In this case, the results of the last batch request will be fetched.

### 3. Structure
The structure of the setup is presented below.
```
llm_annotator
|--data
|   |--prompts
|   |   |--prompt1.txt, prompt2.txt, ...
|   |--Role Features.xlsx
|   |--transcript_data.csv
|--result
|   |--atn_df.csv # Annotated Result
|   |--feature1
|   |   |--batch_meta
|   |   |   |--date_time1
|   |   |   |   |--claude-3-7.json # Meta-data of claude batch request
|   |   |   |   |--gpt-4o.json # Meta-data of GPT-4o batch request
|   |   |   |   |--metadata.json # Meta-data of runtime
|   |--feature2
...
```
