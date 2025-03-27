# LLM Annotator

## Overview
LLM Annotator is a powerful framework for **automated data annotation** using state-of-the-art **Large Language Models (LLMs)**. This tool allows users to leverage **Claude 3-7 Sonnet, ChatGPT-4o, and DeepSeek** for labeling and processing textual data efficiently. 

## Features
- Supports **multiple LLMs** for flexible annotation: Claude 3-7 Sonnet, ChatGPT-4o, and DeepSeek.
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
LLM Annotator requires API keys for the LLMs. Create a `.env` file in the root directory with the following format:
```ini
OPENAI_API_KEY=your-chatgpt-4o-key
ANTHROPIC_API_KEY=your-claude-3-7-sonnet-key
DEEPSEEK_API_KEY=your-deepseek-key
```
Replace `your-chatgpt-4o-key`, `your-claude-3-7-sonnet-key`, and `your-deepseek-key` with the respective API keys.

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
    feature_list=["Mathcompetent"],
    transcript_path="data/alltranscripts_423_clean_segmented.csv",
    sheet_source="data/MOL Roles Features.xlsx",
    if_wait=True
)
```

### 2. Configuring to Wait
The ```if_wait``` parameter, if set to ```True```, loops and 

## Contributing
Feel free to submit issues, feature requests, and pull requests to enhance **LLM Annotator**.

## License
This project is licensed under the **MIT License**.