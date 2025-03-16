from enum import Enum

class ModelType(Enum):
    GPT4O = "gpt-4o"
    O1 = "o1-preview"
    GPT4_TURBO = "gpt-4-turbo"
    CLAUDE = "claude-3-5-sonnet-20241022"
    GEMINI = "gemini-1.5-pro"
    MISTRAL = "mistral--large-latest"
    DEEPSEEK = "deepseek-chat"
    LLAMA3B = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA70B = "meta-llama/Llama-3.3-70B-Instruct"
