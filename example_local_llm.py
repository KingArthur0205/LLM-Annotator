#!/usr/bin/env python3
"""
Example script demonstrating how to use local LLM models (Llama-7b) with the LLM Annotator.

This script shows the simple annotate() function usage for local models.

Requirements:
- Install transformers, torch, and accelerate: pip install transformers torch accelerate
- Have sufficient GPU memory for the model (or use CPU mode)
- Hugging Face access token (if using gated models)
"""

import os
from llm_annotator.main import annotate

def main():
    print("Running annotation with local Llama-7b model...")
    
    # Check if we're in Google Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Detected Google Colab environment")
    except ImportError:
        IN_COLAB = False
        print("Detected local environment")
    
    # Set up paths based on environment
    if IN_COLAB:
        # Google Colab - use Google Drive/Sheets IDs
        transcript_source = "1YAm4aD_UPj64dCGJmeaTjYFhBhG-WjTFp3jnIMx909Q"  # Example Google Sheet ID
        sheet_source = "1miMC8M_UkfY_3XhglTAdt9sC9H8D6d9slzC7_LgcLOc"  # Example Google Sheet ID
        save_dir = "/content/drive/MyDrive/LLM_Annotation_Results/"
    else:
        # Local environment - use local file paths
        transcript_source = "./data/mol.csv"
        sheet_source = "./public/Codebook.xlsx"
        save_dir = "./results"
        
        # Check if local files exist
        if not os.path.exists(transcript_source):
            print(f"Warning: {transcript_source} not found. Please provide a valid local CSV file.")
            return
        if not os.path.exists(sheet_source):
            print(f"Warning: {sheet_source} not found. Please provide a valid local Excel file.")
            return
    
    # Simple annotation call with local model
    annotate(
        model_list=["llama-3b-local"],  # or ["llama-13b-local"] for 13B model
        obs_list=["17"],  # Use "all" to annotate everything
        feature="Mathcompetent",
        transcript_source=transcript_source,
        sheet_source=sheet_source,
        prompt_path="data/prompts/base.txt",
        system_prompt_path="data/prompts/system_prompt.txt",
        if_wait=True,
        n_uttr=10,
        if_test=True,
        save_dir=save_dir
    )
    
    print("Annotation complete!")
    print(f"Results saved to {save_dir}")

def example_multiple_models():
    """Example of using multiple models including local ones"""
    print("Running annotation with multiple models...")
    
    annotate(
        model_list=["llama-3b-local", "claude-3-7"],  # Mix of local and API models
        obs_list=["17"],
        feature="Mathcompetent",
        transcript_source="./data/mol.csv",
        sheet_source="./public/Codebook.xlsx",
        prompt_path="data/prompts/base.txt",
        system_prompt_path="data/prompts/system_prompt.txt",
        if_wait=True,
        n_uttr=10,
        if_test=True,
        save_dir="./results"
    )

if __name__ == "__main__":
    main()
    # Uncomment to run multiple models example
    # example_multiple_models()