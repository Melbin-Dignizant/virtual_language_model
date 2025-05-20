import os
import requests
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

def download_mistral():
    """
    Download the Mistral AI 3.1 Small base model from Hugging Face
    and save it to a local folder
    """
    # Create directory if it doesn't exist
    model_dir = Path("./Mistral-Small-3.1-24B-Instruct-2503")
    model_dir.mkdir(exist_ok=True)
    
    print(f"Downloading Mistral-Small-3.1-24B-Instruct-2503 to {model_dir.absolute()}...")
    
    try:
        # Download the complete model snapshot from Hugging Face
        snapshot_download(
            repo_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503", 
            local_dir=model_dir,
            local_dir_use_symlinks=False  # Actual file download, not symlinks
        )
        
        print(f"Successfully downloaded Mistral AI 3.1 Small to {model_dir.absolute()}")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        
        # Alternative: Try downloading specific files if complete snapshot fails
        print("Trying alternative download method...")
        try:
            # Download the model file
            hf_hub_download(
                repo_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                filename="model.safetensors", 
                local_dir=model_dir
            )
            
            # Download the tokenizer files
            for filename in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
                hf_hub_download(
                    repo_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    filename=filename,
                    local_dir=model_dir
                )
            
            print(f"Successfully downloaded key Mistral files to {model_dir.absolute()}")
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            print("Please ensure you have the 'huggingface_hub' package installed and valid credentials set up.")
            print("You can install it using: pip install huggingface_hub")
            print("You may need to run: huggingface-cli login")

if __name__ == "__main__":
    download_mistral()