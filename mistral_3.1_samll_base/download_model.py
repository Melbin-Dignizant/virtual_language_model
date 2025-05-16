from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model():
    model_name = "mistralai/Mistral-Small-3.1-24B-Base-2503"
    save_path = "./models/Mistral-Small-3.1-24B-Base-2503"
    
    print(f"Downloading model: {model_name}")
    print(f"This may take a while depending on your internet speed...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    print("Tokenizer downloaded and saved successfully")
    
    # Download model
    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto",
        device_map="auto"
    )
    model.save_pretrained(save_path)
    print("Model downloaded and saved successfully")
    
    print(f"Model and tokenizer saved to {save_path}")

if __name__ == "__main__":
    download_model()