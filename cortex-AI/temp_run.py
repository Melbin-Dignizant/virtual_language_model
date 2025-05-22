from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

# Update this to your actual local path
model_path = r"D:\Melbin\VLM-Examples\llava_model\llava-v1.6-mistral-7b-hf"

# Initialize processor and model
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def format_prompt(user_input, chat_history=None):
    """Format the prompt according to LLaVA's expected conversation style"""
    system_prompt = (
        "You are a helpful assistant. "
        "You can answer questions only about architecture based. " 
    )
    
    if chat_history:
        # Join previous conversation turns
        conversation = "\n".join([f"USER: {q}\nASSISTANT: {a}" for q, a in chat_history])
        prompt = f"{system_prompt}\n{conversation}\nUSER: {user_input}\nASSISTANT:"
    else:
        prompt = f"{system_prompt}\nUSER: {user_input}\nASSISTANT:"
    
    return prompt

def chat_with_llava(text_prompt, image_path=None, chat_history=None):
    """
    Function to interact with LLaVA model
    
    :param text_prompt: The text input for the model
    :param image_path: Optional path to an image for multimodal input
    :param chat_history: List of tuples (question, answer) for context
    :return: Model's response
    """
    # Format the prompt
    formatted_prompt = format_prompt(text_prompt, chat_history)
    
    # Prepare inputs
    if image_path:
        image = Image.open(image_path)
        inputs = processor(formatted_prompt, images=image, return_tensors="pt").to("cuda")
    else:
        inputs = processor(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_length=1000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    
    # Decode and clean the response
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    # Remove the prompt from the response
    response = response[len(formatted_prompt):].strip()
    return response

# Example usage with conversation history
if __name__ == "__main__":
    print("LLaVA Chatbot initialized. Type 'quit' to exit.")
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # For image input, you might prompt the user for an image path
        image_path = None  # Set this if you want image input
        
        response = chat_with_llava(user_input, image_path, chat_history)
        print("LLaVA:", response)
        
        # Update chat history (keep last 3 exchanges)
        chat_history.append((user_input, response))
        chat_history = chat_history[-3:]