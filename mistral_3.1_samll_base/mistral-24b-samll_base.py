import os
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import TokensPrompt
import requests
from PIL import Image
from io import BytesIO
from vllm.multimodal import MultiModalDataBuiltins

from mistral_common.protocol.instruct.messages import TextChunk, ImageURLChunk

# Path to locally downloaded model
model_path = "./models/Mistral-Small-3.1-24B-Base-2503"

# Check if model exists locally, otherwise use the HF model name
if os.path.exists(model_path):
    print(f"Using locally downloaded model at {model_path}")
    model_name = model_path
else:
    print("Model not found locally, using Hugging Face model")
    model_name = "mistralai/Mistral-Small-3.1-24B-Base-2503"

sampling_params = SamplingParams(max_tokens=8192)

print(f"Loading model: {model_name}")
llm = LLM(model=model_name, tokenizer_mode="mistral")
print("Model loaded successfully")

# Download image
url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
print(f"Downloading image from {url}")
response = requests.get(url)
image = Image.open(BytesIO(response.content))
print("Image downloaded successfully")

prompt = "The image shows a"
print(f"Prompt: '{prompt}'")

user_content = [ImageURLChunk(image_url=url), TextChunk(text=prompt)]

tokenizer = llm.llm_engine.tokenizer.tokenizer.mistral.instruct_tokenizer
tokens, _ = tokenizer.encode_user_content(user_content, False)

prompt = TokensPrompt(
    prompt_token_ids=tokens, multi_modal_data=MultiModalDataBuiltins(image=[image])
)

print("Generating response...")
outputs = llm.generate(prompt, sampling_params=sampling_params)

print("\nModel output:")
print(outputs[0].outputs[0].text)