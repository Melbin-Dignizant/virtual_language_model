import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

# Initialize model
model_path = r"D:\Melbin\VLM-Examples\llava_model\llava-v1.6-mistral-7b-hf"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def respond(message, image):
    if image is not None:
        inputs = processor(message, images=image, return_tensors="pt").to("cuda")
    else:
        inputs = processor(message, return_tensors="pt").to("cuda")
    
    generate_ids = model.generate(**inputs, max_length=500)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LLaVA v1.6 Mistral 7B Chatbot")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image (Optional)")
            text_input = gr.Textbox(label="Your Message")
            submit_btn = gr.Button("Send")
        with gr.Column():
            output = gr.Textbox(label="LLaVA's Response")
    
    submit_btn.click(
        fn=respond,
        inputs=[text_input, image_input],
        outputs=output
    )
    
    text_input.submit(
        fn=respond,
        inputs=[text_input, image_input],
        outputs=output
    )

demo.launch()