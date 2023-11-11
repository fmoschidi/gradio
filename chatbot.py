import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


def chat_with_model(prompt):
    tokenizer = AutoTokenizer.from_pretrained("allenai/llama-2")
    model = AutoModelForCausalLM.from_pretrained("allenai/llama-2")

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


iface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Ask the chatbot something..."),
    outputs="text",
)

iface.launch()
