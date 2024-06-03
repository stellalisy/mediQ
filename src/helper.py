import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from keys import mykey

# A dictionary to cache models and tokenizers to avoid reloading
models = {}

model_pricing = {
    "gpt-4-1106-preview": [0.01, 0.03],
    "gpt-4-0125-preview": [0.01, 0.03],
    "gpt-4-turbo-preview": [0.01, 0.03],
    "gpt-4-turbo-2024-04-09": [0.01, 0.03],
    "gpt-4-turbo": [0.01, 0.03],
    "gpt-3.5-turbo": [0.0005, 0.0015],
    "gpt-3.5-turbo-0125": [0.0005, 0.0015]
}


def load_model_and_tokenizer(model_name):
    # This function attempts to load a model from a Hugging Face identifier
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()  # Set the model to evaluation mode
        return model, tokenizer
    except Exception as e:
        print(f"Could not load model {model_name} locally: {str(e)}")
        return None, None

def get_response(messages, model_name, temperature=1, max_tokens=512, top_p=1, top_logprobs=0, **kwargs):
    if 'gpt' in model_name:
        # OpenAI's GPT models
        return get_response_openai(messages, model_name, temperature, max_tokens, top_p, top_logprobs, **kwargs)
    else:
        # Local or other Hugging Face supported models
        return get_response_local(messages, model_name, temperature, max_tokens, top_p, top_logprobs, **kwargs)

def get_response_openai(messages, model_name, temperature, max_tokens, top_p, top_logprobs, api_account):
    openai.api_key = mykey[api_account]  # Setup API key appropriately
    response = openai.ChatCompletion.create(
                    engine=model_name.replace('.', ''),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                ) if top_logprobs == 0 else openai.ChatCompletion.create(
                    engine=model_name.replace('.', ''),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=True, 
                    top_logprobs=top_logprobs
                )
    usage += model_pricing[model_name][0]*response["usage"]["prompt_tokens"] + model_pricing[model_name][1]*response["usage"]["completion_tokens"]
    response_text = response.choices[0].text.strip()
    log_probs = response.choices[0].logprobs.top_logprobs if top_logprobs > 0 else None
    return response_text, log_probs

def get_response_local(messages, model_name, temperature=0.6, max_tokens=256, top_p=0.9, top_logprobs=0):
    model, tokenizer = models.get(model_name, (None, None))
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_name)
        models[model_name] = (model, tokenizer)
    
    if model and tokenizer:
        # Check if the model expects a chat-like format
        try:  # This condition can be refined
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        except:
            # Join messages into a single prompt for general language models
            prompt = "\n\n".join([m['content'] for m in messages])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response_text, None
    else:
        return "Model not found or failed to load.", None
