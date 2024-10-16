import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify_user_intent(prompt, model, tokenizer):

    # Prepare for inference
    model.eval()

    # Input text
    template = "Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: [QUERY]\n\nResponse:"

    input_text = template.replace("[QUERY]", prompt)
    # print(input_text)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Find the token ID for <nexa_end>
    nexa_end_token_id = tokenizer.convert_tokens_to_ids('<nexa_end>')

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=nexa_end_token_id
        )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the function call
    response_start = generated_text.find("Response:")
    nexa_end_pos = generated_text.find("<nexa_end>")
    
    if response_start != -1 and nexa_end_pos != -1:
        function_call = generated_text[response_start + 9:nexa_end_pos].strip()
    else:
        function_call = "No function call found"

    print(f"Function call: {function_call}")
    return function_call
