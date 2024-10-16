import onnxruntime_genai as og

def generation_chart_data(text, onnx_model, chat_template):

    # Create the tokenizer
    tokenizer = og.Tokenizer(onnx_model)

    # Set default search options
    search_options = {
        'do_sample': False,
        'max_length': 1024
    }

    # Create the prompt
    prompt = chat_template.format(input=text)
    print(prompt)

    # Encode the input tokens
    input_tokens = tokenizer.encode(prompt)

    # Set up generator parameters
    params = og.GeneratorParams(onnx_model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(onnx_model, params)

    result = ""

    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            result += tokenizer.decode([new_token])

            # Check if the result ends with '<end>'
            if result.endswith('<end>'):
                # Remove '<end>' from the result
                result = result[:-len('<end>')]
                break

    except KeyboardInterrupt:
        print("\n  --control+c pressed, aborting generation--")

    del generator

    print(result)
    return result
