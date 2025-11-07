import onnxruntime_genai as og
import argparse

def main(args):
    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "cpu":
        config.clear_providers()
        config.append_provider(args.execution_provider)
    
    model = og.Model(config)
    print("Model loaded")
    
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    print("Tokenizer created\n")
    
    while True:
        text = input("Prompt (Use quit() to exit): ")
        if not text:
            print("Error, input cannot be empty")
            continue
        
        if text == "quit()":
            break
        
        formatted_prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_tokens = tokenizer.encode(formatted_prompt)
        
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=2048, temperature=0.6, top_p=0.95, top_k=20)
        
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)
        
        print("\nOutput: ", end='', flush=True)
        
        try:
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple ONNX GenAI inference without chat template")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Onnx model folder path')
    parser.add_argument('-e', '--execution_provider', type=str, default='cpu', choices=["cpu", "cuda", "dml"], help="Execution provider")
    args = parser.parse_args()
    main(args)
