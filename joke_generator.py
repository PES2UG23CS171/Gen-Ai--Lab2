import warnings
from transformers import pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize text generation pipeline
print("Loading AI model...")
generator = pipeline('text-generation', model='gpt2')

def generate_punchline(setup):
    """Generate a punchline using AI text generation."""
    
    prompt = f"{setup} "
    
    # Generate text
    result = generator(
        prompt,
        max_new_tokens=30,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    # Extract the generated text and remove the setup
    generated_text = result[0]['generated_text']
    punchline = generated_text[len(setup):].strip()
    
    return punchline


def main():
    print("=== Joke Punchline Generator ===\n")
    
    # Example jokes
    examples = [
        "Why did the chicken cross the road?",
        "Why do programmers prefer dark mode?",
        "How many developers does it take to change a light bulb?",
        "Knock knock!"
    ]
    
    print("Examples:\n")
    for setup in examples:
        punchline = generate_punchline(setup)
        print(f"Setup: {setup}")
        print(f"Punchline: {punchline}\n")
    
    # Interactive mode
    print("\n--- Try your own! ---\n")
    while True:
        try:
            setup = input("Enter joke setup: ").strip()
            if setup:
                punchline = generate_punchline(setup)
                print(f"Punchline: {punchline}\n")
        except KeyboardInterrupt:
            print("\n\nThanks for the laughs!")
            break

if __name__ == "__main__":
    main()