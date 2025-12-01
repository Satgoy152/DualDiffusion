
import random
import re
from datasets import load_dataset


# Utility to extract numeric answer
def extract_answer(text):
    match = re.search(r"Answer:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None

def load_gsm8k():
    """
    Loads the GSM8K dataset.
    This is a placeholder and should be implemented to load the actual dataset.
    """
    print("Loading GSM8K dataset...")

    # return [{"question": "hi", "answer": "2"}]
    # Placeholder data
    testDataset = load_dataset("gsm8k", "main")["test"]
    gsm8k = []
    for i in range(4):
        ex = testDataset[i]
        gsm8k.append({"question": ex["question"], "answer": ex["answer"].split("####")[-1].strip()})

    return gsm8k

def run_gsm8k(generate_func):
    """
    Runs the GSM8K evaluation.

    Args:
        generate_func: The function to generate text from a prompt.
    """
    print("Running GSM8K evaluation...")
    dataset = load_gsm8k()
    correct = 0
    total_tokens = 0
    
    for item in dataset:
        question = item["question"]
        prompt = f"{question}\nRespond with no steps or explanation, just give the final answer in the format 'Answer: <number>'."
        # print("Asking question ", prompt)
        generated_text, num_tokens = generate_func(prompt)
        total_tokens += num_tokens
        # print("We had ", num_tokens, " tokens")
        # print("Answer was ", generated_text, " wanted ", item["answer"])
        pred_answer = extract_answer(generated_text)
        if pred_answer == item["answer"]:
            correct += 1
    
    accuracy = correct / len(dataset)
    print(f"GSM8K Accuracy: {accuracy}")
    return accuracy, total_tokens

if __name__ == '__main__':
    # This is a placeholder for the actual generate function
    def dummy_generate(prompt):
        return "4" # Dummy answer for testing
    
    run_gsm8k(dummy_generate)
