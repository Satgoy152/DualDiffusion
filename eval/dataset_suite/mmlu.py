import re
import random
from datasets import load_dataset

# -------------------------
# Normalize gold answer
# -------------------------
def normalize_mmlu_answer(ans):
    # If ans is an int 0-3, map to A-D
    if isinstance(ans, int):
        return chr(65 + ans)  # 0 -> A, 1 -> B, etc.
    # If already a string, normalize
    return ans.strip().upper()


# -------------------------
# Extract model answer
# -------------------------
letter_regex = re.compile(r"Answer:\s*([A-D])", re.IGNORECASE)

def extract_mmlu_answer(text):
    # Look for explicit "Answer: X"
    match = letter_regex.search(text)
    if match:
        return match.group(1).upper()

    # Fallback: last non-empty line is a single letter
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    last = lines[-1]
    if last in ["A", "B", "C", "D"]:
        return last

    return None

def load_mmlu():
    """
    Loads the MMLU dataset.
    This is a placeholder and should be implemented to load the actual dataset.
    """
    print("Loading MMLU dataset...")
    testDataset = load_dataset('cais/mmlu', 'all')['test']
    mmlu = []
    for i in range(4):
        ex = testDataset[i]
        question = ex["question"]
        choices = ex["choices"]
        choices_block = "\n".join(
                    [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
                    )
        prompt = (f"{question}\n\n"
                f"{choices_block}\n\n"
                "Respond ONLY with one letter from A, B, C, or D"
                "Format: Answer: <letter>\n"
                "Do not show steps.")
        answer = normalize_mmlu_answer(ex["answer"])
        mmlu.append({"question": prompt, "answer": answer})

    return mmlu


def run_mmlu(generate_func):
    """
    Runs the MMLU evaluation.

    Args:
        generate_func: The function to generate text from a prompt.
    """
    print("Running MMLU evaluation...")
    dataset = load_mmlu()
    correct = 0
    for item in dataset:
        prompt = item["question"]
        print("prompt is ", prompt)
        generated_text = generate_func(prompt)
        print("response is ", generated_text)
        print("detected answer is ", extract_mmlu_answer(generated_text))
        print("correct answer is ", item['answer'])

        if item['answer'] == extract_mmlu_answer(generated_text):
            print("correct")
            correct += 1
        else:
            print("wrong")
    
    accuracy = correct / len(dataset)
    print(f"MMLU Accuracy: {accuracy}")
    return accuracy

if __name__ == '__main__':
    # This is a placeholder for the actual generate function
    def dummy_generate(prompt):
        return "Paris" # Dummy answer for testing
    
    run_mmlu(dummy_generate)
