from datasets import load_dataset
import re
import ast
import random

def normalize_arc_answer(ans):
    return ans.strip().upper() if isinstance(ans, str) else None

letter_regex = re.compile(r"Answer:\s*([A-D])", re.IGNORECASE)

def extract_arc_answer(text):
    match = letter_regex.search(text)
    if match:
        return match.group(1).upper()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    if last in ["A", "B", "C", "D"]:
        return last
    return None


def load_arc_challenge():
    """
    Loads the ARC-Challenge dataset.
    This is a placeholder and should be implemented to load the actual dataset.
    """
    print("Loading ARC-Challenge dataset...")
    testDataset = load_dataset("ruibrogandrade/ARC-Challenge_PT-PT")["train"]
    arc = []
    for i in range(4):
        ex = testDataset[i]
        question = ex["question"]
        choices_dict = ast.literal_eval(ex["choices"])
        choices_list = choices_dict["text"]
        gold_answer = normalize_arc_answer(ex["answerKey"])
        choices_block = "\n".join(
                        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices_list)]
                    )
        prompt = (f"{question}\n\n"
                    f"{choices_block}\n\n"
                    "Give ONLY the final answer in the format: Answer: <letter>.\n"
                    "Do not show any reasoning steps."
                )
        arc.append({"question": prompt, "answer": gold_answer})

    return arc

def run_arc_challenge(generate_func):
    """
    Runs the ARC-Challenge evaluation.

    Args:
        generate_func: The function to generate text from a prompt.
    """
    print("Running ARC-Challenge evaluation...")
    dataset = load_arc_challenge()
    correct = 0
    for item in dataset:
        prompt = item["question"]
        # print("prompt is ", prompt)
        generated_text = generate_func(prompt)
        # print("response is ", generated_text)
        # print("detected answer is ", extract_arc_answer(generated_text))
        # print("correct answer is ", item['answer'])

        if item['answer'] == extract_arc_answer(generated_text):
            # print("correct")
            correct += 1
        # else:
        #     print("wrong")
    
    accuracy = correct / len(dataset)
    print(f"ARC-Challenge Accuracy: {accuracy}")
    return accuracy
    

if __name__ == '__main__':
    # This is a placeholder for the actual generate function
    def dummy_generate(prompt):
        return "air" # Dummy answer for testing
    
    run_arc_challenge(dummy_generate)
