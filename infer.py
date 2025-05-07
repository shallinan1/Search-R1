import os
# Set HuggingFace cache directory
os.environ["HF_HOME"] = "/gscratch/xlab/hallisky/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/gscratch/xlab/hallisky/cache/"
os.environ["HF_DATASETS_CACHE"] = "/gscratch/xlab/hallisky/cache/"
os.environ["TORCH_HOME"] = "/gscratch/xlab/hallisky/cache/"

import transformers
import torch
import random
from datasets import load_dataset
import requests
import argparse
import sys
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"search_r1_infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Answer questions using Search-R1 model')
parser.add_argument('--questions', nargs='+', help='List of questions to answer')
parser.add_argument('--model_id', type=str, default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo", 
                    help='Model ID to use for inference')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
args = parser.parse_args()

# Default question if none provided
default_questions = [
    "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?",
    "What is the capital of France?",
    "Who wrote the novel 'Pride and Prejudice'?"
]

questions = args.questions if args.questions else default_questions
logger.info(f"Processing {len(questions)} questions")

# Model ID and device setup
model_id = args.model_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using model: {model_id} on device: {device}")

# Initialize the tokenizer and model
logger.info("Loading tokenizer and model...")
start_time = time.time()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

def answer_question(question):
    # Prepare the question
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    logger.info(f"Processing question: '{question}'")
    
    # Prepare the message
    prompt = f"""Answer the given question. \
    You must conduct reasoning inside <think> and </think> first every time you get new information. \
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
    You can search as many times as your want. \
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(f"Question: {question}")
    print(prompt)
    
    cnt = 0
    final_output = ""
    
    # Encode the chat-formatted prompt and move it to the correct device
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        logger.info(f"Generation iteration {cnt+1}, input length: {input_ids.shape[1]} tokens")
        
        # Generate text with the stopping criteria
        start_time = time.time()
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature
        )
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds, produced {outputs.shape[1] - input_ids.shape[1]} new tokens")

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info("Reached EOS token, generation complete")
            print(output_text)
            final_output += output_text
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...')
            logger.info(f"Search query detected: '{tmp_query}'")
            search_results = search(tmp_query)
        else:
            logger.warning("No search query found in generated text")
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        logger.info(f"Completed iteration {cnt}, adding search results to prompt")
        print(search_text)
        final_output += search_text
    
    logger.info(f"Question answered after {cnt} search iterations")
    return final_output

# Process all questions
logger.info(f"Starting to process {len(questions)} questions")
start_time = time.time()

for i, question in enumerate(questions):
    logger.info(f"Processing question {i+1}/{len(questions)}: '{question}'")
    question_start_time = time.time()
    
    print(f"\n\n===== Question {i+1}/{len(questions)} =====")
    answer = answer_question(question)
    
    question_time = time.time() - question_start_time
    logger.info(f"Question {i+1} completed in {question_time:.2f} seconds")
    
    print(f"\n----- Answer for Question {i+1} -----")
    print(answer)
    print("\n" + "="*50 + "\n")

total_time = time.time() - start_time
logger.info(f"All questions processed in {total_time:.2f} seconds")
