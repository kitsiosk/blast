import os
from typing import List, Tuple
import openai
import json
from utils import *
import pandas as pd

def build_prompt_zeroshot(repo_name, issue_desc, patch):
    prompt = f"""Consider yourself an experienced professional software engineer working on a project. 
    An issue has been created, and you need to develop a test case that will fail on the current version of the code and pass on the pull request that addressed the issue. 
    You will be given three pieces of information.  
    1. Repository name: {repo_name}
    2. Issue Description: {issue_desc}  
    3. The Pull Request patch: {patch}
    Now that all pieces of information are provided to you, simply write down the test with all necessary imports after the "Answer: " token. 
    There is no need to explain the solution. 
    We will copy your generated test and paste it into a Python (.py) file to execute it. 
    Please keep the indentation intact and do not add any spaces or symbols (like `>') that make the program uncompilable.  
    Answer:"""

    return prompt

d = pd.read_pickle('tdd_bench_verified.pickle')

model = 'gpt-4o'
#model = 'llama-3.3-70b-versatile'
#model = 'deepseek-r1-distill-llama-70b'
T = 0.0
use_cached_preds = False # if True, read from file below. If false, query API and write to file

json_file_path_final_preds = f"zeroshot_{model}.jsonl"
failed = []
fname = "test_tdd.py"
for i in range(len(d)):
    instance_id = d['instance_id'].iloc[i]
    print("Starting %s (%d)" % (instance_id, i))
    _, repo_folder, _ = parse_instanceID_string(instance_id)

    repo_base   = 'repos/'
    repo_dir    = repo_base + repo_folder
    base_commit = d['base_commit'].iloc[i]

    patch = d['patch'].iloc[i]
    issue_desc = d['problem_statement'].iloc[i]
    
    try:
        prompt = build_prompt_zeroshot(repo_folder, issue_desc, patch)
        response = query_model(prompt, model=model)
        new_test = response.split('```python')[-1].replace('```', '')
        
        diff = unified_diff_newfile(new_test, fname)
        print("Generated response for i=%d (%s)" % (i, instance_id))
    except Exception as e:
        print("Failed i=%d (%s): %s" % (i, instance_id, e))
        continue

    data = {
        "model_name_or_path": model,
        "instance_id": instance_id,
        "model_patch": diff
    }
    jsonl_line = json.dumps(data)
    with open(json_file_path_final_preds, "a") as f:
        f.write(jsonl_line + "\n")  # Add newline character after each JSON object
