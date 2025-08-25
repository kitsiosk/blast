import os
from typing import List, Tuple
import json
from utils import *
import pandas as pd

# ----------------------------
# Step 1: Select Test File
# ----------------------------

def select_test_file(repo_name: str, issue_description: str, test_files: List[str]) -> str:
    prompt = f"""Suppose you are a very experienced developer. An issue has been created, and you need to choose the best possible test file to write the test case. You will be given three pieces of information. Please write down the test file name after the "Answer:" token.

1. Repository name: {repo_name}
2. Issue Description: {issue_description}
3. List of test file(s) that will be updated or added : {test_files}

Answer:"""
    return query_model(prompt)

def extract_related_functions(issue_description: str) -> List[str]:
    prompt = f"""Suppose you are a very experienced developer. The following issue has been created: {issue_description}

Now that all pieces of information are provided to you, simply write down the function names and test names that are written by the developers and appear in the issue description. Sort them according to relevance. Put the most important ones at the top. No need to write any explanation or anything else.

Write one name each line after the "Answer: " token.

Answer:"""
    raw_output = query_model(prompt)
    return [line.strip() for line in raw_output.splitlines() if line.strip()]

# ----------------------------
# Step 3: Generate and Insert Test Function
# ----------------------------

def generate_test_function(
    repo_name: str,
    issue_description: str,
    related_functions: List[str],
    patch: str,
    test_file_skeleton: str,
    model: str,
) -> Tuple[str, str, str, str]:
    prompt = f"""Suppose you are a very experienced developer. An issue has been created, and you need to write a test that reproduces the issues by failing in the code before the issue patch and passing in the code after the issue patch.
    
You are given the following pieces of information:
1. Repository name: {repo_name}
2. Issue Description: {issue_description}
3. Issue-related Functions: {related_functions}
4. Patch: {patch}
5. Test File Skeleton: {test_file_skeleton}

You have two options.
1. Modify a function: you can modify existing function(s) of the Test File Skeleton. Please rewrite the complete function. Here is the format:
<Modified> <Position>
<body of the function>

2. Write down a completely new function. Here is the format:
<New> <Position>
<body of the function>

For "Position", please write down the name of the adjacent prior function.
Return only one test function at the default indentation level, with the imports INSIDE the function, WITHOUT considering the integration to the test file, e.g., in a unittest.TestCase class because your raw test function will then be inserted after the function you specify by us.
The output format is STRICTLY the following:

```python
<New> test_function_name_after_which_to_place
# your test here
```

An example is:
```python
<New> test_division
def test_division()
    import divide from module
    assert divide(10, 2) == 5
```

Answer:"""

    output = query_model(prompt, model=model)
    return output

import ast
import os
from typing import List, Tuple

def find_function_definitions(
    repo_folder: str, 
    function_names: List[str]
) -> List[Tuple[str, str]]:
    """
    Search all Python files under repo_folder for definitions of the given function names.
    Returns a list of (file_path, source_code_of_function) tuples.
    """
    matched = []
    normalized_names = {name.strip() for name in function_names}

    for root, _, files in os.walk(repo_folder):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source)

                    # Keep track of lines to extract code
                    lines = source.splitlines()

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in normalized_names:
                            start_line = node.lineno - 1
                            end_line = getattr(node, "end_lineno", None)
                            if end_line is None:
                                # Estimate end by walking sibling nodes
                                end_line = estimate_function_end_line(node, lines)
                            func_source = "\n".join(lines[start_line:end_line])
                            matched.append((file_path, func_source))
                except Exception as e:
                    print(f"[Warning] Could not process {file_path}: {e}")
    return matched

def estimate_function_end_line(node, lines):
    """
    Fallback in case ast doesn't have end_lineno (older Python versions).
    We'll walk down until we hit a line with less or equal indentation.
    """
    start = node.lineno
    start_indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
    for i in range(start, len(lines)):
        if lines[i].strip() == "":
            continue
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        if current_indent <= start_indent and not lines[i].lstrip().startswith("#"):
            return i
    return len(lines)

import re
from typing import Tuple

def apply_llm_generated_test(llm_output: str, test_skeleton: str) -> str:
    """
    Parse the LLM output and apply the generated test function
    to the given test_skeleton content.

    Returns updated test_skeleton content.
    """
    lines = llm_output.strip().splitlines()
    if not lines:
        raise ValueError("Empty LLM output.")

    # Step 1: Parse header
    header_match = re.match(r"<?(New|Modified)>?\s+<?(\w+)>?", lines[0])
    if not header_match:
        raise ValueError(f"Could not parse header line: {lines[0]}")
    mode = header_match.group(1)
    reference_function = header_match.group(2)
    new_func_body = "\n".join(lines[1:])

    # Step 2: Apply insertion or modification
    return _insert_or_replace_function(test_skeleton, mode, reference_function, new_func_body)

def _insert_or_replace_function(test_skeleton: str, mode: str, reference_function: str, new_body: str) -> str:
    lines = test_skeleton.splitlines()
    output = []
    inserted = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Match the reference function definition
        match = re.match(rf'^(\s*)def {reference_function}\b', line)
        if match and not inserted:
            indent = match.group(1)

            # Find function block extent
            start_line = i
            end_line = i + 1
            while end_line < len(lines):
                next_line = lines[end_line]
                if next_line.strip() == "":
                    end_line += 1
                    continue
                current_indent = len(next_line) - len(next_line.lstrip())
                if current_indent <= len(indent) and not next_line.lstrip().startswith("#"):
                    break
                end_line += 1

            if mode == "Modified":
                # Replace entire function
                new_body_indented = indent + new_body.replace("\n", "\n" + indent)
                output.extend(lines[:start_line])
                output.append(new_body_indented)
                output.extend(lines[end_line:])
                inserted = True
                break

            elif mode == "New":
                # Insert after this function
                output.extend(lines[:end_line])
                output.append("")  # Blank line
                new_body_indented = indent + new_body.replace("\n", "\n" + indent)
                output.append(new_body_indented)
                output.extend(lines[end_line:])
                inserted = True
                break
        i += 1

    if not inserted:
        # Fallback: append at the end
        output = lines + ["", new_body, ""]

    return "\n".join(output)

def find_test_filenames(repo_folder: str) -> List[str]:
    """
    Recursively search repo_folder for Python test files.
    A test file is defined as one starting with 'test_' and ending with '.py'.

    Returns:
        A list of relative file paths (from repo_folder root).
    """
    test_files = []
    for root, _, files in os.walk(repo_folder):
        for filename in files:
            if filename.startswith("test_") and filename.endswith(".py"):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_folder)
                test_files.append(rel_path)
    return test_files

d = pd.read_pickle('tddbench_verified_processed.pickle')

model = 'gpt-4o'
#model = 'llama-3.3-70b-versatile'
#model = 'deepseek-r1-distill-llama-70b'

T = 0.0
json_file_path_final_preds = f"autotdd_{model}.jsonl"
failed = []

# Get all the trivial instances
with open('trivial_ids.txt', 'r') as f:
    trivial_ids = f.readlines()
trivial_ids = [iid.strip() for iid in trivial_ids]

for i in range(len(d)):
    instance_id = d['instance_id'].iloc[i]
    if instance_id in trivial_ids: # skip trivial instances
        continue
        
    print("Starting %s (%d)" % (instance_id, i))
    _, repo_folder, _ = parse_instanceID_string(instance_id)

    repo_base   = 'repos/'
    repo_dir    = repo_base + repo_folder
    base_commit = d['base_commit'].iloc[i]

    patch = d['patch'].iloc[i]
    issue_desc = d['problem_statement'].iloc[i]
    

    current_branch = run_command("git rev-parse --abbrev-ref HEAD", cwd=repo_dir)
    run_command(f"git checkout {base_commit}", cwd=repo_dir)
    try:
        test_candidates = find_test_filenames(repo_dir)
        
        selected_file = select_test_file(repo_folder, issue_desc, test_candidates)
        selected_file = selected_file.split(':')[-1].strip()
        selected_file_full = os.path.join(repo_dir, selected_file)
        if not os.path.isfile(selected_file_full):
            print("LLM failed to return an existing file")
            continue
        with open(selected_file_full, 'r') as f:
            test_skeleton = f.read()
        #print(f"Selected Test File: {selected_file}")

        funcs = extract_related_functions(issue_desc)
        funcs = [xx for xx in funcs if not xx.startswith('__')] # remove __init__ etc because there are 100s of them and the prompt blows up

        #print(f"Issue-related Functions: {funcs}")
        found_defs = find_function_definitions(repo_dir, funcs)
        context = ""
        for path, code in found_defs:
            context += f"\nIn file: {path}\n---\n{code}\n"
        #print("%d functions found" % len(found_defs))

        llm_out = generate_test_function(repo_folder, issue_desc, context, patch, test_skeleton, model)
        if model == 'deepseek-r1-distill-llama-70b':
            llm_out = llm_out.split('</think>')[-1].strip()
        llm_out = llm_out.replace('```python', '').replace('```', '')
        updated = apply_llm_generated_test(llm_out, test_skeleton)
        
        diff = unified_diff(test_skeleton, updated, selected_file, selected_file)

    except Exception as e:
        print("The following exception occured for i=%d: %s Skipping" % (i, e))
        failed.append(instance_id)
        continue
    finally:
        run_command(f"git checkout {current_branch}", cwd=repo_dir)  # Reset to the original commit


    data = {
        "model_name_or_path": model,
        "instance_id": instance_id,
        "model_patch": diff
    }
    jsonl_line = json.dumps(data)
    with open(json_file_path_final_preds, "a") as f:
        f.write(jsonl_line + "\n")  # Add newline character after each JSON object