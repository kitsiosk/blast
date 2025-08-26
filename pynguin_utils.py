import os
import io
import tarfile
import re
import openai
from groq import Groq
import json


def get_api_key(key_name, config_file="keys.json"):
    # Try environment variable first
    key = os.environ.get(key_name)
    if key:
        return key

    # Fallback: try JSON config
    if os.path.isfile(config_file):
        with open(config_file) as f:
            config = json.load(f)
        return config.get(key_name)

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY



def extract_module_from_diff(diff_text: str) -> str:
    """
    Extracts the Python module name from a git diff text, in a format compatible with Pynguin.
    
    Args:
        diff_text (str): The full diff text.
    
    Returns:
        str: The module name (e.g., 'lib.matplotlib' or 'lib.matplotlib.axis').
    """
    # Find the first "diff --git" line
    match = re.search(r'^diff --git a/(.+?) b/', diff_text, re.MULTILINE)
    if not match:
        raise ValueError("No valid diff header found.")
    
    file_path = match.group(1)  # example: 'lib/matplotlib/__init__.py' or 'lib/matplotlib/axis.py'
    
    if not file_path.endswith('.py'):
        raise ValueError(f"Edited file '{file_path}' is not a Python file.")
    
    # Remove the .py suffix
    module_path = file_path[:-3]
    
    # If it's an __init__.py, remove that part
    if module_path.endswith('__init__'):
        module_path = module_path[:-(len('__init__'))]
        if module_path.endswith('/'):
            module_path = module_path[:-1]
    
    # Convert slashes to dots
    module_name = module_path.replace('/', '.')
    
    return module_name


def extract_modules_from_diff(diff_text: str) -> list[str]:
    matches = re.findall(r'^diff --git a/(.+?) b/', diff_text, re.MULTILINE)
    module_names = []
    for file_path in matches:
        if not file_path.endswith('.py'):
            continue
        module_path = file_path[:-3]
        if module_path.endswith('__init__'):
            module_path = module_path[:-(len('__init__'))]
            if module_path.endswith('/'):
                module_path = module_path[:-1]
        module_name = module_path.replace('/', '.')
        module_names.append(module_name)
    return module_names



def run_in_container(container, command, logger, env={}, remove_from_logs=None):
    """
    Run a command inside the container and log the output.
    Raise RuntimeError if the command exits with non-zero code.
    """
    logger.info(f"Running command inside container: {command}")

    exec_result = container.exec_run(
        cmd=command,
        stdout=True,
        stderr=True,
        stream=False,
        environment=env
    )

    output = exec_result.output.decode('utf-8').strip()

    # use remove_from_logs=['a', ...] to shrink the log file
    for line in output.splitlines():
        if not remove_from_logs or not any([remove_this in line for remove_this in remove_from_logs]):
            logger.info(line)
        else:
            output.replace(line, '')

    return output


def copy_folder_from_container(container, container_folder_path, local_folder_path):
    """
    Copy a folder from inside the Docker container to a local folder.
    """
    bits, stat = container.get_archive(container_folder_path)
    file_like = io.BytesIO(b''.join(bits))
    
    with tarfile.open(fileobj=file_like) as tar:
        tar.extractall(path=local_folder_path)


def write_variable_to_container_file(container, x: str, container_path: str):
    """
    Write the contents of a Python variable x (str) into a file inside the Docker container.
    
    Args:
        container: The running Docker container object.
        x: The string content to write.
        container_path: The target path inside container, e.g., /testbed/mypatch.diff
    """
    data = x.encode('utf-8')
    tar_stream = io.BytesIO()

    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        tarinfo = tarfile.TarInfo(name=container_path.lstrip('/'))  # remove leading slash
        tarinfo.size = len(data)
        tar.addfile(tarinfo, io.BytesIO(data))

    tar_stream.seek(0)
    container.put_archive(path='/', data=tar_stream.getvalue())

def extract_failed_test_names(pytest_output: str) -> list:
    """
    Extracts the names of failed tests from pytest output.

    Args:
        pytest_output (str): The full output text from pytest.

    Returns:
        List[str]: A list of failed test names.
    """
    failed_tests = []
    # Find all lines starting with "FAILED " followed by the test path
    for match in re.finditer(r"^FAILED\s+([^\s]+)", pytest_output, re.MULTILINE):
        failed_tests.append(match.group(1))
    
    return failed_tests

def extract_passed_tests(pytest_output):
    match = re.search(r"(\d+)\s+passed", pytest_output)
    return int(match.group(1)) if match else None

def ensure_pytest_version(container, logger):
    from packaging import version  # very useful for version comparison
    """
    Checks the pytest version inside the container.
    If it's < 7.2, install pytest==7.2.
    """
    try:
        # Step 1: Check current pytest version inside testbed conda env
        output = run_in_container(
            container,
            "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python -c \"import pytest; print(pytest.__version__)\"'",
            logger
        ).strip()

        logger.info(f"Detected pytest version inside container: {output}")

        # Step 2: Compare versions
        current_version = version.parse(output)
        if current_version < version.parse("7.2"):
            logger.info(f"Pytest version {current_version} is <= 7.2; reinstalling pytest==7.2")
            
            run_in_container(
                container,
                "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pip install pytest==7.2'",
                logger
            )
        else:
            logger.info(f"Pytest version {current_version} is > 7.2; no need to reinstall.")

    except Exception as e:
        logger.error(f"Failed to check or reinstall pytest: {e}")
        raise


def query_model(prompt, model="gpt-4o", T=0.0):
    if model.startswith("gpt"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=T
        )
        return response.choices[0].message.content.strip()
    
    elif model.startswith("o3"): # does not accept temperature
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    
    elif model.startswith("o1"): # temperature does not apply in o1 series
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()


        return completion.choices[0].message['content']
    elif model.startswith("llama"):
        GROQ_API_KEY = get_api_key("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=model, 
            messages=messages, 
            max_tokens=700,
            temperature=T
        )

        return completion.choices[0].message.content
    elif model.startswith("deepseek"):
        GROQ_API_KEY = get_api_key("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an experienced software tester specializing in developing regression tests. Follow the user's instructions for generating a regression test. The output format is STRICT: do all your reasoning in the beginning, but the end of your output should ONLY contain python code, i.e., NO natural language after the code."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

def get_pynguin_command(module_name, budget_seconds, extra_flags, python_version):
    if python_version in ['3.8', '3.9']:
        timeout_command = "--budget " # pynguin 0.17
    else:
        timeout_command = "--maximum-search-time " # pynguin 0.40

    # hard timeout at the os level because sometimes the algorithm is stuck forever
    timeout = budget_seconds + 600
    pynguin_command = (
        f"bash -c '"
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate testbed && "
        f"timeout {timeout}s pynguin --project-path /testbed "
        " --seed 42 "
        f"--output-path /testbed/pynguin-tests "
        f"{timeout_command} {budget_seconds} "
        f"--module-name {module_name} {extra_flags}"
        f"'"
    )

    return pynguin_command

def extract_ids_from_file(file_path):
    # Define the pattern to search for
    pattern = r"Test generation failed for (\S+), skipping\.\.\."

    # Initialize an empty list to store the IDs
    ids = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the pattern in the line
            match = re.search(pattern, line)
            if match:
                # If a match is found, extract the ID and add it to the list
                ids.append(match.group(1))
    return ids

# To handle issue where sympy confuses paths and things that import sympy.core.numbers 
# is a class in sympy/core/core.py, while it actually is the file sympy/core/numbers.py
def transform_sympy_core_aliased_imports(code_str):
    """
    Transforms 'import sympy.core.<module> as <alias>' into
    'from sympy.core import <module> as <alias>'
    """
    pattern = re.compile(
        r'^\s*import\s+sympy\.core\.([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$',
        re.MULTILINE
    )

    def replacer(match):
        module = match.group(1)
        alias = match.group(2)
        print(f'from sympy.core import {module} as {alias}')
        return f'from sympy.core import {module} as {alias}'

    return pattern.sub(replacer, code_str)

def transform_sympy_simplify_aliased_imports(code_str):
    """
    Transforms 'import sympy.simplify.<module> as <alias>' into
    'from sympy.simplify import <module> as <alias>'
    """
    pattern = re.compile(
        r'^\s*import\s+sympy\.simplify\.([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$',
        re.MULTILINE
    )

    def replacer(match):
        module = match.group(1)
        alias = match.group(2)
        print(f'from sympy.simplify import {module} as {alias}')
        return f'from sympy.simplify import {module} as {alias}'

    return pattern.sub(replacer, code_str)

def transform_sympy_sets_aliased_imports(code_str):
    """
    Transforms 'import sympy.sets.<module> as <alias>' into
    'from sympy.sets import <module> as <alias>'
    """
    pattern = re.compile(
        r'^\s*import\s+sympy\.sets\.([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$',
        re.MULTILINE
    )

    def replacer(match):
        module = match.group(1)
        alias = match.group(2)
        print(f'from sympy.sets import {module} as {alias}')
        return f'from sympy.sets import {module} as {alias}'

    return pattern.sub(replacer, code_str)

def transform_sympy_matrices_aliased_imports(code_str):
    """
    Transforms 'import sympy.matrices.expressions.<module> as <alias>' into
    'from sympy.matrices.expressions import <module> as <alias>'
    """
    pattern = re.compile(
        r'^\s*import\s+sympy\.matrices\.expressions\.([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$',
        re.MULTILINE
    )

    def replacer(match):
        module = match.group(1)
        alias = match.group(2)
        print(f'from sympy.matrices.expressions import {module} as {alias}')
        return f'from sympy.matrices.expressions import {module} as {alias}'

    return pattern.sub(replacer, code_str)

def transform_sympy_logic_aliased_imports(code_str):
    """
    Transforms 'import sympy.logic.<module> as <alias>' into
    'from sympy.logic import <module> as <alias>'
    """
    pattern = re.compile(
        r'^\s*import\s+sympy\.logic\.([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$',
        re.MULTILINE
    )

    def replacer(match):
        module = match.group(1)
        alias = match.group(2)
        print(f'from sympy.logic import {module} as {alias}')
        return f'from sympy.logic import {module} as {alias}'

    return pattern.sub(replacer, code_str)

def build_prompt_for_seeding_pg(issue_description, patch, module_name, module_code):
    # TODO: maybe give more explictly the available functions/methods?
    prompt = f"""You are a software tester with experience in the Pynguin test generation tool.
A user posted the following issue in your repository.
### Issue/bug:
{issue_description}

A developer opened a pull request to resolve the above issue with the changes below.
### Developer changes:
{patch}

From the developer changes, we see that the module-under-test is `{module_name}` and the associated code (on top of which the changes will be applied) is shown below:
```python
{module_code}
```
Our end goal is to use Pynguin to generate python tests that fail when ran in the buggy code and pass when we run it in the fixed code, hence serving as a regression test for the issue above.
Your task is to **generate tests that will serve as a seed to be further mutated by Pynguin**, and for this reason they should
A) Be as semantically close as possible to serving as regression tests for the above issue, i.e., by failing in the pre-PR code and passing in the post-PR code, and
B) **Strictly** following the internal Pynguin test fromat, which is described in detail below. This is more important that being semantically related to the issue, which means that the restrictions below should ALWAYS be met, even if your test is not a regression test, because Pynguin will mutate it further. In other words, even partial tests that contain key values (e.g., strings, integers) that could reproduce the issue are useful.

### What is permitted in the internal Pynguin test format, and hence, in the tests you will generate:
All test functions must be top-level `def` functions (no classes or nested functions).

###### Disallowed Statements
- Decorators (e.g., `@pytest.mark.parametrize`)
- Control flow constructs: `for`, `while`, `if`, `try`, `with`, `return`, `raise`, `break`, `continue`
- Assignments using unsupported expressions:
- **Binary operations**: `x + y`, `a * b`, `a and b`, `a or b`
- **Indexing**: `x = arr[0]`, `x = d['key']`
- **Attribute access**: `x = obj.attr`
- **Comparison operators**: `x = a > b`, `x = a == b`
- **Comprehensions**: [x for x in y], etc
- **Lambda expressions**
- Function calls to unknown or external modules not part of the module-under-test
- Assertions on expressions not first assigned to a variable
- Imports that are unused or refer only to unrelated modules
- Imports in the global namespace, outside of functions; each function should import what it needs locally 

###### Allowed Statements
- `import <module> as <alias>` (must use `as` form for all imports)
- Assignments where the right-hand side is:
- A literal constant: `None`, strings, numbers
- A function call (see below)
- A list, dict, tuple, or set literal
- A unary operation (`-x`, `not x`)
- Assert statements like `assert x is not None` if `x` was defined earlier
- Standalone expressions that are valid function calls
- Imports inside functions

###### Function Calls
- Calls must be to functions or constructors from the module-under-test
- Use `import <module> as <alias>` and invoke with `alias.func(...)`
- All arguments must be previously assigned to variables
- Do not use inline expressions or nested function calls as arguments

###### Goal
Each test must be a **Pynguin-compatible seed**: flat, minimal, valid, and parsable. Output **only** raw test code, with **no explanation or extra text**.


Return one python test function without any explanation or anything else. 
"""
    return prompt


def get_pynguin_3_10_patch():
    patch = """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1447,6 +1447,8 @@ def __analyse_included_functions(
         if current in seen_functions:
             continue
         seen_functions.add(current)
+        if current.__module__ is None or type(current.__module__) == type(None):
+            continue
         __analyse_function(
             func_name=current.__qualname__,
             func=current,
"""
    return patch