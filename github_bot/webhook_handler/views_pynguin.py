import hmac
import hashlib
import json
import requests
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
import re
import os
import logging
from datetime import datetime
import traceback
import subprocess
import jwt
import time
from .paper_utils import *
import uuid

use_llm_seeds = False

logger = logging.getLogger("myapp")
logger.debug("Entered webhook")

# Directory where webhook requests will be saved. Replace with your desired path.
is_in_server = os.path.isdir("/home/ubuntu")
if is_in_server:
    WEBHOOK_RAW_LOG_DIR = "/home/ubuntu/logs/raw/" # for raw requests
    WEBHOOK_LOG_DIR     = "/home/ubuntu/logs/" # for parsed requests
else:
    WEBHOOK_RAW_LOG_DIR = "/Users/konstantinos/local-desktop/Test Generation Project/github_bot_logs_paper/" # for raw requests
    WEBHOOK_LOG_DIR     = "/Users/konstantinos/local-desktop/Test Generation Project/github_bot_logs_paper/" # for parsed requests

GITHUB_TOKEN = get_api_key("GITHUB_PAT")

# GitHub webhook secret. Must be the same when setting up the hook in GH.
GITHUB_WEBHOOK_SECRET = "1234" # placeholder, change if needed


HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}

openai.api_key = OPENAI_API_KEY

comment_template_generation = """Hi! 🤖 The test below is automatically generated and could serve as a regression test for this PR because it:
- passes, and
- fails in the codebase before the PR.

```python
%s
```

If you find this regression test useful, feel free to insert it to your test suite.
Our automated pipeline inserted the test at the end of the `%s` file before running it.

This is part of our research at the [ZEST](https://www.ifi.uzh.ch/en/zest.html) group of University of Zurich in collaboration with [Mozilla](https://www.mozilla.org).
If you have any suggestions, questions, or simply want to learn more, feel free to contact us at konstantinos.kitsios@uzh.ch and mcastelluccio@mozilla.com.

<details>
<summary> Click to see which lines were covered.</summary>

```diff
%s
```

Line coverage\\* achieved: %0.1f%%

\\* Line coverage is calculated over the lines added in this PR.

<details>
""" 


def run_pynguin(payload, dockerfile=None, 
        model_test_generation=None, 
        iAttempt=0,
        post_comment=False,
        model="gpt-4o",
        timestamp=0):

    # Extract data from payload
    pr_number      = payload["pull_request"]["number"]
    pr_title       = payload["pull_request"]["title"]
    pr_description = payload["pull_request"]["body"]
    pr_url         = payload["pull_request"]["url"]
    owner          = payload["repository"]["owner"]["login"]
    repo           = payload["repository"]["name"]
    diff           = payload["pull_request"]["diff_url"]
    base_branch    = payload["pull_request"]["base"]["ref"]
    base_commit    = payload["pull_request"]["base"]["sha"]
    head_branch    = payload["pull_request"]["head"]["ref"]
    head_commit    = payload["pull_request"]["head"]["sha"]
    instance_id    = f"{owner}__{repo}-{pr_number}"
    image_tag      = f"image_{instance_id}"
    if pr_description is None:
        pr_description = ""

    os.makedirs(WEBHOOK_LOG_DIR, exist_ok=True)
    this_instance_log_dir = os.path.join(WEBHOOK_LOG_DIR, "pynguin_%d_%s_%s"%(use_llm_seeds, instance_id, timestamp))
    os.makedirs(this_instance_log_dir, exist_ok=True)


    # Get the file contents from the github API (we could also get them by cloning the repo in the docker)
    files = fetch_pr_files(pr_number, owner, repo)

    code_fname_arr          = []
    code_content_before_arr = []
    code_content_after_arr  = []
    test_fname_arr          = []
    test_content_before_arr = []
    test_content_after_arr  = []
    at_least_one_python_code_file = False
    for file_dict in files:
        # Get the version of the file AFTER the PR
        fname = file_dict["filename"]
        after_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{head_commit}/{fname}"
        response_after = requests.get(after_url, headers=HEADERS)
        if response_after.status_code == 200:
            content_after = response_after.text
        else:
            print("File %s not found in url %s" % (fname, after_url))
            content_after = "" # probably file deleted

        # Get the version of the file BEFORE the PR
        before_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{base_commit}/{fname}"
        response_before = requests.get(before_url, headers=HEADERS)
        if response_before.status_code == 200:
            content_before = response_before.text
        else:
            content_before = "" # probably file deleted

        if is_test_file(fname):
            test_fname_arr.append(fname)
            test_content_before_arr.append(content_before)
            test_content_after_arr.append(content_after)
        else:
            code_fname_arr.append(fname)
            code_content_before_arr.append(content_before)
            code_content_after_arr.append(content_after)
            if fname.endswith(".py") and not at_least_one_python_code_file:
                at_least_one_python_code_file = True

    if not at_least_one_python_code_file: # if the PR changed only non-python files return
        logger.info("No .py code files (except maybe for test) were modified, raising error")
        raise ValueError("No .py code files in %s, raising error to skip writing to results csv file" % pr_number)
    
    # If test file already exists, we do amplification, otherwise generation
    contains_test_file = len(test_fname_arr) > 0

    # Get golden code patch
    diffs = []
    for (fname, fcontent_before, fcontent_after) in zip(code_fname_arr, code_content_before_arr, code_content_after_arr):
        diff = unified_diff_with_function_context(fcontent_before, fcontent_after, fname)
        diffs.append(diff)
    golden_code_patch = "\n\n".join(diffs)+"\n\n"

    # We re-calculate the code contents after because we want to capture the offset of the golden patch
    code_content_after_arr_from_patch, stderr = apply_patch(code_content_before_arr, golden_code_patch)
    try:
        offsets = extract_offsets_from_stderr(stderr)
    except AssertionError as e:
        logger.info("Different offsets in a single file for %s, skipping" % instance_id)
        exit(0)

    # Slice golden files
    if code_fname_arr: # sometimes all the changes are counted as tests e.g., test_test_scheduling.py
        code_content_before_sliced_arr = slice_golden_file(
            code_content_before_arr, 
            golden_code_patch,
            "",
            return_file="pre",
            append_line_numbers=True
            )
    else:
        code_content_before_sliced_arr = code_content_before_arr.copy()


    # Check if the PR is linked to a GH Issue
    has_linked_issue, linked_issue, issue_title, issue_description, issue_comments = check_if_has_linked_issue(pr_description, owner, repo)
    issue_description = f"{issue_title}\n{issue_description}"
    if has_linked_issue:
        logger.info("Linked issue: %d" % linked_issue)
    else:
        logger.info("No linked issue")

    # Build Docker image of the repo-under-test
    client = docker.from_env()
    if dockerfile is None: # if no mock dockerfile given
        if repo == "bugbug" and owner == "kitsiosk" and pr_number == 5:
            dockerfile = f"dockerfiles/Dockerfile_bugbug_old1" # for integration testing
        else:
            dockerfile = f"dockerfiles/pynguin/Dockerfile_{repo}"
    build_docker_image(client, dockerfile, base_commit, image_tag=image_tag)

    # Create central datastructure containing all the PR/Issue data
    instance = {}
    instance["instance_id"]          = instance_id
    instance["patch"]                = golden_code_patch
    instance["golden_test_names"]    = test_fname_arr
    instance["golden_test_contents"] = test_content_after_arr
    instance["hints_text"]           = ""
    instance["golden_code_names"]    = code_fname_arr
    instance["golden_code_contents"] = code_content_before_arr
    instance["title"]                = pr_title
    instance["description"]          = pr_description
    instance["base_commit"]          = base_commit
    instance["problem_statement"]    = issue_description
    instance["golden_code_contents_sliced_long"] = code_content_before_sliced_arr


    logger.info("=============== Test Generation Started ===============")
    generation_completed = False

    # Calculate temporal coupling to find where to inject the test
    tmp_repo_dir = "tmp_repo_dir_" + str(uuid.uuid1())[:8]
    res = subprocess.run(["git", "clone", f"https://github.com/{owner}/{repo}.git", tmp_repo_dir], capture_output=True, check=True)
    try:
        test_filename, test_file_content, test_file_content_sliced = get_contents_of_test_file_to_inject(instance, tmp_repo_dir)
        if test_filename == "":
            logger.info("No suitable file found for %s, skipping" % instance_id)
            exit(0)
        test_filename = test_filename.replace(tmp_repo_dir+'/', '')
        instance['predicted_test_file_content_sliced'] = test_file_content_sliced
    finally:
        res = subprocess.run(["rm", "-rf", tmp_repo_dir], capture_output=True, check=True)


    #### Run test in pre-PR codebase
    file_to_test = [x for x in code_fname_arr if x.endswith('.py')][0]
    print(file_to_test)

    #### Run test in post-PR codebase
    golden_code_patch = instance["patch"]

    if use_llm_seeds:
        issue_description = instance['problem_statement']
        module_code = '\n'.join(instance['golden_code_contents_sliced_long'])
        patch = instance['patch']
        module_name = file_to_test[:-3]
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


Return up to five (5) python test functions without any explanation or anything else. 
"""

        model_response = query_model(prompt)
        generated_test = "\n".join(model_response.splitlines()[1:-1])

        # move the folder pynguin-tests/ of the container to the folder `this_log_dir` (python variable name) of my local machine
        os.makedirs(this_instance_log_dir, exist_ok=True)
        with open(os.path.join(this_instance_log_dir, "model_response.txt"), "w") as f:
            f.write(model_response)
        with open(os.path.join(this_instance_log_dir, "generated_test.py"), "w") as f:
            f.write(generated_test)
        with open(os.path.join(this_instance_log_dir, "prompt.md"), "w") as f:
            f.write(prompt)

        module_name_short = module_name.split('/')[-1]
        #TODO: pass test_file_name, and generated_test to function 
        pynguin_test_file_name = os.path.join("/app/testbed/pynguin_seed/", f"test_{module_name_short}.py")

        extra_flags = "--initial_population_seeding True --initial_population_data /app/testbed/pynguin_seed/ -v "
    else:
        generated_test=None
        pynguin_test_file_name=None
        extra_flags=""

    failed_tests, pynguin_test = run_pynguin_in_container(client, image_tag, file_to_test, 
                                                          this_instance_log_dir, golden_code_patch=golden_code_patch,
                                                          use_llm_seeds=use_llm_seeds, generated_test=generated_test,
                                                          test_file_name=pynguin_test_file_name, extra_flags=extra_flags
                                                          )
    
    
    if failed_tests:
        # Add comment to the PR
        comment = comment_template_generation % (pynguin_test, 
                                                 test_filename,
                                                 golden_code_patch, 
                                                 0) #TODO: calculate coverage

        if post_comment:
            status_code, response_data = add_comment_to_pr(owner, repo, pr_number, comment)
            if status_code == 201:
                logger.info("Comment added successfully!")
            else:
                logger.info(f"Failed to add comment: {status_code}", response_data)

        else:
            logger.info("Debugging: would add comment to PR")
            #logger.info("Debugging: would add this comment to PR:\n%s\n" % comment)

        
        generation_completed = True
    elif not isFail2Pass:
        logger.info("No Fail-to-Pass test generated")
        generation_completed = False

    logger.info("=============== Test Generation Finished ===============")

    # Whether to stop or try again with different prompt inputs
    stop = generation_completed
    return JsonResponse({"status": "success"}), stop


def verify_signature(request):
    """Verify the webhook signature."""
    signature = request.headers.get('X-Hub-Signature-256')
    if not signature:
        return False
    sha_name, signature = signature.split('=')
    if sha_name != 'sha256':
        return False
    # Encode the request body using the same secret
    mac = hmac.new(GITHUB_WEBHOOK_SECRET.encode(), msg=request.body, digestmod=hashlib.sha256)
    # If the two encodings are the same, we are good.
    return hmac.compare_digest(mac.hexdigest(), signature)

def check_if_has_linked_issue(pr_description, owner, repo):
    """
    Extracts issue numbers from a PR description.
    Supports both "#issue_number" and full URL references.
    """
    # Pattern to match all occurrences like #123 (we assume that it is followed by Fixes, Resolves, etc)
    issue_pattern = r'#(\d+)'

    # Pattern for full URL references: Fixes https://github.com/owner/repo/issues/56
    url_pattern = fr'\bhttps://github\.com/{re.escape(owner)}/{re.escape(repo)}/issues/(\d+)\b'

    # Extract matches
    matches = re.findall(issue_pattern, pr_description)
    url_matches = re.findall(url_pattern, pr_description)

    # Combine both types of matches
    all_matches = matches + url_matches
    for match in all_matches:
        match_int = int(match)  # Convert from string to int
        issue_or_pr, title, description, comments = is_issue_or_pr(owner, repo, match_int)
        if issue_or_pr == "Issue":
            logger.info("Linked with issue #%d" % match_int)
            return True, match_int, title, description, comments  # Stop at the first issue found

    return False, None, None, None, None


def is_issue_or_pr(owner, repo, number):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    
    response = requests.get(url, headers=HEADERS)    
    
    if response.status_code == 200:
        issue_data = response.json()
        comments = get_issue_comments(owner, repo, number)

        if "pull_request" in issue_data:
            return "PR", None, None, comments
        else:
            return "Issue", issue_data["title"], issue_data["body"], comments
    else:
        logger.info(f"Failed to fetch data for #{number}: {response.status_code}")
        return None, None, None, None
    

def get_issue_comments(owner, repo, number):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        comments = response.json()
        comments_concat = ""
        for comment in comments:
            comments_concat += comment['body']
        
    else:
        comments_concat = ""
    
    return comments_concat

#### Helpers to construct test string (fname::class::method)
import ast
import difflib
from typing import List, Dict
def get_function_definitions(source: str) -> dict:
    """Extract function and method definitions from the source code - only for functions starting with test*"""
    tree = ast.parse(source)
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_name = node.name
            if function_name.startswith('test'):
                function_body = ast.get_source_segment(source, node)
                functions[function_name] = function_body
    
    return functions

def find_changed_functions(old_source: str, new_source: str) -> List[str]:
    """Find functions that have changed between two versions of a Python file."""
    old_funcs = get_function_definitions(old_source)
    new_funcs = get_function_definitions(new_source)
    
    changed_functions = []
    
    for func_name, new_body in new_funcs.items():
        old_body = old_funcs.get(func_name)
        if old_body is None:
            # Function is new
            changed_functions.append(func_name)
        elif old_body and old_body != new_body:
            # Function exists but has changed
            diff = list(difflib.unified_diff(old_body.splitlines(), new_body.splitlines()))
            if diff:
                changed_functions.append(func_name)
    
    return changed_functions

def extract_functions_and_methods(source: str) -> Dict[str, str]:
    """
    Extracts global function names and class methods from Python source code.
    
    :param source: A string containing Python source code.
    :return: A dictionary mapping function/method names to "global" or their class name.
    """
    tree = ast.parse(source)
    result = {}
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Global function
            
            result[node.name] = "global"
        elif isinstance(node, ast.ClassDef):
            # Class with methods
            class_name = node.name
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    # Method inside the class
                    result[sub_node.name] = class_name
    
    return result

def extract_test_scope(test_file_content, new_test_file_content, test_filename):
    # Extract string of the type fname::class::method
    func2class             = extract_functions_and_methods(new_test_file_content)
    contributing_functions = find_changed_functions(test_file_content, new_test_file_content)
    func2test_arr          = []
    if contributing_functions:
        for func in contributing_functions:
            scope = func2class.get(func, "")
            if scope == "":
                pass
            elif scope == "global":
                func2test_arr.append(f"{test_filename}::{func}")
            else: # class scope
                func2test_arr.append(f"{test_filename}::{scope}::{func}")

    return func2test_arr

#### Helpers to run the tests in docker
import docker
import sys
import re

# def read_dockerfile(commit_hash, dockerfile_path="Dockerfile"):
#     """Reads the Dockerfile, replaces the commit hash, and returns the modified content."""
#     with open(dockerfile_path, "r") as f:
#         content = f.read()

#     # Replace the commit hash dynamically
#     content = content.replace("RUN git checkout <commit_hash>", f"RUN git checkout {commit_hash}")

#     return content

def build_docker_image(client, dockerfile_path, commit_hash, image_tag="no_name_image"):
    """Builds a Docker image using the Python Docker SDK."""

    logger.info(f"[*] Building Docker image based on commit {commit_hash}")
    
    # Build the Docker image
    build_args = {"commit_hash": commit_hash}
    try:
        image, build_logs = client.images.build(path=".", 
                                                tag=image_tag, 
                                                dockerfile=dockerfile_path,
                                                buildargs=build_args,
                                                network_mode="host")


        logger.info(f"[+] Docker image '{image_tag}' built successfully.")
    # except docker.errors.BuildError as e:
    #     logger.info(f"[!] Build failed: {e}")
        # sys.exit(1)
    except docker.errors.APIError as e:
        logger.info(f"[!] Docker API error: {e}")
        sys.exit(1)


import tempfile
import os
import tarfile
import io

def run_test_in_container(client, image_tag, model_test_patch, tests_to_run, golden_code_patch=None):
    """Creates a container, applies the patch, runs the test, and returns the result."""

    # Create a temporary patch file
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as patch_file:
        patch_file.write(model_test_patch)
        patch_file_path = patch_file.name

    try:
        logger.info("[*] Creating container...")
        container = client.containers.create(
            image=image_tag,
            command="/bin/sh -c 'sleep infinity'",  # Keep the container running
            tty=True,  # Allocate a TTY for interactive use
            detach=True
        )

        container.start()
        logger.info(f"[+] Container {container.short_id} started.")

        # Create placeholder empty files for PRs that add new files
        handle_newly_added_files(model_test_patch, container)


        #### A) Test patch (Always)
        model_test_patch_fname = "test_patch.diff"
        patch_dest_path = f"/app/testbed/{model_test_patch_fname}"
        # Create a tar archive
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(patch_file_path, arcname=model_test_patch_fname)
        tar_stream.seek(0)
        # Copy the tar archive to the container
        container.put_archive("/app/testbed", tar_stream.getvalue())
        logger.info(f"[+] Patch file copied to {patch_dest_path}")

        

        # Apply the patch inside the container
        apply_patch_cmd = f"/bin/sh -c 'cd /app/testbed && git apply {model_test_patch_fname}'"
        exec_result = container.exec_run(apply_patch_cmd)

        if exec_result.exit_code != 0:
            logger.info(f"[!] Failed to apply patch: {exec_result.output.decode()}")
            return "ERROR", exec_result.output.decode(), ""

        logger.info("[+] Test patch applied successfully.")


        if golden_code_patch is not None:

            # Create a temporary patch file
            with tempfile.NamedTemporaryFile(delete=False, mode="w") as patch_file:
                patch_file.write(golden_code_patch)
                patch_file_path = patch_file.name

            # Create placeholder empty files for PRs that add new files
            handle_newly_added_files(golden_code_patch, container)

            #### B) Model patch (Only in post-PR code)
            golden_code_patch_fname = "golden_code_patch.diff"
            patch_dest_path = f"/app/testbed/{golden_code_patch_fname}"
            # Create a tar archive
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(patch_file_path, arcname=golden_code_patch_fname)
            tar_stream.seek(0)
            # Copy the tar archive to the container
            container.put_archive("/app/testbed", tar_stream.getvalue())
            logger.info(f"[+] Patch file copied to {patch_dest_path}")

            # Apply the patch inside the container
            apply_patch_cmd = f"/bin/sh -c 'cd /app/testbed && git apply {golden_code_patch_fname}'"
            exec_result = container.exec_run(apply_patch_cmd)
    
            if exec_result.exit_code != 0:
                logger.info(f"[!] Failed to apply patch: {exec_result.output.decode()}")
                return "ERROR", exec_result.exit_code
    
            logger.info("[+] Code patch applied successfully.")

        # Run the test command
        coverage_report_separator = "COVERAGE_REPORT_STARTING_HERE"
        test_command = (
            "/bin/sh -c 'cd /app/testbed && "
            f"coverage run --branch -m pytest -rA -vv -o console_output_style=classic --tb=short {" ".join(tests_to_run)} ; " # Here we use ";" instead of "&&" so that the next command runs even if the test fails
            "coverage report -m > coverage_report.txt && "
            f"echo '{coverage_report_separator}' && "
            "cat coverage_report.txt'"
        )
        exec_result = container.exec_run(test_command, stdout=True, stderr=True)
        stdout_output_all = exec_result.output.decode()
        try: # TODO: fix, find a better way to handle the "test-not-ran" error
            stdout, coverage_report = stdout_output_all.split(coverage_report_separator)
        except:
            logger.info("Internal error: docker command failed with: %s" % stdout_output_all)
        logger.info("[+] Test command executed.")

        # Determine PASS/FAIL from output
        if "= FAILURES =" in stdout or "= ERRORS =" in stdout: # if at least one test failed, we consider it a failure
            test_result = "FAIL" # because we may run one AI test with many developer tests
        else:
            test_result = "PASS"

        logger.info(f"[+] Test result: {test_result}")

        return test_result, stdout, coverage_report

    finally:
        # Cleanup
        #os.remove(patch_file_path)
        container.stop()
        container.remove()
        logger.info("[*] Container stopped and removed.")

def handle_newly_added_files(patch, container):
    """Finds and creates files only if their patch chunk starts with '@@ -0,0 +'."""
    new_files = set()
    current_file = None
    create_file = False

    for line in patch.splitlines():
        # Detect file changes
        match = re.match(r"^diff --git a/(.+) b/(.+)", line)
        if match:
            current_file = match.group(2)  # Get file path after 'b/'
            create_file = False  # Reset flag for each new file

        # Detect start of a new file
        if line.startswith("@@ -0,0 +"):
            create_file = True  # Mark this file for creation
        
        # If the file should be created, ensure it exists
        if current_file and create_file:
            new_files.add(current_file)
            print(f"Creating empty file in Docker: {current_file}")

            # Ensure parent directory exists inside the container
            parent_dir = os.path.dirname(current_file)
            if parent_dir:
                exec_result = container.exec_run(f"mkdir -p {parent_dir}", stdout=True, stderr=True)
                if exec_result.exit_code != 0:
                    print(f"Error creating directory: {exec_result.stderr.decode()}")

            # Create the empty file inside the Docker container
            exec_result = container.exec_run(f"touch {current_file}", stdout=True, stderr=True)
            if exec_result.exit_code != 0:
                print(f"Error creating file: {exec_result.stderr.decode()}")

            print(f"Created empty file in Docker: {current_file}")
            create_file = False  # Reset flag after creating

    return new_files
######### Helper to fetch file contents that changed in the PR

import requests
def fetch_pr_files(pr_number, repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/files"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 403 and "X-RateLimit-Reset" in response.headers:
        reset_time = int(response.headers["X-RateLimit-Reset"])
        wait_time = reset_time - int(time.time()) + 1
        #logger.info(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
        time.sleep(max(wait_time, 1))
        return fetch_pr_files(pr_number)
        
    response.raise_for_status()
    return response.json()



def add_comment_to_pr(owner, repo, pr_number, comment):
    """Add a comment to the PR"""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"body": comment}
    response = requests.post(url, json=data, headers=headers)
    return response.status_code, response.json()


### Helpers for test amplification
def get_missed_lines_and_decorate_patch(edited_files, code_content_before_arr, code_content_after_arr, golden_code_patch, offsets, coverage_report):
    # In code_after_labeled, we will label every line that is not covered with a 
    # comment: "# NOT COVERED"
    code_after_labeled_arr    = []
    modified_and_missed_lines = []
    
    for (edited_file, code_after, offset, ii) in zip(edited_files, code_content_after_arr, offsets, range(len(edited_files))):
        code_after_labeled = code_after.splitlines()
        
        this_file_coverage = [l for l in coverage_report.splitlines() if l.startswith(edited_file)]
        if not this_file_coverage:
            # If the file does not even appear in coverage.txt, it means
            # that it was not covered at all
            all_lines_in_file_missed = True
        else:
            all_lines_in_file_missed = False
            this_file_coverage = this_file_coverage[0]
            line_range_str = this_file_coverage.split('%')[-1]
            missed_lines, missed_branches = parse_missed_lines_and_branches(line_range_str)

        line_number_of_edited_lines = get_line_number_of_edited_lines(golden_code_patch)
        for (line, line_no, line_file) in line_number_of_edited_lines:
            if line_file == edited_file:
                # + offset because of fuzzy diff | -1 because it's 1-indexed
                line_no_adjusted = line_no+offset-1
                # logger.info(line)
                # logger.info(code_after.splitlines()[line_no_adjusted].strip())
                # logger.info("=========")
                assert line == code_after.splitlines()[line_no_adjusted].strip(), "Line mismatch"
                # Make it 1-indexed again
                if line_no_adjusted+1 in missed_lines or all_lines_in_file_missed:
                    modified_and_missed_lines.append(code_after.splitlines()[line_no_adjusted].strip()) # here it's 0-indexed
                    code_after_labeled[line_no_adjusted] = code_after_labeled[line_no_adjusted] + " ###NOT COVERED###"
    
        
        code_after_labeled_arr.append("\n".join(code_after_labeled)+"\n")
    
        
    golden_patch_labeled = ""
    for (c, c_labeled, fname) in zip(code_content_before_arr, code_after_labeled_arr, edited_files):
        
        golden_patch_labeled += unified_diff(c, 
                                        c_labeled, 
                                        fromfile=fname, 
                                        tofile=fname) + "\n"
        
    # if modified_and_missed_lines is empty, golden_patch_labeled is the same as golden_patch
    return modified_and_missed_lines, golden_patch_labeled


######################### The commented code is for the Github App version of the hook ####
# # GitHub API Base URL
# GITHUB_API_URL = "https://api.github.com"
# # Load GitHub App Credentials from Environment Variables
# GITHUB_APP_ID = 1131987 # os.getenv("GITHUB_APP_ID")
# GITHUB_PRIVATE_KEY_PATH = "/home/ubuntu/pr-tester-bot.2025-02-03.private-key.pem" #os.getenv("GITHUB_PRIVATE_KEY_PATH")

# def generate_github_jwt():
#     """Generate a JWT for authenticating as a GitHub App."""
#     with open(GITHUB_PRIVATE_KEY_PATH, "r") as key_file:
#         private_key = key_file.read()

#     payload = {
#         "iat": int(time.time()),  # Issued at
#         "exp": int(time.time()) + 600,  # Expires in 10 minutes
#         "iss": GITHUB_APP_ID  # GitHub App ID
#     }

#     return jwt.encode(payload, private_key, algorithm="RS256")

# def verify_signature(request):
#     """Verify the webhook signature using HMAC SHA-256."""
#     signature = request.headers.get("X-Hub-Signature-256")
#     if not signature:
#         return False

#     mac = hmac.new(GITHUB_WEBHOOK_SECRET.encode(), request.body, hashlib.sha256)
#     expected_signature = f"sha256={mac.hexdigest()}"

#     return hmac.compare_digest(expected_signature, signature)

# def get_installation_access_token(installation_id):
#     """Get an access token for the GitHub App installation."""
#     jwt_token = generate_github_jwt()
#     url = f"{GITHUB_API_URL}/app/installations/{installation_id}/access_tokens"

#     headers = {
#         "Authorization": f"Bearer {jwt_token}",
#         "Accept": "application/vnd.github.v3+json"
#     }

#     response = requests.post(url, headers=headers)
#     if response.status_code == 201:
#         return response.json()["token"]
#     else:
#         raise Exception(f"Failed to get installation token: {response.text}")


# @csrf_exempt
# def github_webhook(request):
#     """Handle GitHub App webhook events (PRs & Installations)."""
#     logger.info("Entered")
#     if request.method != "POST":
#         return JsonResponse({"message": "Invalid request"}, status=400)

#     if not verify_signature(request):
#         return JsonResponse({"message": "Invalid signature"}, status=403)

#     try:
#         payload = json.loads(request.body)

#         # Save the payload to the logs
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         filename = f"webhook_{timestamp}.json"
#         file_path = os.path.join(WEBHOOK_RAW_LOG_DIR, filename)
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(payload, f, indent=4)
#         logger.info(f"Webhook saved to {file_path}")  # Log the save action

#         # Extract installation ID (needed for authentication)
#         installation_id = payload.get("installation", {}).get("id")
#         if not installation_id:
#             return JsonResponse({"message": "No installation ID found"}, status=400)

#         logger.info("Found installation ID")

#         # Handle PR events
#         if "pull_request" in payload:
#             logger.info("In PR event")
#             pr_number      = payload["pull_request"]["number"]
#             repo_full_name = payload["repository"]["full_name"]

            
#             # Save the payload to the logs
#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             filename = f"{repo_full_name.replace('/', '__')}_{pr_number}_{timestamp}.json"
#             file_path = os.path.join(WEBHOOK_LOG_DIR, filename)
#             with open(file_path, "w", encoding="utf-8") as f:
#                 json.dump(payload, f, indent=4)
#             logger.info(f"Webhook saved to {file_path}")  # Log the save action


#             logger.info(pr_number)
#             # Get a dynamic token for this specific repo installation
#             installation_token = get_installation_access_token(installation_id)

#             # For now, don't post comment yet
#             return JsonResponse({"message": "Webhook received"}, status=200)
        
#             ###################################################################
#             ### TODO: Code to query the model and run the tests goes here######
#             ###################################################################
            
#             # Post a bot comment on the PR
#             post_github_comment(repo_full_name, pr_number, installation_token)

#         # Handle installation events (when someone installs the app)
#         if payload.get("action") == "created" and "installation" in payload:
#             logger.info(f"GitHub App was installed on a new repository")

#     except json.JSONDecodeError:
#         return JsonResponse({"message": "Invalid JSON"}, status=400)

#     return JsonResponse({"message": "Webhook received"}, status=200)


# # def post_github_comment(repo_full_name, pr_number, token):
#     """Posts a comment on a GitHub PR using the App's authentication."""
#     url = f"{GITHUB_API_URL}/repos/{repo_full_name}/issues/{pr_number}/comments"

#     headers = {
#         "Authorization": f"token {token}",
#         "Accept": "application/vnd.github.v3+json",
#         "User-Agent": "MyWebhookBot"
#     }

#     data = {"body": "🤖 Hello! This is an automated bot comment triggered by a GitHub App webhook!"}

#     response = requests.post(url, headers=headers, json=data)

#     if response.status_code == 201:
#         logger.info(f"✅ Comment posted on PR #{pr_number}")
#         return JsonResponse({"message": "Webhook received"}, status=200)

#     else:
#         logger.info(f"❌ Failed to post comment: {response.status_code} - {response.text}")
#         return JsonResponse({"message": "Invalid JSON"}, status=400)
######################### The commented code is for the Github App version of the hook ####

def run_pynguin_in_container(client, image_tag, file_to_test, this_log_dir, golden_code_patch=None, 
                             use_llm_seeds=False, generated_test=None,
                             test_file_name=None, extra_flags=""):
    max_search_time = 60

    # Taken from TDD-Bench-Verified/pynguin_utils.py, see there for details (TODO: One place)
    # Remove the .py suffix
    module_path = file_to_test[:-3]
    # If it's an __init__.py, remove that part
    if module_path.endswith('__init__'):
        module_path = module_path[:-(len('__init__'))]
        if module_path.endswith('/'):
            module_path = module_path[:-1]
    # Convert slashes to dots
    module_name = module_path.replace('/', '.')


    
    try:

        logger.info("[*] Creating container...")
        container = client.containers.create(
            image=image_tag,
            command="/bin/sh -c 'sleep infinity'",  # Keep the container running
            tty=True,  # Allocate a TTY for interactive use
            detach=True
        )

        container.start()
        logger.info(f"[+] Container {container.short_id} started.")


        if use_llm_seeds:
            print(test_file_name)
            write_variable_to_container_file(container, generated_test, test_file_name)
            
            # Move my filterer to container
            with open('../TDD-Bench-Verified/llm_tests/keep_compatible_seed_statements.py', 'r') as f:
                filterer_content = f.read()
            write_variable_to_container_file(container, filterer_content, "/app/testbed/keep_compatible_seed_statements.py")
            res = container.exec_run(f"python3 keep_compatible_seed_statements.py {test_file_name} {test_file_name} {module_name}", environment={"PYTHONPATH":"/app/testbed"},
                               stderr=True, stdout=True)
            res = res.output.decode('utf-8').strip()

        # Create a temporary patch file
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as patch_file:
            patch_file.write(golden_code_patch)
            patch_file_path = patch_file.name

        # Create placeholder empty files for PRs that add new files
        handle_newly_added_files(golden_code_patch, container)

        #### B) Model patch (Only in post-PR code)
        golden_code_patch_fname = "golden_code_patch.diff"
        patch_dest_path = f"/app/testbed/{golden_code_patch_fname}"
        # Create a tar archive
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(patch_file_path, arcname=golden_code_patch_fname)
        tar_stream.seek(0)
        # Copy the tar archive to the container
        container.put_archive("/app/testbed", tar_stream.getvalue())
        logger.info(f"[+] Patch file copied to {patch_dest_path}")

        # Apply the patch inside the container
        apply_patch_cmd = f"/bin/sh -c 'cd /app/testbed && git apply {golden_code_patch_fname}'"
        exec_result = container.exec_run(apply_patch_cmd)
        if exec_result.exit_code != 0:
            logger.info(f"[!] Failed to apply patch: {exec_result.output.decode()}")
            return None, exec_result.exit_code
        logger.info("[+] Code patch applied successfully.")


        pynguin_command = (
            "timeout 150s pynguin --project-path /app/testbed "
            " --seed 42 "
            "--output-path /app/testbed/pynguin-tests "
            f"--maximum-search-time {max_search_time} "
            f"--module-name {module_name} {extra_flags}"
        )
        env = {"PYNGUIN_DANGER_AWARE": 1}
        print(pynguin_command)
        exec_result = container.exec_run(pynguin_command, environment=env, stdout=True, stderr=True)
        res = exec_result.output.decode('utf-8').strip()
        with open(os.path.join(this_log_dir, "pynguin_first_output.txt"), "w") as f:
            f.write(res)

        if "AttributeError: 'NoneType' object has no attribute 'startswith'" in res:
            return None, None # run test generation in the highest level module only (e.g., bugbug)

        if "RuntimeError: Bug in Pynguin!" in res:
            logger.info("'RuntimeError: Bug in Pynguin!' error")
            return None, None
        
        res_ls = container.exec_run("ls -A /app/testbed/pynguin-tests/", stdout=True, stderr=True)
        res_ls = res_ls.output.decode('utf-8').strip()
        print(res_ls)
        with open(os.path.join(this_log_dir, "ls_first_output.txt"), "w") as f:
            f.write(res_ls)
        if "No such file or directory" in res_ls:
            logger.info("Fatal: pynguin could not even start")
            return None, None

        if not res_ls.strip():
            # if not tests generated, try a different, simpler mutation strategy
            logger.info("Default assertion generation failed, trying simple one")
            extra_flags += " --assertion_generation SIMPLE "
            pynguin_command = (
                "timeout 150s pynguin --project-path /app/testbed "
                " --seed 42 "
                "--output-path /app/testbed/pynguin-tests "
                f"--maximum-search-time {max_search_time} "
                f"--module-name {module_name} {extra_flags}"
            )
            exec_result = container.exec_run(pynguin_command, environment=env, stdout=True, stderr=True)
            res = exec_result.output.decode('utf-8').strip()
            with open(os.path.join(this_log_dir, "pynguin_second_output.txt"), "w") as f:
                f.write(res)

            # If still not tests are generated, abort
            res_ls = container.exec_run("ls -A /app/testbed/pynguin-tests/", stdout=True, stderr=True)
            res_ls = res_ls.output.decode('utf-8').strip()
            print(res_ls)
            with open(os.path.join(this_log_dir, "ls_second_output.txt"), "w") as f:
                f.write(res_ls)

            if not res_ls.strip():
                logger.info("Test generation failed, skipping...")
                return None, None

        if use_llm_seeds:
            if not "Parsed testcases:" in res:
                logger.info("Could not seed from LLM successfully, skipping...")
                return None, None
            print("Successful  seeding!")
            logger.info("Successful seeding!")


        os.makedirs(this_log_dir, exist_ok=True)
        copy_folder_from_container(container, "/app/testbed/pynguin-tests", this_log_dir)

        # Check if at least one passing test was generated
        pynguin_test_fname = f"test_{module_name.replace('.', '_')}.py"
        with open(os.path.join(this_log_dir, "pynguin-tests", pynguin_test_fname), "r") as f:
            pynguin_test = f.read()
        if not "def test_" in pynguin_test:
            logger.info("No test generated for, skipping...")
            return None, None

        # Delete test_*_failing.py so only the passing tests remain
        pynguin_test_fname_failing = f"/app/testbed/test_{module_name.replace('.', '_')}_failing.py"
        container.exec_run(f"rm {pynguin_test_fname_failing}")
        


        # run the tests in the fixed code
        res_run = container.exec_run("pytest pynguin-tests/", environment={"PYTHONPATH":"/app/testbed"})
        res_run = res_run.output.decode('utf-8').strip()
        with open(os.path.join(this_log_dir, "pynguin_fixed_code_output.txt"), "w") as f:
            f.write(res_run)
        if "errors during collection" in res_run:
            logger.info("Running of the pynguin-generated tests failed")
            return None, None

        # If no passing tests were generated (e.g., only xfail or fail), skip
        n_passed_tests = extract_passed_tests(res_run)
        if n_passed_tests == 0:
            logger.info("No passing tests generated")
            return None, None

        failedTestsFixedCode = extract_failed_test_names(res_run)



        ### Undo the patch changes
        apply_patch_cmd = f"/bin/sh -c 'cd /app/testbed && git apply -R {golden_code_patch_fname}'"
        exec_result = container.exec_run(apply_patch_cmd)
        if exec_result.exit_code != 0:
            logger.info(f"[!] Failed to reverse patch: {exec_result.output.decode()}")
            return None, exec_result.exit_code
        logger.info("[+] Code patch reversed successfully.")

    
        # run the tests in the buggy code
        res_run = container.exec_run("pytest pynguin-tests/", environment={"PYTHONPATH":"/app/testbed"})
        res_run = res_run.output.decode('utf-8').strip()
        with open(os.path.join(this_log_dir, "pynguin_buggy_code_output.txt"), "w") as f:
            f.write(res_run)
        if "errors during collection" in res_run:
            logger.info("Running of the pynguin-generated tests failed")
            return None, None


        failedTestsBuggyCode = extract_failed_test_names(res_run)
        f2p_tests = list(set(failedTestsBuggyCode) - set(failedTestsFixedCode))
        if len(f2p_tests) > 0:
            f2p_tests = "\n".join(f2p_tests)
            logger.info("Yes Fail-to-pass tests: %s" % (f2p_tests))
            with open(os.path.join(this_log_dir, "failed_tests.txt"), "w") as f:
                f.write(f2p_tests)
        else:
            logger.info("No Fail-to-pass tests")

        
        return f2p_tests, pynguin_test

    finally:
        # Cleanup only in the second time, when we run in the buggy code. In the 
        # first time, we need the container because it contains the pynguin tests
        os.remove(patch_file_path)
        container.stop()
        container.remove()
        logger.info("[*] Finished")


# The two below come from TDD-Bench-Verified/pynguin_utils.py. TODO: Unify through refactoring
def copy_folder_from_container(container, container_folder_path, local_folder_path):
    """
    Copy a folder from inside the Docker container to a local folder.
    """
    bits, stat = container.get_archive(container_folder_path)
    file_like = io.BytesIO(b''.join(bits))
    
    with tarfile.open(fileobj=file_like) as tar:
        tar.extractall(path=local_folder_path)

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
