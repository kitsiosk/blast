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

logger = logging.getLogger("myapp")
logger.debug("Entered webhook")

# Directory where webhook requests will be saved
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



PROMPT_COMBINATIONS_GEN = {
    "include_golden_code"        : [1, 1, 1, 1, 0],
    "include_pr_desc"            : [0, 1, 0, 0, 0],
    "include_predicted_test_file": [1, 0, 0, 0, 0],
    "sliced"                     : ["LongCorr", "LongCorr", "LongCorr", "No", "No"],
    "include_issue_comments"     : [0, 1, 1, 1, 0]
}

@csrf_exempt
def github_webhook(request):
    """Handle GitHub webhook events"""
    if request.method != 'POST':
        logger.info("Method is not POST")
        return HttpResponseForbidden("Invalid method")

    if not verify_signature(request):
        logger.info("Invalid signature")
        return HttpResponseForbidden("Invalid signature")
    
    payload = json.loads(request.body)
    # Save the payload to the logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(WEBHOOK_RAW_LOG_DIR, exist_ok=True)
    filename = f"webhook_{timestamp}.json"
    file_path = os.path.join(WEBHOOK_RAW_LOG_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    logger.info(f"Webhook saved to {file_path}")  # Log the save action

    
    event = request.headers.get('X-GitHub-Event')
    if event == "pull_request": 
        # Only trigger when PR opens or new revision is pushed (or if it is my repo)
        if (payload.get("action") in ["opened", "synchronize"] or payload["repository"]["owner"]["login"]=="kitsiosk") and not payload["pull_request"]["user"]["login"]=="dependabot[bot]":
            
            run_all      = True  # run all variations, regardless of success or not
            post_comment = True
            stop         = False # we stop when successful
            models       = ["gpt-4o", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]

            for model in models:
                iAttempt = 0

                while iAttempt<len(PROMPT_COMBINATIONS_GEN) and (not stop or run_all):
                    print("Starting combination %d with model %s" % (iAttempt, model))
                    try:
                        response, stop = run(payload, iAttempt=iAttempt, model=model,
                                            timestamp=timestamp, post_comment=post_comment)
                    except Exception as e:
                        err = traceback.format_exc()
                        print("Failed with error:\n%s" % err)

                    iAttempt +=1
                    if stop:
                        post_comment = False # we only post comment once
                    with open('results.csv', 'a') as f:
                        f.write("%s,%s,%s,%s\n" % (payload["number"], model, iAttempt, stop))
        

            # o3-mini (last resort)
            if (not stop or run_all):
                model = "o3-mini"
                try:
                    response, stop = run(payload, iAttempt=1, model=model,
                                            timestamp=timestamp, post_comment=post_comment)
                except Exception as e:
                    err = traceback.format_exc()
                    print("Failed with error:\n%s" % err)

                if stop:
                    post_comment = False # we only post comment once
                with open('results.csv', 'a') as f:
                    f.write("%s,%s,%s,%s\n" % (payload["number"], model, iAttempt, stop))

            if response:
                return response
            else:
                return JsonResponse({"status": "success"})

        else:
            logger.info("PR event, but not opening or revision of PR, or PR opened by dependabot, so skipping...")
            return JsonResponse({"status": "success"})
    else:
        logger.info("Non-PR event")
        return JsonResponse({"status": "success"})


def run(payload, dockerfile=None, 
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

    # Check if the PR is linked to a GH Issue
    has_linked_issue, linked_issue, issue_title, issue_description, issue_comments = check_if_has_linked_issue(pr_description, owner, repo)
    #issue_description = f"{issue_title}\n{issue_description}\n{issue_comments}" # concatenate title and description
    issue_description = f"{issue_title}\n{issue_description}"
    if has_linked_issue:
        logger.info("Linked issue: %d" % linked_issue)
    else:
        logger.info("No linked issue, or the linked issue is not labeled as Bug")
        raise ValueError("No linked issue %s, raising error to skip writing to results csv file" % pr_number)


    os.makedirs(WEBHOOK_LOG_DIR, exist_ok=True)
    this_instance_log_dir = os.path.join(WEBHOOK_LOG_DIR, instance_id+"_%s"%timestamp, "i%s"%iAttempt+"_%s"%model)
    os.makedirs(this_instance_log_dir, exist_ok=True)
    os.makedirs(os.path.join(this_instance_log_dir, "generation"))
    with open(os.path.join(WEBHOOK_LOG_DIR, instance_id+"_%s"%timestamp, "payload_%s_%s_%s.json" % (repo, pr_number, timestamp)), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    # # Add PR comments
    # pr_comments = get_pr_comments(owner, repo, pr_number)
    # print("Comments: " + pr_comments)
    # pr_description = pr_description + "\n" + pr_comments

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
    
    # Get golden code patch
    diffs = []
    for (fname, fcontent_before, fcontent_after) in zip(code_fname_arr, code_content_before_arr, code_content_after_arr):
        diff = unified_diff_with_function_context(fcontent_before, fcontent_after, fname)
        diffs.append(diff)
    golden_code_patch = "\n\n".join(diffs)+"\n\n"
    # Get golden test patch
    diffs = []
    for (fname, fcontent_before, fcontent_after) in zip(test_fname_arr, test_content_before_arr, test_content_after_arr):
        diff = unified_diff(fcontent_before, fcontent_after, fromfile=fname, tofile=fname)
        diffs.append(diff)
    golden_test_patch = "\n".join(diffs)+"\n"

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

    if test_fname_arr: # sometimes there are no tests
        test_content_after_sliced_arr = slice_golden_file(
            test_content_before_arr, 
            golden_test_patch,
            "",
            return_file="post",
            append_line_numbers=True
            )
    else:
        test_content_after_sliced_arr = test_content_before_arr.copy()



    # Build Docker image of the repo-under-test
    client = docker.from_env()
    if dockerfile is None: # if no mock dockerfile given
        if repo == "bugbug" and owner == "kitsiosk" and pr_number == 5:
            dockerfile = f"dockerfiles/Dockerfile_bugbug_old1" # for integration testing
        else:
            dockerfile = f"dockerfiles/Dockerfile_{repo}"
    build_docker_image(client, dockerfile, base_commit, image_tag=image_tag)

    # Create central datastructure containing all the PR/Issue data
    instance = {}
    instance["instance_id"]          = instance_id
    instance["patch"]                = golden_code_patch
    instance["golden_test_names"]    = test_fname_arr
    instance["golden_test_contents"] = test_content_after_arr
    instance["golden_test_contents_sliced"] = test_content_after_sliced_arr
    instance["problem_statement"]    = issue_description
    instance["hints_text"]           = ""
    instance["golden_code_names"]    = code_fname_arr
    instance["golden_code_contents"] = code_content_before_arr
    instance["golden_code_contents_sliced_long"] = code_content_before_sliced_arr
    instance["title"]                = pr_title
    instance["description"]          = pr_description
    instance["base_commit"]          = base_commit




    
    logger.info("=============== Test Generation Started ===============")
    generation_completed    = False

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


    # Build prompt
    use_sbst_seed = True
    if use_sbst_seed:
        sbst_seeds_dir = "/Users/konstantinos/local-desktop/Test Generation Project/github_bot_logs_paper"
        sbst_tests = get_sbst_tests(instance, sbst_seeds_dir)
        if sbst_tests == "":
            # Pynguin did not run successfully, skip
            return JsonResponse({"status": "success"}), False
        else:
            instance['sbst_test_file_content'] = sbst_tests
        
    include_issue_description = True
    include_golden_code       = PROMPT_COMBINATIONS_GEN["include_golden_code"][iAttempt]
    sliced                    = PROMPT_COMBINATIONS_GEN["sliced"][iAttempt]
    include_issue_comments    = PROMPT_COMBINATIONS_GEN["include_issue_comments"][iAttempt]
    include_pr_desc           = PROMPT_COMBINATIONS_GEN["include_pr_desc"][iAttempt]
    include_predicted_test_file = PROMPT_COMBINATIONS_GEN["include_predicted_test_file"][iAttempt]
    prompt = build_prompt(instance,
                        include_issue_description=include_issue_description,
                        include_golden_code=include_golden_code, 
                        sliced=sliced, 
                        include_issue_comments=include_issue_comments, 
                        include_pr_desc=include_pr_desc,
                        include_predicted_test_file=include_predicted_test_file,
                        use_sbst_seed=use_sbst_seed)

    if len(prompt)>=1048576: # gpt4o limit
        logger.info("Prompt exceeds limits, skipping...")
        raise ValueError("")
    
    with open(os.path.join(this_instance_log_dir, "generation", "prompt.txt"), "w") as f:
        f.write(prompt)


    if model_test_generation is None: # if not mock, query model
        # Query model
        T        = 0.0
        response = query_model(prompt, model=model, T=T)
        new_test = response.split("```python\n")[-1] # for deepseek
        new_test = new_test.split("```")[0]
        new_test = new_test.replace('```python', '')
        new_test = new_test.replace('```', '')
        new_test = adjust_function_indentation(new_test)

        with open(os.path.join(this_instance_log_dir, "generation", "raw_model_response.txt"), "w") as f:
            f.write(response)
    else:
        new_test = model_test_generation


    with open(os.path.join(this_instance_log_dir, "generation", "generated_test.txt"), "w") as f:
        f.write(new_test)

    # Append generated test to existing test file
    try:
        new_test_file_content = append_function(test_file_content, new_test, insert_in_class="NOCLASS")
    except SyntaxError as e: # invalid python code
        print(e)
        return JsonResponse({"status": "success"}), False
    # Construct test patch
    model_test_patch = unified_diff(test_file_content, new_test_file_content, fromfile=test_filename, tofile=test_filename)+"\n"

    try:
        test_to_run = extract_test_scope(test_file_content, new_test_file_content, test_filename)
    except SyntaxError as e: # invalid python code
        print(e)
        return JsonResponse({"status": "success"}), False

    #### Run test in pre-PR codebase
    test_result_before, stdout_before, coverage_report_before = run_test_in_container(client, image_tag, model_test_patch, test_to_run)
    with open(os.path.join(this_instance_log_dir, "generation", "before.txt"), "w") as f:
        f.write(stdout_before)
    with open(os.path.join(this_instance_log_dir, "generation", "coverage_report_before.txt"), "w") as f:
        f.write(coverage_report_before)
    with open(os.path.join(this_instance_log_dir, "generation", "new_test_file_content.py"), "w") as f:
        f.write("#%s\n%s" % (test_filename, new_test_file_content))

    #### Run test in post-PR codebase
    golden_code_patch = instance["patch"]
    test_result_after, stdout_after, coverage_report_after = run_test_in_container(client, image_tag, model_test_patch, test_to_run, golden_code_patch=golden_code_patch)
    with open(os.path.join(this_instance_log_dir, "generation", "after.txt"), "w") as f:
        f.write(stdout_after)
    with open(os.path.join(this_instance_log_dir, "generation", "coverage_report_after.txt"), "w") as f:
        f.write(coverage_report_after)

    isFail2Pass = (test_result_before == "FAIL") and (test_result_after=="PASS")

    if isFail2Pass:
        missed_lines, decorated_patch = get_missed_lines_and_decorate_patch(code_fname_arr, code_content_before_arr, code_content_after_arr_from_patch, golden_code_patch, offsets, coverage_report_after)
        decorated_patch_new_lines = []
        for ln in decorated_patch.splitlines():
            if "###NOT COVERED###" in ln:
                new_line = ln.replace("###NOT COVERED###", "")
            elif ln.startswith("+") and not ln.startswith("+++"):
                new_line = ln + "# ✅ Covered by above test"
            else:
                new_line = ln
            decorated_patch_new_lines.append(new_line)
        decorated_patch_new = "\n".join(decorated_patch_new_lines)

        # Calculate patch coverage
        modified_lines = [l[1:].strip() for l in golden_code_patch.splitlines() if l.startswith('+') and not l.startswith('+++')]
        n_modified = len(modified_lines)
        patch_coverage = (n_modified - len(missed_lines))/n_modified

        # Add comment to the PR
        comment = comment_template_generation % (new_test, 
                                                 test_filename,
                                                 decorated_patch_new, 
                                                 patch_coverage*100)
        if post_comment:
            status_code, response_data = add_comment_to_pr(owner, repo, pr_number, comment)
        else:
            status_code, response_data = 201, ""
            logger.info("Debugging: would add this comment to PR:\n%s\n" % comment)

        if status_code == 201:
            logger.info("Comment added successfully!")
        else:
            logger.info(f"Failed to add comment: {status_code}", response_data)
        
        generation_completed = True
    elif not isFail2Pass:
        logger.info("No Fail-to-Pass test generated")
        generation_completed = False

    logger.info("=============== Test Generation Finished ===============")

    # Whether to stop or try again with different prompt inputs
    return JsonResponse({"status": "success"}), generation_completed


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

        # Check if it's a Pull Request
        if "pull_request" in issue_data:
            return "PR", None, None, comments
        
        # Check for "Bug" type: in custom fields or labels. If it's not labeled as bug, we skip it
        is_bug = False

        # Check for custom field "Type" = "Bug" if such metadata exists
        if issue_data.get("type", {}).get("name", "").lower() == "bug":
            is_bug = True

        # Check if any label is named "Bug"
        labels = issue_data.get("labels", [])
        if any(label.get("name", "").lower() == "bug" for label in labels):
            is_bug = True

        # Only return issue data if it's marked as a Bug
        if is_bug:
            return "Issue", issue_data.get("title"), issue_data.get("body"), comments
        else:
            logger.info(f"Issue #{number} is not labeled as a bug.")
            return None, None, None, None

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

    # # Read the modified Dockerfile content
    # dockerfile_content = read_dockerfile(commit_hash, dockerfile_path)

    # # Write a temporary Dockerfile (this avoids modifying the original file)
    # temp_dockerfile = "Dockerfile.temp"
    # with open(temp_dockerfile, "w") as f:
    #     f.write(dockerfile_content)

    logger.info(f"[*] Building Docker image based on commit {commit_hash}")
    
    # Build the Docker image
    build_args = {"commit_hash": commit_hash}
    try:
        image, build_logs = client.images.build(path=".", 
                                                tag=image_tag, 
                                                dockerfile=dockerfile_path,
                                                buildargs=build_args,
                                                network_mode="host")

        # # Print build logs
        # for log in build_logs:
        #     if "stream" in log:
        #         print(log["stream"].strip())

        logger.info(f"[+] Docker image '{image_tag}' built successfully.")
    except docker.errors.BuildError as e:
        logger.info(f"[!] Build failed: {e}")
        sys.exit(1)
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
        os.remove(patch_file_path)
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