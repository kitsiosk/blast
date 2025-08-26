# IMPORTANT: This code is duplicated from "swt-bench" repo. Find a more permanent solution.
import re
import subprocess
import os
import openai
import tokenize
from collections import defaultdict
from io import BytesIO
import ast
import difflib
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


def extract_edited_files(diff_content):
    """
    Extract the filenames of all edited files from a unified diff.

    Parameters:
        diff_content (str): The unified diff content as a string.

    Returns:
        list: A list of relative paths of the edited files.
    """
    # Regular expression to capture the file paths from `+++ b/` lines
    matches = re.findall(r'^\+\+\+ b/(.+)$', diff_content, re.MULTILINE)
    return matches

def get_golden_file(retrieved_funcs, func_in_golden_patch):
    """
        - func_in_golden_patch is the name of the golden file
        - retrieved_funcs is a string containing the code for all the BM25-retrieved functions
        This function extracts and returns the code of the golden file
    """
    t = retrieved_funcs.split(func_in_golden_patch)
    if len(t) == 1:
        return "" # func in golden patch was not retrieved
    else:
        return t[1][2:-9] # trim some garbage in the beginning and end
    
def add_line_numbers(code):
    """Input (String): 
            x = 1
            print(x)
        Output (String):
            1 x = 1
            2 print(x)
    """
    code_lines = code.splitlines()
    code_with_line_nums = []
    for (i,line) in enumerate(code_lines):
        code_with_line_nums.append(f"{i+1} {line}")
    return "\n".join(code_with_line_nums)


def build_prompt(
        row, 
        include_issue_description=False,
        include_golden_code=False, 
        sliced="No", 
        include_issue_comments=False, 
        include_pr_desc=False, 
        include_golden_test_code=False,
        test_code_sliced=False,
        include_uncovered_lines_by_dvlpr_test=False,
        isCoT_amplification=False,
        include_predicted_test_file=False,
        use_sbst_seed=False
        ):

    golden_patch = row['patch']
    cot_text = ""
    predicted_test_file_text = ""
    predicted_test_file_contents = ""

    if include_golden_test_code:
        # If we include the golden_test_code, we are talking about Test Amplification, where we give the 
        # developer (golden) test to the model and ask for a test that increases coverage

        test_names_with_code = ""
        test_filenames = row['golden_test_names']
        if test_code_sliced:
            test_code = row['golden_test_contents_sliced']
        else:
            test_code = row['golden_test_contents']
            print("Warning, using Test Amplification without slicing the test code, performance may be bad")

        for (fname, fcode) in zip(test_filenames, test_code):
            test_names_with_code += "File %s\n%s\n\n" % (fname, fcode)

        if include_uncovered_lines_by_dvlpr_test:
            golden_patch = row['patch_labeled']
            task = "The developer has also submitted some tests in the PR that fail before the <patch> is applied and pass after the <patch> is applied, hence validating that the <patch> resolves the <issue>. The these fail-to-pass tests are shown in the <developer_tests> brackets (only parts relevant to the PR are shown with their respective line numbers; lines added in the PR start with '+'). However, these tests do not cover all the added code; specifically, the <patch> lines that are not covered end with the comment ###NOT COVERED###. Your task is to **write an additional fail-to-pass test that covers at least some ###NOT COVERED### lines**. If a test function from the <developer_tests> can be modified to cover ###NOT COVERED### lines, feel free to do it, otherwise (e.g., not covered lines are in a different file) you can ignore the <developer_tests>. You must import any needed modules in your test function. "
            task2 = "<developer_tests>\n%s\n</developer_tests>\n\nGenerate another fail-to-pass test that covers lines of the new code (<patch>) that were not covered by the <developer_tests>. " % test_names_with_code
        else:
            task = "The developer has also submitted some tests in the PR that fail before the <patch> is applied and pass after the <patch> is applied, hence validating that the <patch> resolves the <issue>. The these fail-to-pass tests are shown in the <developer_tests> brackets (only parts relevant to the PR are shown with their respective line numbers; lines added in the PR start with '+'). However, these tests do not cover all the added code. Your task is to **write an additional fail-to-pass test that covers at least some of the lines missed by the <developer_tests>. You must import any needed modules in your test function. "
            task2 = "<developer_tests>\n%s\n</developer_tests>\n\nGenerate another fail-to-pass test that covers lines of the new code (<patch>) that were not covered by the <developer_tests>. " % test_names_with_code


        if isCoT_amplification:
            cot_text = "Think step-by-step to generate the test:\n1. Select one or more ###NOT COVERED### line(s) from <code>.\n2. If the line(s) you selected belongs to a file already tested by one of the <developer_tests>, modify the developer test to cover the ###NOT COVERED### line(s) \n3. If, on the other hand, the line(s) you selected 1. are not covered by any developer test, write a new test function to cover them.\n"
            task3 = ", without any explanation or any natural language in general."
        else:
            task3 = "The test function should be self-contained and to-the-point, containing only the necessary assertions to test the ###NOT COVERED### line(s)."
    else:
        task = "Your task is to write one test function that fails before the changes in the <patch> and passes after the changes in the <patch>, hence verifying that the <patch> resolves the <issue>. "
        task2 = "Generate one test function that checks whether the <patch> resolves the <issue>.\n"
        if include_predicted_test_file:
            predicted_test_file_text = "Your generated test function will then be manually inserted by us in the test file shown in the <test_file> brackets; you can use the contents in these brackets for motivation if needed. "
            predicted_test_file_contents = "<test_file>\n%s\n</test_file>\n\n" % row["predicted_test_file_content_sliced"]
            
            task3 = ", or at most you can include a decorator to parameterize the test inputs, if one is used by the a test in <test_file> from which you drew motivation (if any). The test function should be self-contained (e.g., no parameters unless a decorator is used to parameterize inputs) and to-the-point."
        
        elif use_sbst_seed:
            predicted_test_file_text = "As a hint, we provide you with a test file automatically generated by the Pynguin tool in the <test_file> brackets below. These tests cover the modules that changed in the PR and are guaranteed to be syntactically correct and pass in the post-PR code. As such, you can use them, if needed, to get a skeleton for setting up your own test. However, since these tests are automatically generated, it is possible that they are not related to the issue above, in which case feel free to ignore them. "
            
            nTests = 3 # only first 3 pynguin tests to the prompt
            test_code = row['sbst_test_file_content']
            test_code = keep_first_n_pynguin_tests(test_code, nTests)
            predicted_test_file_contents = "<test_file>\n%s\n</test_file>\n\n" % test_code
            
            task3 = "."

        else: # simple generation
            task3 = ". The test function should be self-contained and to-the-point, containing only the necessary assertions to verify that the issue is resolved. Generate your test with simplicity in mind, focusing only on the core functionality described in the issue and mocking mock APIs, databases, or other external services to make the test simpler."

    if include_issue_description:
        issue_text = row['problem_statement']
    else:
        issue_text = row['problem_statement'].split('\n')[0]

    if include_golden_code:
        # Add golden code contents
        # - whole file
        # - random part of file
        # - targeted part of file (through AST)
        
        # filenames of the files changed by the golden patch
        code_filenames = row['golden_code_names']
        if sliced == "Short":
            code = row['golden_code_contents_sliced']
            sliced_text = " (only parts relevant to the patch are shown with their respective line numbers)"
        elif sliced == "Long" or sliced=="LongCorr":
            code = row['golden_code_contents_sliced_long']
            sliced_text = " (only parts relevant to the patch are shown with their respective line numbers)"
        elif sliced == "No":
            code = row['golden_code_contents'] # whole code
            code = [add_line_numbers(x) for x in code]
            sliced_text = ""
        else:
            raise ValueError("Unrecongnized value for 'sliced': %s" % sliced)

        code_string = "This patch will be applied to the file(s) shown within the <code> brackets%s. "%sliced_text

        fnames_with_code = ""
        for (fname, fcode) in zip(code_filenames, code):
            fnames_with_code += "File %s\n%s\n\n" % (fname, fcode)
        code_string2 = "<code>\n%s\n</code>\n\n"%fnames_with_code
    else:
        code_string = ""
        code_string2 = ""

    if include_issue_comments:
        comments_string = "\nIssue comments (discussion):\n %s"%(row['hints_text'])
    else:
        comments_string = ""

    if include_pr_desc:
        pr_desc_string = ". The description of this Pull Request is shown in the <pr_description> brackets"
        pr_desc_string2 = "<pr_description>\nPR Title: %s\n%s\n</pr_description>\n\n"%(row['title'], row['description'])
    else:
        pr_desc_string = ""
        pr_desc_string2 = ""

    _, repo_name, _ = parse_instanceID_string(row['instance_id'])
    prompt = f"""The following text contains a user issue (in <issue> brackets) posted at the {repo_name} repository. A developer has submitted a Pull Request (PR) that resolves this issue{pr_desc_string}. Their modification is provided in the form of a unified diff format inside the <patch> brackets. {code_string}{task}{predicted_test_file_text}You must only return a raw test function and you must import anything you need INSIDE that test function. More details at the end of this text.
    
<issue>
{issue_text}{comments_string}
</issue>

<patch>
{golden_patch}
</patch>

{code_string2}{predicted_test_file_contents}{pr_desc_string2}{task2}{cot_text}Return only one test function at the default indentation level WITHOUT considering the integration to the test file, e.g., in a unittest.TestCase class because your raw test function will then be inserted in a file by us, either as a standalone function or as a method of an existing unittest.TestCase class, depending on the file conventions; Return only one test function and nothing else{task3} Import anything you need INSIDE that test function."""            

#     if include_predicted_test_file:
#         x1 = "in the <test_file>"
#     else:
#         x1 = "in a file"

#     prompt = f"""You are an experienced software tester working at the {repo_name} repository, where your main responsibility is writing regression tests.
# The <issue> brackets contain an issue posted by a user in your repository.
# The <pr> brackets contain the changes introduced in a recent Pull Request (PR) that resolves the <issue>.
# Your task as an experienced software tester is to write a REGRESSION TEST for this <issue>.
# A regression test is a test that:
# A) FAILS on the current version of the code (shown in the <code> brackets).
# B) PASSES after the <pr> is applied to the <code>.


# <issue>
# {issue_text}
# </issue>

# <pr>
# {golden_patch}
# </pr>

# {code_string2}

# {predicted_test_file_contents}

# Think step-by-step to generate a REGRESSION test, i.e., a test function that:
# A) FAILS when we run it in the current version of the <code>.
# B) PASSES when we run it after applying the <pr> to the <code>. 
# Note that the changes will be applied by us externally, you only have to provide one raw test function that satisfies A) and B).


# Return only one test function at the default indentation level WITHOUT considering the integration to 
# the test file, e.g., in a unittest.TestCase class because your raw test function will then be inserted
# in a file by us, either as a standalone function or as a method of an existing unittest.TestCase 
# class, depending on the file conventions; you must provide only one raw test function and must import 
# any needed module inside your test function. The test function should be self-contained and to-the-point, containing only the necessary assertions to verify that the test is a regression test.
# """

    return prompt


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
    elif model.startswith("qwen") or model.startswith("deepseek"):
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


####### Naming Convention + Git History Injection #######
from collections import Counter

# Function to run a shell command and return its output
def run_command(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None

# Function to get the last N commits that edit a file
def get_last_N_commits(file_path, repo_dir, N=10):
    command = f"git log -n {N} --pretty=format:'%H' -- {file_path}"
    commits = run_command(command, cwd=repo_dir)
    return commits.splitlines() if commits else []

# Function to get the files edited in a specific commit
def get_files_in_commit(commit_hash, repo_dir):
    command = f"git show --name-only --pretty=format:'' {commit_hash}"
    files = run_command(command, cwd=repo_dir)
    return files.splitlines() if files else []

# Function to find the most commonly co-edited file for each file in file_list
# starting from base_commit
def find_coedited_files(file_list, repo_dir, n_last_commits=10, n_files=3):

    common_files = []
    for file in file_list:
        # Get the last 10 commits for the current file
        commits = get_last_N_commits(file, repo_dir, n_last_commits)
        # Get all files edited in the same commits
        coedited_files = []
        for commit in commits:
            coedited_files.extend(get_files_in_commit(commit, repo_dir))

        # Remove the file itself from the list of coedited files
        coedited_files = [f for f in coedited_files if f != file]

        # Filter files starting with "test*"
        coedited_files = [f for f in coedited_files if is_test_file(f)]

        # Find the most common coedited file and its count
        if coedited_files:
            most_common_files = Counter(coedited_files).most_common(n_files)
            common_files.extend(most_common_files)

    return common_files

def is_test_file(filepath, test_folder=''):
    is_in_test_folder = False

    # If a predefined test folder is given, we check if the filepath contains it
    if test_folder:
        is_in_test_folder = (test_folder in filepath)
    else:
    # Otherwise, we want the file to be in a dir where at least one folder in the dir path starts with test
        parts = filepath.split('/')
        for part in parts[:-1]:
            if part.startswith('test'):
                is_in_test_folder = True
                break

    if is_in_test_folder and parts[-1].startswith('test') and parts[-1].endswith(".py"):
        return True
    else:
        return False


def inject_test(row, repo_dir, new_test):

    test_file_to_inject, test_content = find_file_to_inject(row, repo_dir)
    if test_file_to_inject is None:
        print("No suitable file found for %s, skipping" % row['instance_id'])
        return None, None, None

    new_test_file_content = append_function(test_content, new_test, insert_in_class="NOCLASS")

    return test_file_to_inject, test_content, new_test_file_content

def find_file_to_inject(row, repo_dir):
    base_commit  = row['base_commit']
    current_branch = run_command("git rev-parse --abbrev-ref HEAD", cwd=repo_dir)
    run_command(f"git checkout {base_commit}", cwd=repo_dir)

    try:
        edited_files = extract_edited_files(row['patch'])
        ### First search for the file "test_<x>.py" where "<x>.py" was changed by the golden patch
        for edited_file in edited_files:
            matching_test_files = [] # could be more than 1 matching files in different dirs

            # ".../x.py" => ".../test_x.py"
            potential_test_file = 'test_' + edited_file.split('/')[-1]
            # Search in all test_ folders for that name
            for root, dirs, files in os.walk(repo_dir):
                if any(part.startswith("test") for part in root.split(os.sep)):
                    for file in files:
                        if file == potential_test_file:
                            test_file_to_inject = root + '/' + file
                            matching_test_files.append(test_file_to_inject)
            if matching_test_files: # stop in the first file for which we find (possibly >1) matching tests
                break

        if matching_test_files:
        ### Then, if the simple naming rule did not work, try Git History
            matching_test_files_relative = [y.replace(repo_dir+'/', '') for y in matching_test_files] # make relative
            test_file_to_inject = find_most_similar_matching_test_file(edited_file, matching_test_files_relative)
            test_file_to_inject = repo_dir + '/%s' % test_file_to_inject # make absolute again
        else:
            n_last_commits = 10
            coedited_files = find_coedited_files(edited_files, repo_dir, n_last_commits)
            if not coedited_files:
                # if we did not find in the last 10 commits, go to last 100 (only in pylint-dev__pylint-4661)
                coedited_files = find_coedited_files(edited_files, repo_dir, 100)
                if not coedited_files:
                    # inject to the first test file we find
                    first_random_test_file = get_first_test_file(repo_dir)
                    coedited_files = [(first_random_test_file, 1)] # name is just to fit in


            coedited_files = sorted(coedited_files, key=lambda x: -x[1]) # sort by # of co-edits


            test_file_to_inject = None
            for coedited_file in coedited_files: # coedited_file is a tuple (fname, #coedits)
                # we need to check if these files still exist because they 
                # come from a past commit
                if os.path.isfile(repo_dir + '/' + coedited_file[0]):
                    test_file_to_inject = repo_dir + '/' + coedited_file[0]
                    break # the first one we find that exists we keep it
            
            if not test_file_to_inject: # if none of the coedited files exist anymore, select the first file you find again
                first_random_test_file = get_first_test_file(repo_dir)
                test_file_to_inject = repo_dir + '/' + first_random_test_file

        # Read the contents of the test file
        with open(test_file_to_inject, 'r', encoding='utf-8') as f:
            test_content = f.read()

    finally:
        run_command(f"git checkout {current_branch}", cwd=repo_dir)  # Reset to the original commit

    return test_file_to_inject, test_content

def get_first_test_file(root_dir: str) -> str | None:
    """
    Our last resort: find the first file in a subfolder where at least one component starts with 'test' 
    and the filename itself also starts with 'test'.
    
    :param root_dir: The root directory to search in.
    :return: The first matching file path relative to root_dir, or None if no such file exists.
    """

    # First search in folders starting with "test" for files starting with "test"
    for dirpath, _, filenames in os.walk(root_dir):
        if not any(part.startswith(".") for part in dirpath.split(os.sep)) and any(part.startswith("test") for part in dirpath.split(os.sep)):
            for filename in filenames:
                if filename.startswith("test"):
                    return os.path.relpath(os.path.join(dirpath, filename), root_dir)
    
    # If nothing is found, search only for files starting with "test" (e.g., requests repo in old commits)
    for dirpath, _, filenames in os.walk(root_dir):
        if not any(part.startswith(".") for part in dirpath.split(os.sep)):
            for filename in filenames:
                if filename.startswith("test"):
                    return os.path.relpath(os.path.join(dirpath, filename), root_dir)
    return None


def find_most_similar_matching_test_file(code_filepath, test_filepaths):
    """
    Find the test file whose path is most similar to the code path.

    Parameters:
        code_filepath (str): Path to the code file (e.g., "astropy/utils/misc.py").
        test_filepaths (list): List of test file paths (e.g., ["astropy/utils/tests/test_misc.py", "astropy/visualization/wcsaxes/tests/test_misc.py"]).

    Returns:
        str: The test file path most similar to the code path.
    """
    # Split the code filepath into components
    code_components = code_filepath.split(os.sep)

    def similarity(test_filepath):
        """Calculate similarity between the code path and a test file path."""
        test_components = test_filepath.split(os.sep)
        # Count the number of matching components from the start of the paths
        match_count = 0
        for code_comp, test_comp in zip(code_components, test_components):
            if code_comp == test_comp:
                match_count += 1
            else:
                break
        return match_count

    # Find the test file with the highest similarity
    most_similar_test = max(test_filepaths, key=similarity)
    return most_similar_test

####### LIBRO Injection #######
def run_libro_injection(repo_dir, commitID, new_test, i, libro_granularity):
    """
        1. Save the current HEAD commit
        2. Checkout to the input commitID, which is the commitID before the PR
        3. Find the best file to inject new_test based on LIBRO injection (file-level or class-level)
        4. Append the generated test and return the new contents of the test file
    """
    original_dir = os.getcwd()  # Save the original working directory
    
    # Change to repo_dir
    os.chdir(repo_dir)
    
    # Save the current HEAD state
    original_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_dir, text=True
    ).strip()
    
    try:

        # Checkout the specified commit
        subprocess.run(["git", "checkout", commitID], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if libro_granularity == "file":
            # File-level
            test_filename = get_best_file_to_inject(repo_dir, new_test) 
        else:
            # Class-level
            test_filename, test_class = get_best_file_and_class_to_inject(repo_dir, new_test)
        test_filename = test_filename[0]
        
        if not test_filename:
            print(f"No suitable test file found for new test at i={i}. Skipping.")
            #cleanup(original_branch, original_dir) # this is not needed because the 
            # cleanup from the "finally" block will be executed
            return None, None, None
        
        # Read the contents of the test file
        with open(test_filename, 'r', encoding='utf-8') as f:
            test_content = f.read()
            
        # Append the function to the test file contents
        if libro_granularity == "file":
            # File-level
            new_test_file_contents = append_function(test_content, new_test, insert_in_class="NOCLASS")
        else:
            # Class-level
            new_test_file_contents = append_function(test_content, new_test, insert_in_class=test_class)

        return test_filename, test_content, new_test_file_contents
    
    except Exception as e:
        print(f"The following exception occurred for i={i}: {e}. Skipping.")
        return None, None, None

    finally:
        cleanup(original_branch, original_dir)

# LIBRO has two modes: 1) file mode: return the best file to inject and 
# 2) class mode: return the best file and class name to inject
def extract_tokens_from_code(code):
    """
    Extracts all tokens from a Python code string using the tokenize module.
    Returns a set of token strings (ignores comments and whitespace).
    """
    tokens = set()
    try:
        for token in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline):
            if token.type in {tokenize.NAME, tokenize.NUMBER, tokenize.STRING}:
                tokens.add(token.string)
    except tokenize.TokenError:
        pass  # Handle incomplete input gracefully
    return tokens

def get_best_file_to_inject(repo_dir, gen_test):
    """
    Finds the Python test file in "repo_dir" that is most similar to "gen_test" based on tokens.
    Searches through all subfolders starting with "test*/" for .py files.
    """
    # Prepare tokens for the generated test
    gen_test_tokens = extract_tokens_from_code(gen_test)

    file_scores = defaultdict(float)

    # Walk through the repo directory, focusing on test*/ subfolders and .py files
    for root, dirs, files in os.walk(repo_dir):
        if any(part.startswith("test") for part in root.split(os.sep)) and not any(part.startswith(".") for part in root.split(os.sep)):
            for file in files:
                if file.startswith("test") and file.endswith(".py"):
    
                    file_path = os.path.join(root, file)
    
                    # Read the content of the Python file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except (OSError, UnicodeDecodeError):
                        continue  # Skip files that cannot be read or decoded
    
                    # Extract tokens and compute similarity
                    file_tokens = extract_tokens_from_code(file_content)
                        
                    if file_tokens:
                        intersection = len(gen_test_tokens & file_tokens)
                        similarity_score = intersection/len(gen_test_tokens)
                        file_scores[file_path] = similarity_score

    # Identify the most similar file
    if file_scores:
        best_file = max(file_scores, key=file_scores.get)
        return best_file, file_scores[best_file]
    else:
        return None, 0.0  # No suitable file found
    
# Helper for get_best_file_and_class_for_injection
def extract_classes_from_code(file_tokens, code):
    class_tokens = {}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_start = node.lineno - 1
                # If Python 3.8+, node should have an end_lineno
                if hasattr(node, 'end_lineno'):
                    class_end = node.end_lineno
                else:
                    # Fallback if end_lineno doesn't exist:
                    class_end = max((child.lineno for child in ast.walk(node) if hasattr(child, 'lineno')), default=node.lineno)

                # class_end here is 1-based index. We don't need to add +1 because slicing excludes the end index 
                # and we're using node.end_lineno directly which is inclusive.
                # So we do not subtract one from class_end because end_lineno is the *last* line number (1-based).
                # When slicing, we use [class_start:class_end], because class_end is 1-based and slicing excludes 
                # the end line. For a 1-based end line number, slicing up to that number will include it because the 
                # slice index is zero-based. Let's break it down:
                #   - If end_lineno = 10, that means line 10 (1-based).
                #   - Our split lines are zero-based: line 10 is at index 9.
                #   - slice: splitlines()[class_start:class_end]
                #     If class_end=10 (1-based), slicing up to index 10 will include index 9.
                
                class_code = "\n".join(code.splitlines()[class_start:class_end])
                class_tokens[node.name] = extract_tokens_from_code(class_code)

    except (SyntaxError, ValueError):
        pass
    return class_tokens


def get_best_file_and_class_to_inject(repo_dir, gen_test):
    """
    Finds the Python test class in "repo_dir" that is most similar to "gen_test" based on tokens.
    Searches through all subfolders starting with "test*/" for .py files.
    """
    # Prepare tokens for the generated test
    gen_test_tokens = extract_tokens_from_code(gen_test)

    class_scores = []  # Stores (filename, class_name, similarity_score)

    # Walk through the repo directory, focusing on test*/ subfolders and .py files
    for root, dirs, files in os.walk(repo_dir):
        if any(part.startswith("test") for part in root.split(os.sep)):
            for file in files:
                if file.startswith("test") and file.endswith(".py"):

                    file_path = os.path.join(root, file)

                    # Read the content of the Python file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except (OSError, UnicodeDecodeError):
                        continue  # Skip files that cannot be read or decoded

                    # Tokenize the entire file
                    file_tokens = extract_tokens_from_code(file_content)

                    # Extract classes and their tokens
                    classes = extract_classes_from_code(file_tokens, file_content)

                    if classes:
                        for class_name, class_token_set in classes.items():
                            if class_token_set:
                                intersection = len(gen_test_tokens & class_token_set)
                                union = len(gen_test_tokens)
                                similarity_score = intersection / union if union > 0 else 0
                                class_scores.append((file_path, class_name, similarity_score))
                    else:
                        # Treat the whole file as a pseudo-class if no classes exist
                        if file_tokens:
                            intersection = len(gen_test_tokens & file_tokens)
                            union = len(gen_test_tokens)
                            similarity_score = intersection / union if union > 0 else 0
                            class_scores.append((file_path, "NOCLASS", similarity_score))

    # Identify the most similar class
    if class_scores:
        best_file, best_class, best_score = max(class_scores, key=lambda x: x[2])
        return best_file, best_class
    else:
        return None, None  # No suitable class found
    



# If anything goes wrong, reset to the initial commit
def cleanup(original_branch, original_dir):
    subprocess.run(["git", "fetch"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", original_branch], check=True, 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Change back to the original directory
    os.chdir(original_dir) 
################################################################

################## Golden Injection ############################

def extract_function_signatures(code_string):
    """
    Extract the signatures of all functions from a string of Python code.

    Assumes functions have only positional arguments and no default arguments.

    Parameters:
        code_string (str): A string of Python code defining one or more functions.

    Returns:
        list: A list of function signatures as strings.
    """
    # Parse the code into an Abstract Syntax Tree (AST)
    tree = ast.parse(code_string)
    
    # Collect all function signatures
    function_signatures = []
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Extract function name
            func_name = node.name
            
            # Extract positional arguments
            args = [arg.arg for arg in node.args.args]
            
            # Construct the function signature
            args_str = ", ".join(args)
            signature = f"{func_name}({args_str})"
            
            # Append the signature to the list
            function_signatures.append(signature)
    
    return function_signatures


def add_self_argument(function_signature):
    """
    Adds 'self' as the first argument in a Python function signature.

    Args:
        function_signature (str): The function signature as a string.

    Returns:
        str: The updated function signature with 'self' as the first argument.
    """
    # Regex to match the function name and argument list
    pattern = re.compile(r'^(\w+\s*\()(.*)(\))$')
    match = pattern.match(function_signature)
    if match:
        function_name = match.group(1)         # The function name and opening parenthesis
        arguments     = match.group(2).strip() # The argument list
        closing_parenthesis = match.group(3)   # The closing parenthesis

        if arguments:
            # if there are already arguments
            if "self" not in arguments: 
                # and they do not already contain "self", add it.
                # This is necessary to avoid adding "self" two times
                updated_arguments = f"self, {arguments}"
            else:
                # if "self" is already there, don't change anything. Sometimes the LLM
                # will output the function with "self" as a param
                updated_arguments = arguments
        else:
            updated_arguments = "self"

        return f"{function_name}{updated_arguments}{closing_parenthesis}"
    return function_signature

def get_best_file_to_inject_golden(initial_content_arr, changed_content_arr, fname_arr, new_test):
    """The golden test patch may contain >1 edited test file. In that case,
    we seek the most similar (token-wise) to the generated test (new_test) and
    inject the new test there.
    """
    # Only consider files start with test*.py
    files_starting_with_test = [x for x in fname_arr if x.startswith('test')]
    
    r = {}
    for (initial_content, changed_content, fname) in zip(initial_content_arr, changed_content_arr, fname_arr):
        if files_starting_with_test and not fname in files_starting_with_test:
            # If there is at least one file starting with test*, we skip files not starting with test*
            continue
        changed_funcs_or_classes, success = find_changed_funcs_or_classes(initial_content, changed_content)
        # changed_funcs_or_classes: [("function"/"class", name, code)]
        if success:
            r[fname] = changed_funcs_or_classes

    # return the class/func with the highest token similarity with the 
    # generated test
    new_test_tokens = extract_tokens_from_code(new_test)
    max_similarity = 0
    max_similarity_file = None
    max_similarity_class_or_func = None
    for file, class_or_func_arr in r.items():
        for class_or_func in class_or_func_arr:
            contents = class_or_func[2]
            tkns = extract_tokens_from_code(contents)
            
            if tkns:
                intersection = len(new_test_tokens & tkns)
                similarity_score = intersection/len(new_test_tokens)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    max_similarity_class_or_func = class_or_func
                    max_similarity_file = file

    success = max_similarity_class_or_func is not None
    return max_similarity_class_or_func, max_similarity_file, success


def find_changed_funcs_or_classes(initial_content, changed_content):
    """
    We want to find changed funcs or classes between two versions of a test file
    to locate in which place we should insert the model-generated test:
    we apply the golden test patch to the golden test file and check for changed
    funcs/classes and insert our model-generated test patch there
    """
    success = True # for handling errors in the higher level
    def extract_top_level_defs(content):
        """Extract top-level function and class definitions from the Python content."""
        tree = ast.parse(content)
        definitions = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start_lineno = node.lineno
                end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else None
                name = node.name
                type_ = 'class' if isinstance(node, ast.ClassDef) else 'function'
                definitions[name] = (type_, start_lineno, end_lineno)
        return definitions

    def extract_code_block(content, start, end):
        """Extract the specific code block based on line numbers."""
        lines = content.splitlines()
        return '\n'.join(lines[start - 1:end])

    try:
        initial_defs = extract_top_level_defs(initial_content)
        changed_defs = extract_top_level_defs(changed_content)
    except SyntaxError as e:
        # If the file content is not compilable we cannot use the AST to extract top level defs.
        # In this case, skip the instance
        success = False
        return [], success

    modified_or_missing = []

    # For model only uncomment these
    for name, (type_, start, end) in initial_defs.items():
        if name not in changed_defs:
            # If the definition is missing in changed_content
            code = extract_code_block(initial_content, start, end)
            modified_or_missing.append((type_, name, code))
        else:
            # Check if the content has changed
            initial_block = extract_code_block(initial_content, start, end)
            changed_type, changed_start, changed_end = changed_defs[name]
            changed_block = extract_code_block(changed_content, changed_start, changed_end)

            if list(difflib.unified_diff(initial_block.splitlines(), changed_block.splitlines())):
                modified_or_missing.append((type_, name, initial_block))

    # # For golden_and_model uncomment these
    # for name, (type_, start, end) in changed_defs.items():
    #     if name not in initial_defs:
    #         # If the definition is missing in changed_content
    #         code = extract_code_block(changed_content, start, end)
    #         modified_or_missing.append((type_, name, code))
    #     else:
    #         # Check if the content has changed
    #         changed_block = extract_code_block(changed_content, start, end)
    #         initial_type, initial_start, initial_end = initial_defs[name]
    #         initial_block = extract_code_block(initial_content, initial_start, initial_end)

    #         if list(difflib.unified_diff(initial_block.splitlines(), changed_block.splitlines())):
    #             modified_or_missing.append((type_, name, changed_block))

    return modified_or_missing, success


def apply_patch(file_content_arr, patch):
    """
    Apply a patch to file_content using the equivalent of "git apply".

    Parameters:
        file_content (str): The original content of the file.
        patch (str): The patch content in unified diff format.

    Returns:
        str: The updated file content after applying the patch.

    IMPORTANT: For `git apply` (and hence, this function) to run, it must be called from
    within a git repository. Alternatively we would have to use `apply` which is less nice.
    """
    temp_dir = './git_sandbox/'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Important: in order for "git apply" to work, it needs to be inside of a .git repo
    # so we initialize one and delete the .git at the end
    res = subprocess.run(["git", "init", "."], check=True, cwd=temp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Extract all the names of the changed files and save them in a list.
    # Each element of the list is of the form "a/<file_path>"
    patch_lines = patch.splitlines()
    original_file_aprefix_arr = []
    for line in patch_lines:
        if line.startswith("--- "):
            original_file_aprefix = line.split(" ")[1]
            original_file_aprefix_arr.append(original_file_aprefix)
    
    # Files mentioned in the patch should be the same number as the ones
    # whose content is provided. We also assume that the i-th file in the 
    # first content corresponds to the i-th file in the second.
    assert len(original_file_aprefix_arr) == len(file_content_arr), patch

    file_path_arr = []
    for (original_file_aprefix, file_content) in zip(original_file_aprefix_arr, file_content_arr):
        # remove intermediate folders: "a/django/tests/x.py" => "a/x.py"
        file_path  = original_file_aprefix.replace('/', '_')

        file_path_arr.append(file_path)
        file_path_aprefix = 'a/' + file_path
        file_path_bprefix = 'b/' + file_path
        patch = patch.replace(original_file_aprefix, file_path_aprefix) # update patch accordingly
        original_file_bprefix = 'b/' + '/'.join(original_file_aprefix.split('/')[1:])
        patch = patch.replace(original_file_bprefix, file_path_bprefix)
        
        # Write the file content and patch content to their respective files
        with open(temp_dir + file_path, "w") as file:
            file.write(file_content)

    patch_path = "patch.diff"
    with open(temp_dir + patch_path, "w") as patch_file:
        patch_file.write(patch)

    # Apply the patch using git apply
    try:
        #subprocess.run(["git", "apply", patch_path], check=True)
        #with open(patch_path, "r") as file:
            # result = subprocess.run(
            #     ["patch", "-p1"],
            #     stdin=file,  # equivalent to "patch -p1 > patch.diff" but redirection does not work with subprocess
            #     check=False,  # Raises an exception if the command fails
            #     text=True, 
            #     capture_output=True,  # Captures stdout and stderr
            # )
        res = subprocess.run(["git", "apply", "--reject", patch_path], 
                             check=True, 
                             capture_output=True,
                             text=True,
                             cwd=temp_dir)
    except subprocess.CalledProcessError as e:
        os.remove(temp_dir + patch_path)
        for file_path in file_path_arr:
            os.remove(temp_dir + file_path)
            #pass
        subprocess.run(["rm", "-rf", ".git/"], cwd=temp_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        raise ValueError(f"Failed to apply patch: {e}")


    updated_content_all_files = []
    for file_path in file_path_arr:
        # Read the updated file content
        with open(temp_dir + file_path, "r") as file:
            updated_content = file.read()

        os.remove(temp_dir + file_path)
        updated_content_all_files.append(updated_content)

    os.remove(temp_dir + patch_path)
    subprocess.run(["rm", "-rf", ".git/"], cwd=temp_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return updated_content_all_files, res.stderr

##### Test File Slicing (and Short Code File Slicing)###########
# # We keep only the direct parent of the changed lines + the global statements
# def slice_files_short(file_before_patch_arr, filename_arr, patch, return_file="post"):
#     file_arr_after_patch, _ = apply_patch(file_before_patch_arr, patch)
#     # Assumption: file_arr contains the files in the order they appear in patch
#     patch_arr = patch.split('diff --git a/')[1:] # only relevant parts of the patch
#     all_files_sliced = []

#     if return_file=="post":
#         file_arr = file_arr_after_patch
#     else:
#         file_arr = file_before_patch_arr

#     for (this_patch, this_file, this_filename) in zip(patch_arr, file_arr, filename_arr):
#         if not this_filename.endswith('.py'): # only slice python files (e.g., pylint-dev__pylint-4661)
#             all_files_sliced.append(this_file)
#             continue
#         assert this_filename in this_patch, this_patch # assert above assumption
#         # First slice the file keeping only the direct parents of changed lines
#         this_file_sliced = slice_file_short(this_patch, this_file, return_file=return_file)

#         # Then add the global statements
#         this_file_added_global_stmts = extract_global_added_statements(this_patch, this_file, return_file=return_file)
        
#         this_file_sliced = this_file_added_global_stmts + "\n" +  this_file_sliced 
        
#         all_files_sliced.append(this_file_sliced)
#     return all_files_sliced


# def slice_file_short(patch: str, file_content: str, return_file="post") -> str:
#     """
#     Extracts the parents (functions, methods, classes, or global) of all added lines
#     in a unified diff patch and slices the file content to include only those parents.

#     Args:
#         patch (str): The unified diff patch.
#         file_content (str): The full code of the file after the patch has been applied.
#         return_file (str): 
#             - "pre" to return pre-patch version with "-" in front of removed lines
#             - "pre_lineno" to return pre-patch version with line number in front of all lines
#             - "post" to return post-patch version with "+" in front of added lines
#             - "post_lineno" to return post-patch version with line number in front of all lines
#     Returns:
#         str: The sliced file content containing only the parents of the added lines.
#     """
#     # Identify added lines in the patch
#     added_lines = []  # List of line numbers
#     current_line_number = None

#     assert return_file in ["pre", "pre_lineno", "post", "post_lineno"], "Invalid option for return_file"

#     if return_file=="post" or return_file=="post_lineno":
#         symbol="+"
#         opposite_symbol="-"
#         three_symbols="+++"
#     else: # "pre" or "pre_lineno"
#         symbol="-"
#         opposite_symbol="+"
#         three_symbols="---"

#     for line in patch.splitlines():
#         #print(line)
#         if line.startswith("@@"):
#             # Extract the starting line number for the new file (after the patch)
#             match = re.search(r'\+([0-9]+)', line)
#             if match:
#                 current_line_number = int(match.group(1)) - 1

#         elif current_line_number is not None:
#             if not line.startswith(opposite_symbol):
#                 current_line_number += 1

#             if line.startswith(symbol) and not line.startswith(three_symbols):
#                 added_lines.append(current_line_number)

#     # print(added_lines)

#     # Parse the file content using the AST
#     class CodeVisitor(ast.NodeVisitor):
#         def __init__(self, total_lines):
#             self.parent_map = {line: 0 for line in range(1, total_lines + 1)}
#             self.parent_ranges = []
#             self.current_parent = None

#         def visit_ClassDef(self, node):
#             previous_parent = self.current_parent
#             self.current_parent = node.lineno
#             for child in ast.iter_child_nodes(node):
#                 if hasattr(child, 'lineno'):
#                     self.parent_map[child.lineno] = node.lineno
#                     self.parent_ranges.append((node.lineno, node.end_lineno))
#             self.generic_visit(node)
#             self.current_parent = previous_parent

#         def visit_FunctionDef(self, node):
#             previous_parent = self.current_parent
#             self.current_parent = node.lineno
#             for child in ast.iter_child_nodes(node):
#                 if hasattr(child, 'lineno'):
#                     self.parent_map[child.lineno] = node.lineno
#                     self.parent_ranges.append((node.lineno, node.end_lineno))
#             self.generic_visit(node)
#             self.current_parent = previous_parent

#         def visit(self, node):
#             if hasattr(node, 'lineno') and self.current_parent is not None:
#                 self.parent_map[node.lineno] = self.current_parent
#             super().visit(node)

#     file_content_lines = file_content.splitlines()
#     # for (i,l) in enumerate(file_content_lines, start=1):
#     #     print("%d %s" % (i, l))
#     total_lines = len(file_content_lines)
#     tree = ast.parse(file_content)
#     visitor = CodeVisitor(total_lines)
#     visitor.visit(tree)

#     #print(visitor.parent_ranges)

#     # Identify the parents of the added lines
#     parent_ranges = []
#     for line_number in added_lines:
#         smallest_range = None
#         for start, end in visitor.parent_ranges:
#             if start <= line_number <= end:
#                 if smallest_range is None or (end - start) < (smallest_range[1] - smallest_range[0]):
#                     smallest_range = (start, end)
#         # isDef = file_content_lines[line_number-1].strip().startswith("def") or file_content_lines[line_number-1].strip().startswith("class")
#         # if isDef:
#         #     print(file_content_lines[line_number-1].strip())
#         # Remove whitespaces
#         if smallest_range and file_content_lines[line_number-1].strip():
#             parent_ranges.append(smallest_range)

#     # Deduplicate and sort the parent ranges
#     parent_ranges = sorted(set(parent_ranges))
#     #print(parent_ranges)

#     # Deduplicate and sort the parent ranges, removing contained ranges
#     parent_ranges = sorted(set(parent_ranges))
#     deduplicated_ranges = []
#     for start, end in parent_ranges:
#         if not any(ds <= start and de >= end for ds, de in deduplicated_ranges):
#             deduplicated_ranges.append((start, end))

#     # Extract the relevant code
#     code_lines = file_content.splitlines()
#     extracted_code = []
#     for start, end in deduplicated_ranges:
#         for i in range(start, end + 1):
#             line = code_lines[i - 1]
#             if (return_file=="post" or return_file=="pre") and i in added_lines:
#                 line = symbol + line
#             if (return_file=="post_lineno" or return_file=="pre_lineno"):
#                 extracted_code.append('%d %s' % (i, line))
#             else:
#                 extracted_code.append(line)

#     return '\n'.join(extracted_code)


# def extract_global_added_statements(patch: str, file_content: str, return_file: str = "post_lineno") -> str:
#     """
#     Extracts all added statements in the global scope from a unified diff patch.

#     Args:
#         patch (str): The unified diff patch.
#         file_content (str): The full code of the file after the patch has been applied.

#     Returns:
#         str: The extracted global statements that were added.
#     """
#     # Identify added lines in the patch
#     added_lines = []  # List of line numbers
#     current_line_number = None

#     assert return_file in ["pre", "pre_lineno", "post", "post_lineno"], "Invalid option for return_file"

#     if return_file=="post" or return_file=="post_lineno":
#         symbol="+"
#         opposite_symbol="-"
#         three_symbols="+++"
#     else: # "pre" or "pre_lineno"
#         symbol="-"
#         opposite_symbol="+"
#         three_symbols="---"

#     for line in patch.splitlines():
#         if line.startswith("@@"):
#             # Extract the starting line number for the new file (after the patch)
#             match = re.search(r'\+([0-9]+)', line)
#             if match:
#                 current_line_number = int(match.group(1)) - 1

#         elif current_line_number is not None:
#             if not line.startswith(opposite_symbol):
#                 current_line_number += 1

#             if line.startswith(symbol) and not line.startswith(three_symbols):
#                 added_lines.append(current_line_number)

#     # Parse the file content using the AST
#     tree = ast.parse(file_content)

#     #global_nodes = set()
#     imports_to_include = []
#     for node in tree.body:  # Direct children of the Module node are in the global scope
#         if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr)):  # Global-level import constructs
#             imports_to_include.extend(range(node.lineno, getattr(node, 'end_lineno', node.lineno) + 1))
#         # if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr)):  # Global-level constructs
#         #     global_nodes.update(range(node.lineno, getattr(node, 'end_lineno', node.lineno) + 1))
#     print(imports_to_include)
#     # Extract the added global statements
#     # added_global_lines = [line for line in added_lines if line in global_nodes]
#     code_lines = file_content.splitlines()

#     extracted_code = []

#     # Include all import statements, including multi-line imports
#     for lineno in sorted(set(imports_to_include)):
#         line = code_lines[lineno - 1]
#         #if lineno in added_lines:
#         if return_file in ["pre_lineno", "post_lineno"]:
#             line = "%d %s" % (lineno, line)
#         else:
#             line = symbol + line
#         extracted_code.append(line)

#     # # Include other added global statements
#     # print(added_global_lines)
#     # for i in added_global_lines:
#     #     if i not in imports_to_include and code_lines[i - 1].strip():
#     #         if return_file in ["pre_lineno", "post_lineno"]:
#     #             line = "%d %s" % (i, code_lines[i - 1])
#     #         else:
#     #             line = symbol + code_lines[i - 1]
#     #         extracted_code.append(line)

#     return '\n'.join(extracted_code)



################################################################

def adjust_function_indentation(function_code: str) -> str:
    """
    Adjusts the indentation of a Python function so that the function definition
    has no leading spaces, and the internal code indentation is adjusted accordingly.

    :param function_code: A string representing the Python function.
    :return: The adjusted function code as a string.

    # Example Usage
    function_code = \"\"\"
        def example_function(param):
            if param:
                print("Hello, world!")
            else:
                print("Goodbye, world!")
    \"\"\"

    adjusted_code = adjust_function_indentation(function_code)
    print(adjusted_code)
    """
    lines = function_code.splitlines()
    
    if not lines:
        return ""

    # Find the leading spaces of the first non-empty line
    first_non_empty_line = next(line for line in lines if line.strip())
    leading_spaces = len(first_non_empty_line) - len(first_non_empty_line.lstrip())

    # Adjust the indentation by removing the leading spaces
    adjusted_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            adjusted_lines.append(line[leading_spaces:])
        else:  # Empty line
            adjusted_lines.append("")

    return "\n".join(adjusted_lines)


def append_function(file_content: str, new_function: str, insert_in_class: str = "NOCLASS") -> str:
    """
    Append the function new_function to file_content. If insert_in_class is a class name, 
    insert new_function as a method of that class. Otherwise, insert new_function at the bottom
    of the file_content.

    This handles the indentation.
    """
    # Parse the content using the AST module
    tree = ast.parse(file_content)
    
    if insert_in_class != "NOCLASS":
        # Add the self argument - if not already exists
        new_function_signature = extract_function_signatures(new_function)
        for signature in new_function_signature: # we may have >1 added functions
            signature_with_self = add_self_argument(signature)
            new_function = new_function.replace(signature, signature_with_self)
            
        # Search for the specified class
        target_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == insert_in_class:
                target_class = node
                break
        
        if target_class is None:
            raise ValueError(f"Class '{insert_in_class}' not found in the file content!")

        # Find the indentation level of the class
        lines = file_content.splitlines()
        class_start_line = target_class.lineno - 1
        class_indentation = len(lines[class_start_line]) - len(lines[class_start_line].lstrip())

        # Locate the last method in the class
        last_method = None
        for body_node in target_class.body:
            if isinstance(body_node, ast.FunctionDef):
                last_method = body_node

        if last_method:
            # Insert after the last method
            last_line = last_method.end_lineno  # Use end_lineno for accurate insertion
        else:
            # If no methods, insert at the end of the class
            last_line = target_class.end_lineno

        # Format the new function with the correct indentation
        indented_new_function = "\n".join(
            " " * (class_indentation + 4) + line if line.strip() else "" for line in new_function.splitlines()
        )
        
        # Insert the new function
        updated_lines = lines[:last_line] + [indented_new_function] + lines[last_line:]
        return "\n".join(updated_lines)

    else:
        # Default behavior: Append to the bottom of the file
        top_level_items = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                top_level_items.append(node)

        if not top_level_items:
            raise ValueError("No top-level classes or functions found in the file content!")

        # Find the last top-level item
        last_item = top_level_items[-1]

        if isinstance(last_item, ast.ClassDef): # last function was class
            # Add the self argument - if not already exists
            new_function_signature = extract_function_signatures(new_function)
            for signature in new_function_signature: # we may have >1 added functions
                signature_with_self = add_self_argument(signature)
                new_function = new_function.replace(signature, signature_with_self)

            # Handle classes by finding the last method
            last_method = None
            for body_node in last_item.body:
                if isinstance(body_node, ast.FunctionDef):
                    last_method = body_node


            if not last_method:
                raise ValueError(f"No methods found in the class '{last_item.name}'!")

            # Determine indentation
            last_func_line = last_method.lineno - 1 # line of the last function def
            last_code_line = last_method.end_lineno # last code line of the above function def

        else: # last function was a function in the top-level, not inside a class
            # For top-level functions
            last_func_line = last_item.lineno - 1 # line of the last function def
            last_code_line = last_item.end_lineno # last code line of the above function def

        # Extract indentation
        lines = file_content.splitlines()
        last_func_line_content = lines[last_func_line]
        indentation = len(last_func_line_content) - len(last_func_line_content.lstrip())

        # Add the new function
        indented_new_function = "\n".join(
            " " * indentation + line if line.strip() else "" for line in new_function.splitlines()
        )

        #updated_content = file_content.rstrip() + "\n\n" + indented_new_function + "\n"
        updated_lines = lines[:last_code_line] + ["\n"+indented_new_function] + lines[last_code_line:]
        updated_content =  "\n".join(updated_lines)
        return updated_content
    
def unified_diff(string1, string2, fromfile="original", tofile="modified", context_lines=3):
    """
    Prints the unified diff format of two strings. This is needed to calculate the diff
    given the new test file contents, since the evaluation harness works with the diff.

    Parameters:
        string1 (str): The original string.
        string2 (str): The modified string.
        fromfile (str): Name of the "from" file in the diff output.
        tofile (str): Name of the "to" file in the diff output.
        context_lines (int): Number of lines before and after the changes. Default is 3.

    # Example usage
    original_text = \"\"\"line 1
    line 2
    line 3\"\"\"

    modified_text = \"\"\"line 1
    line 2 modified
    line 3
    line 4 added\"\"\"

    unified_diff(original_text, modified_text)

    """
    import difflib

    fromfile = "a/" + fromfile
    tofile   = "b/" + tofile
    
    # Split the strings into lines
    lines1 = string1.splitlines(keepends=True)
    lines2 = string2.splitlines(keepends=True)

    # Generate the unified diff
    diff = difflib.unified_diff(lines1, lines2, fromfile=fromfile, tofile=tofile, n=context_lines)

    git_header = f"diff --git {fromfile} {tofile}\n"


    # Print the diff
    return git_header + "".join(diff)

def unified_diff_with_function_context(string1, string2, fname="tempfile.py", context_lines=3):
    """
    Writes two input strings to temporary files and uses `git diff --no-index`
    to compute the diff, including function context. This is important when you feed a diff
    to a model.

    Parameters:
    - string1: Original file content.
    - string2: Modified file content.
    - fname: The filename to simulate in the diff output.
    - context_lines: The number of context lines to show in the diff.

    Returns:
    - A string containing the Git-formatted diff.
    """

    temp_dir = './tmp_diff/'

    # Ensure the temp directory exists
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    try:
        file_dir = "/".join(fname.split('/')[:-1])
        os.makedirs(os.path.join(temp_dir, file_dir))

        file1 = os.path.join(temp_dir, f"{fname}.oldfordiffonly")
        file2 = os.path.join(temp_dir, f"{fname}.newfordiffonly")

        # Write original content
        with open(file1, "w") as f:
            f.write(string1)

        # Write modified content
        with open(file2, "w") as f:
            f.write(string2)

        # Run `git diff --no-index`
        result = subprocess.run(
            ["git", "diff", "-p", f"-U{context_lines}", "--no-index", file1, file2],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        diff = result.stdout.strip()
        diff = diff.replace(temp_dir, '') # make paths relative to target repo again
        diff = diff.replace(f"{fname}.oldfordiffonly", fname) # >>
        diff = diff.replace(f"{fname}.newfordiffonly", fname) # >>
        diff_lines = diff.splitlines()
        diff_lines.pop(1) # this is the "index 09b...." line
        diff = "\n".join(diff_lines)
        return diff
    finally:
        subprocess.run(["rm", "-rf", temp_dir], check=True)



def parse_instanceID_string(instance_id):
    # instanceIDs are of the form "<owner>__<repo>-<pr_number>"
    owner = instance_id.split('__')[0]
    tmp = instance_id.split('__')[1].split('-')
    if len(tmp) == 2: # "django-1111"
        repo, pr_number = tmp
    else: # "scikit-learn-1111"
        repo = '-'.join(tmp[0:-1])
        pr_number = tmp[-1]
    return owner, repo, pr_number

# ===================== Slicing helpers ============
def slice_golden_file(golden_contents_before_arr, patch, issue_description, return_file="pre", append_line_numbers=False):
    golden_contents_after_arr, stderr = apply_patch(golden_contents_before_arr, patch)
    # Create an array where element i is the relevant patch for file i
    # Assumption: file arrays contain the files in the order they appear in the patch
    #  => true by construction, we assert this in a different place
    patch_arr = ["diff --git" + x for x in patch.split("diff --git")[1:]]
    sliced_code_arr = [] # sliced code for each changed file

    for (golden_contents_before, golden_contents_after, this_patch) in zip(golden_contents_before_arr, golden_contents_after_arr, patch_arr):
        
        # lists where each element is {line_text: function_it_belongs}
        line2func_before, line2func_after, removed_lines_list, added_lines_list = get_edited_functions(golden_contents_before, golden_contents_after, this_patch)
        if not line2func_before and not line2func_after:
            # probably not a Python file, which made ast to fail => don't slice
            sliced_code_arr.append(golden_contents_before)
            continue 
        
        # Get the functions from which lines were REMOVED
        edited_functions_before = [list(x.values())[0] for x in line2func_before]
        edited_lines_before     = [list(x.keys())[0] for x in line2func_before]
        # Get the functions to which lines were ADDED
        edited_functions_after  = [list(x.values())[0] for x in line2func_after]
        edited_lines_after      = [list(x.keys())[0] for x in line2func_after]

        # Add them together and we have the edited functions        
        #functions_to_keep = list(set(edited_functions_before + edited_functions_after))
        #functions_called_in_issue_desc = extract_python_function_calls(issue_description)

        #if return_file == "pre":
        mapping_before = map_functions_to_classes(golden_contents_before, edited_functions_before)
        #else: # "post"
        mapping_after = map_functions_to_classes(golden_contents_after, edited_functions_after)
        # TODO: by concatenating the mappings, we will have problems in the (very unlikely) scenario
        # where a method is moved from one class to another.
        mapping = mapping_before + mapping_after
        class2methods = {}
        for method2class in mapping:
            for (k, v) in method2class.items():
                class2methods[v] = class2methods.get(v, []) + [k]
        global_funcs = class2methods.pop('global', [])
        #global_funcs = []

        if return_file == "pre": # apply slicing to the file before the patch
            sliced_code = slice_python_code(golden_contents_before, global_funcs, class2methods, append_line_numbers=append_line_numbers, edited_lines=removed_lines_list)
        else:
            sliced_code = slice_python_code(golden_contents_after, global_funcs, class2methods, append_line_numbers=append_line_numbers, edited_lines=added_lines_list)

        sliced_code_arr.append(sliced_code)
    return sliced_code_arr
    

def get_edited_functions(code_before, code_after, diff):
    """
    Given:
      1) code: the contents of a .py file (string)
      2) diff: a patch in unified diff format (string)

    TODO: It only looks functions containing lines added in the updated version of the code
    so if a function is modified by only deleting lines it wouldn't get returned

    Returns:
      A list of dicts, each with the form:
        {added_line: high_level_function}
      where 'added_line' is the exact text of the line added by the patch,
      and 'high_level_function' is the *highest-level* function to which
      this line belongs (or "global" if it is not inside any function).
    """

    # -------------------------------------------------------------------------
    # 1) Parse the diff to identify which lines were added and their line
    #    numbers in the updated file.
    # -------------------------------------------------------------------------
    added_lines_info = []
    removed_lines_info = []

    hunk_header_regex = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
    diff_lines = diff.splitlines()
    i = 0

    while i < len(diff_lines):
        line = diff_lines[i]
        match = hunk_header_regex.match(line)
        if match:
            # Extract the start lines for the old and new files
            old_start = int(match.group(1))
            new_start = int(match.group(2))

            # Initialize the counters for line numbers
            current_line_original = old_start - 1
            current_line_updated = new_start - 1
            i += 1

            # Process the lines in this diff hunk
            while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
                patch_line = diff_lines[i]
                
                # Lines that begin with '+' but not "+++" are added lines
                if patch_line.startswith('+') and not patch_line.startswith('+++'):
                    current_line_updated += 1
                    # We only track "new" line numbers for added lines
                    added_text = patch_line[1:]  # remove leading '+'
                    added_lines_info.append((current_line_updated, added_text))
                
                # Lines that begin with '-' but not "---" are removed lines
                elif patch_line.startswith('-') and not patch_line.startswith('---'):
                    current_line_original += 1
                    # We only track "old" line numbers for removed lines
                    removed_text = patch_line[1:]  # remove leading '-'
                    removed_lines_info.append((current_line_original, removed_text))

                else:
                    # Unchanged lines (or lines like '---'/'+++') appear in both old & new files
                    # So increment both counters
                    current_line_original += 1
                    current_line_updated += 1

                i += 1
        else:
            i += 1
    # Extract only the line numbers to return them
    removed_lines_list = [x[0] for x in removed_lines_info]
    added_lines_list = [x[0] for x in added_lines_info]

    # -------------------------------------------------------------------------
    # 2) Parse the updated code into an AST, and figure out for *every line*
    #    which top-level function it belongs to (or "global").
    # -------------------------------------------------------------------------

    # Parse code into an AST
    try:
        tree_after = ast.parse(code_after)
    except SyntaxError as e:
        # Maybe a non-python file was edited (e.g., .cfg), return empty array
        tree_after = None
    
    try:
        tree_before = ast.parse(code_before)
    except SyntaxError as e:
        # Maybe a non-python file was edited (e.g., .cfg), return empty array
        tree_before = None

    class ASTScopeAnalyzer(ast.NodeVisitor):
        """
        Visits each node in the AST.
        Maintains a stack of function names for nested FunctionDef nodes.
        (ClassDef does NOT push onto this stack, because classes are not function scopes.)
        For every node, we mark all lines in [lineno, end_lineno] as belonging to
        the *outermost* function in stack (stack[0]) or "global" if empty.
        """
        def __init__(self, line_scope_map):
            super().__init__()
            self.line_scope_map = line_scope_map
            self.func_stack = []  # stack of function names

        def _mark_lines(self, node):
            """Assign all lines from node.lineno to node.end_lineno the scope of stack[0] or 'global'."""
            start = getattr(node, 'lineno', None)
            end = getattr(node, 'end_lineno', None)
            if start is None or end is None:
                return  # Node might not have line info (e.g. Python <3.8 or synthetic nodes)
            scope_name = self.func_stack[0] if self.func_stack else "global"
            for ln in range(start, end+1):
                self.line_scope_map[ln] = scope_name

        def visit_FunctionDef(self, node):
            # Mark this function's entire span with the scope of the *currently* outermost function
            self._mark_lines(node)

            # Push this function's name onto the stack
            self.func_stack.append(node.name)

            # Visit the children (parameters, body, etc.)
            self.generic_visit(node)

            # Pop the function
            self.func_stack.pop()

        def visit_ClassDef(self, node):
            # Mark this function's entire span with the scope of the *currently* outermost function
            self._mark_lines(node)

            # Push this function's name onto the stack
            self.func_stack.append(node.name)

            # Visit the children (parameters, body, etc.)
            self.generic_visit(node)

            # Pop the function
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            # Same as FunctionDef, just for async defs
            self._mark_lines(node)

            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_ClassDef(self, node):
            # Mark the class lines with the current outer scope
            self._mark_lines(node)

            # IMPORTANT: we don't push or pop anything for classes,
            # because classes are not function scopes. 
            self.generic_visit(node)

        def visit_Module(self, node):
            # For the entire module, mark lines as "global" (if no function stack)
            self._mark_lines(node)
            self.generic_visit(node)

        def generic_visit(self, node):
            # For any node, we mark lines if available
            self._mark_lines(node)
            super().generic_visit(node)

    # Walk the AST to fill line_scope_map with top-level function scopes or 'global'
    # We'll store the scope of each line in a dictionary: line -> function_name or "global"
    line_scope_map_after = {}
    map_arr_after        = []
    if tree_after is not None:
        analyzer_after = ASTScopeAnalyzer(line_scope_map_after)
        analyzer_after.visit(tree_after)

        for (added_line_number, added_line_text) in added_lines_info:
            scope = line_scope_map_after.get(added_line_number, "global")
            map_arr_after.append({added_line_text: scope})

    line_scope_map_before = {}
    map_arr_before        = []
    if tree_before is not None:
        analyzer_before = ASTScopeAnalyzer(line_scope_map_before)
        analyzer_before.visit(tree_before)

        for (added_line_number, added_line_text) in removed_lines_info:
            scope = line_scope_map_before.get(added_line_number, "global")
            map_arr_before.append({added_line_text: scope})


    return map_arr_before, map_arr_after, removed_lines_list, added_lines_list

import ast

def map_functions_to_classes(code_str, function_list):
    """
    Given:
      1) code_str: a string containing valid Python (.py) source code
      2) function_list: a list of function names that we want to check
    
    Returns:
      A list of dicts: [{"function_name": scope}, ...] where `scope` is either
      the class name in which the function is defined, or "global" if
      it is defined at the top level.
      
      Example return value:
      [
          {"my_func": "MyClass"},
          {"other_func": "global"}
      ]
    """
    
    class ScopeTrackingVisitor(ast.NodeVisitor):
        """
        AST visitor that keeps track of the current class scope, so we know if
        we're inside a class when we encounter a function definition.
        """
        def __init__(self):
            self.current_class = None
            # function_map will map function_name -> class_name or "global"
            self.function_map = {}
        
        def visit_ClassDef(self, node):
            # Temporarily store the current class (if any), then set to this class
            saved_class = self.current_class
            self.current_class = node.name
            
            # Visit all nodes inside this class
            self.generic_visit(node)
            
            # Restore the previous class after exiting this class
            self.current_class = saved_class
        
        def visit_FunctionDef(self, node):
            if self.current_class:
                self.function_map[node.name] = self.current_class
            else:
                self.function_map[node.name] = "global"
            
            # Visit possible inner nodes (e.g., decorators)
            self.generic_visit(node)
    
    # Parse the source code into an AST
    tree = ast.parse(code_str)
    
    # Create and run our custom visitor
    visitor = ScopeTrackingVisitor()
    visitor.visit(tree)
    
    # For each function in our list, report the scope we found (or "global" if missing)
    results = []
    for func_name in function_list:
        scope = visitor.function_map.get(func_name, "global")
        results.append({func_name: scope})
    
    return results

#KK: recently changed this slice_python_code function completely, have to check how it works again
from typing import List, Dict, Set

def slice_python_code(
    source_code: str,
    global_funcs: List[str],
    class_methods: Dict[str, List[str]],
    append_line_numbers: bool = False,
    edited_lines: List[int] = [],
) -> str:
    """
    Return a 'sliced' version of the given Python source code, preserving
    original whitespace (and optionally annotating lines with original line numbers).

    The resulting code includes:
      1. All global variables (including import statements).
      2. Global functions whose names are in `global_funcs`.
      3. Classes (defined in the global scope) whose names are keys in `class_methods`.
         For each kept class:
           - Keep all class-level assignments (properties).
           - Keep the constructor (__init__) if defined.
           - Keep only the methods listed in class_methods[class_name].
           - Keep docstrings (which appear as Expr nodes with string constants).
    If `append_line_numbers` is True, each kept line starts with the line number of the original source.
    If it is False, then instead of line numbers, the edited lines start with a '+' (edited lines are in the edited_lines list)
    NOTE: This approach *skips entire lines* belonging to unwanted AST nodes.
          It does not attempt partial-line slicing.  
    """

    # Parse the code into an AST
    tree = ast.parse(source_code)

    # We'll create a "skip set" of line numbers to exclude from final output.
    # Python's AST nodes have lineno (start) and end_lineno (end), 1-based.
    # We'll later skip all lines in these ranges for nodes we *don't* want.
    lines_to_skip: Set[int] = set()

    # Convert the source into a list of lines for easy indexing
    source_lines = source_code.splitlines(keepends=True)  # keep original \n

    # --- Helper functions ---

    def is_docstring_expr(node: ast.AST) -> bool:
        """
        Returns True if 'node' is an Expr node containing a string constant
        (i.e., a docstring).
        """
        return (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, 'value', None), ast.Constant)
            and isinstance(getattr(node.value, 'value', None), str)
        )

    def mark_lines_skip(start: int, end: int) -> None:
        """Mark all lines [start, end] (inclusive) to be skipped."""
        for ln in range(start, end + 1):
            lines_to_skip.add(ln)

    def keep_top_level_node(node: ast.AST) -> bool:
        """Decide if a top-level node is to be kept."""
        # 1. Keep import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        # 2. Keep top-level assignments
        if isinstance(node, ast.Assign):
            return True
        # 3. Keep top-level functions if name is in global_funcs
        if isinstance(node, ast.FunctionDef):
            return (node.name in global_funcs)
        # 4. Keep top-level classes if name is in class_methods
        if isinstance(node, ast.ClassDef):
            return (node.name in class_methods)
        # # 
        # if is_docstring_expr(node):
        #     return False
        # Otherwise, skip
        return False

    def keep_class_child(node: ast.AST, class_name: str) -> bool:
        """
        Decide if a node *inside* a class body is to be kept.
        """
        # Keep class-level assignments
        if isinstance(node, ast.Assign):
            return True
        # Keep docstrings
        if is_docstring_expr(node):
            return True
        # Keep methods if they match criteria
        if isinstance(node, ast.FunctionDef):
            if node.name == '__init__':
                return True
            # Or if the method is in the allowed list
            if node.name in class_methods[class_name]:
                return True
        # Otherwise, skip
        return False

    def mark_node(node: ast.AST, keep: bool, parent_class: str = None) -> None:
        """
        Recursively mark lines to keep or skip based on the 'keep' flag.
        
        If we skip, all lines of this node are marked as skipped.
        If we keep, we *may still skip children* if we are inside a class and
        the child isn't wanted.
        """
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            # Some nodes (e.g. interactive mode) may not have line info
            return

        # If we're skipping this node, skip all its lines
        if not keep:
            mark_lines_skip(node.lineno, node.end_lineno)
            return

        # If we're keeping this node but it's a ClassDef, we need to process children
        if isinstance(node, ast.ClassDef):
            # The node itself is kept, but we might skip unwanted child nodes
            for child in node.body:
                child_keep = keep_class_child(child, node.name)
                mark_node(child, child_keep, parent_class=node.name)

    # --- Main slicing logic ---

    # 1) Walk top-level nodes and mark them keep/skip
    for node in tree.body:
        keep_flag = keep_top_level_node(node)
        mark_node(node, keep_flag, parent_class=None)

    # 2) Rebuild final code, skipping lines we marked
    result_lines = []
    for i, original_line in enumerate(source_lines, start=1):
        if i not in lines_to_skip :#and original_line.strip():
            if append_line_numbers:
                # Strip trailing newline if any, append comment, then re-add newline
                stripped_line = original_line.rstrip('\n')
                annotated_line = f"{i} {stripped_line}\n"
                result_lines.append(annotated_line)
            else:
                if i in edited_lines:
                    annotated_line = f"+{original_line}"
                    result_lines.append(annotated_line)
                else:
                    annotated_line = f" {original_line}"
                    result_lines.append(annotated_line)

    res = "".join(result_lines)
    res_cln = filter_stray_decorators(res)
    if append_line_numbers: # Collapse multiple newlines starting with an integer to one
        res_cln = re.sub(r'(^\d+ \n)(\d+ \n)+', r'\1', res_cln, flags=re.MULTILINE)
    else: # Collapse multiple newlines to one
        res_cln = re.sub(r'(\n )+', r'\n ', res_cln) 
    return res_cln

### Dealing with decorators in the slice_python_code function
def is_decorator_start(line: str) -> bool:
    """
    Check if a line starts with optional digits/spaces followed by '@'.
    e.g. "300 @something", "   @something", "@something"
    """
    return bool(re.match(r'^\s*\d*\s*@', line))

def is_def_or_class_start(line: str) -> bool:
    """
    Check if a line starts with optional digits/spaces followed by 'def' or 'class'.
    e.g. "300 def foo():", " class Bar:", "def something():"
    """
    return bool(re.match(r'^\s*\d*\s*(?:def|class)\b', line))

def find_decorator_end(lines: list[str], start_index: int) -> int:
    """
    Given a list of lines and the index of a line that starts a decorator ('@'),
    return the last line index that belongs to this decorator block.

    A decorator can span multiple lines if parentheses are opened '(' and not yet closed.
    We'll count '(' and ')' across lines until balanced or until we run out of lines.
    """
    open_parens = 0
    end_index = start_index
    i = start_index

    while i < len(lines):
        line = lines[i]
        # Count parentheses
        for char in line:
            if char == '(':
                open_parens += 1
            elif char == ')':
                open_parens -= 1
        
        end_index = i
        i += 1
        
        # If parentheses are balanced (or never opened), stop.
        if open_parens == 0:
            break

    return end_index

def filter_stray_decorators(text: str) -> str:
    """
    1) Finds blocks of consecutive decorators (each block may be multi-line if parentheses).
    2) Keeps all those decorator blocks only if the next line afterward is 'def' or 'class'
       (with optional digits/spaces).
    3) Otherwise, discards them. Any non-decorator lines are kept automatically.
    """
    lines = text.splitlines()
    kept = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # If this line starts a decorator
        if is_decorator_start(line):
            # We'll collect all consecutive decorator blocks
            decorator_blocks = []

            # Keep going while the line starts with '@'
            while i < n and is_decorator_start(lines[i]):
                # Find the end of this one decorator block
                block_start = i
                block_end = find_decorator_end(lines, block_start)
                
                # Slice out that block
                block_lines = lines[block_start : block_end + 1]
                decorator_blocks.append(block_lines)
                
                # Move i to the line after the block
                i = block_end + 1

            # Now i is at a line that does NOT start with '@' or it's beyond the end.
            # We check if that line starts with 'def' or 'class'
            if i < n and is_def_or_class_start(lines[i]):
                # Keep *all* consecutive decorator blocks
                for block in decorator_blocks:
                    kept.extend(block)
            else:
                # Discard them all (do nothing)
                pass

        else:
            # Normal line => keep it
            kept.append(line)
            i += 1

    return "\n".join(kept)
### End - Dealing with decorators in the slice_python_code function


# We use this to extract functions called from the issue description
def extract_python_function_calls(text):
    """
    Return a list of Python function names invoked in the given string.
    For example, given the string:

        from astropy.modeling import models as m
        from astropy.modeling.separable import separability_matrix

        cm = m.Linear1D(10) & m.Linear1D(5)
        It's separability matrix as you might expect is a diagonal:

        ```python
        >>> separability_matrix(cm)
        array([[ True, False],
               [False,  True]])
        ```

    This function should return:
        ['separability_matrix', 'Linear1D', 'array']
    """

    # This regex will match zero or more dotted components followed by
    # the final function name, capturing only that final name.
    pattern = r'(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*)\s*\('

    # Find all matches (the capturing group returns the function name)
    raw_matches = re.findall(pattern, text)

    # We remove duplicates while preserving order
    # (a common Python "idiom" for preserving insertion order is just using a list check)
    unique_functions = []
    for fn in raw_matches:
        if fn not in unique_functions:
            unique_functions.append(fn)

    return unique_functions


############# For Computing Non-covered Lines in Test Amplification #############
def parse_missed_lines_and_branches(line_string):
    """
    Parses a string describing missed line ranges and branches, returning two outputs.

    Args:
        line_string (str): The input string describing missed lines and branches.

    Returns:
        tuple: A tuple containing:
            - A list of integers representing the missed lines.
            - A list of tuples representing missed branches as (x, y).
    """
    missed_lines = []
    missed_branches = []

    # Correct regex to handle both '-' and '->'
    pattern = r'(\d+)(?:\s*(-|->)\s*(\d+))?'

    # Find all matches
    matches = re.findall(pattern, line_string)

    for match in matches:
        start = int(match[0])
        if match[1] == '->':
            # Handle missed branches
            end = int(match[2])
            missed_branches.append((start, end))
        else:
            # Handle missed lines and ranges
            end = int(match[2]) if match[2] else start
            missed_lines.extend(range(start, end + 1))

    return missed_lines, missed_branches

def get_line_number_of_edited_lines(diff_string):
    """
    Parse a unified diff string to find all added lines, their line numbers,
    and the file being edited.

    Args:
        diff_string (str): The unified diff string.

    Returns:
        list: A list of tuples where each tuple contains:
              - The added line (str)
              - The line number in the updated file (int)
              - The file being edited (str)
    """
    added_lines = []
    current_file = None
    updated_line_number = None

    # Split the diff into lines
    lines = diff_string.splitlines()

    for line in lines:
        # Detect file being edited from "+++" lines
        if line.startswith("+++"):
            match = re.match(r"\+\+\+ b/(.+)", line)
            if match:
                current_file = match.group(1)
            continue

        # Match line number changes from @@ -old_start,old_length +new_start,new_length @@
        header_match = re.match(r"@@ -\d+,?\d* \+(\d+),?\d* @@", line)
        if header_match:
            updated_line_number = int(header_match.group(1))
            continue

        # Detect added lines (those starting with '+')
        if line.startswith('+') and not line.startswith('+++'):
            added_line = line[1:].strip()  # Strip the '+' and any surrounding whitespace
            if updated_line_number is not None and current_file is not None:
                added_lines.append((added_line, updated_line_number, current_file))
                updated_line_number += 1

        # Detect unchanged lines (those not starting with '-', '+', or '@')
        elif not line.startswith('-') and not line.startswith('@'):
            if updated_line_number is not None:
                updated_line_number += 1

    return added_lines

def extract_offsets_from_stderr(stderr: str):
    """
    Parses stderr messages from patch application and returns an offset array.
    Each element i in the array corresponds to the offset of the i-th file, 
    with an assumption that all hunks of a file have the same offset.

    Args:
        stderr (str): The stderr message from the patch application.

    Returns:
        list: A list of offsets, where each offset corresponds to a file. 
              If no offset is mentioned for a file, the value is 0.

    Raises:
        AssertionError: If multiple hunks in the same file have different offsets.
        AssertionError: If Hunk #1 is missing from a file but later hunks exist.

    Example:
        Input:
            stderr_text = \"\"\"
            Checking patch a_lib_matplotlib_axes__base.py...
            Hunk #1 succeeded at 3264 (offset 2 lines).
            Hunk #2 succeeded at 3647 (offset 2 lines).
            Checking patch a_lib_matplotlib_ticker.py...
            Checking patch a_lib_mpl_toolkits_mplot3d_axes3d.py...
            Applied patch a_lib_matplotlib_axes__base.py cleanly.
            Applied patch a_lib_matplotlib_ticker.py cleanly.
            Applied patch a_lib_mpl_toolkits_mplot3d_axes3d.py cleanly.
            \"\"\"
        
        Output:
            [2, 0, 0]
    """
    file_pattern = re.compile(r'Checking patch (.*?)\.\.\.')
    hunk_pattern = re.compile(r'Hunk #(\d+) succeeded at \d+ \(offset ([+-]?\d+) line(?:s?)\)')
    
    current_file = None
    file_offsets = {}
    first_hunk_seen = {}
    
    for line in stderr.splitlines():
        file_match = file_pattern.match(line)
        if file_match:
            current_file = file_match.group(1)
            file_offsets[current_file] = None  # Default to None to detect first offset
            first_hunk_seen[current_file] = False
        
        hunk_match = hunk_pattern.search(line)
        if hunk_match and current_file:
            hunk_number = int(hunk_match.group(1))
            offset = int(hunk_match.group(2))
            
            if hunk_number == 1:
                first_hunk_seen[current_file] = True
            else:
                assert first_hunk_seen[current_file], f"OffsetHunk #1 missing for {current_file}, but later hunks exist."
            
            if file_offsets[current_file] is None:
                file_offsets[current_file] = offset  # Set initial offset
            else:
                assert file_offsets[current_file] == offset, f"Mismatched offsets for {current_file}: {file_offsets[current_file]} vs {offset}"
    
    return [offset if offset is not None else 0 for offset in file_offsets.values()]

def get_missed_lines(report_path, d):
    modified_and_missed_lines_per_instance = {}
    modified_lines_per_instance = {}
    for instance_folder in sorted(os.listdir(report_path)):
        instance_folder_fullpath = report_path + instance_folder
        if instance_folder.startswith('.'): # '.DS_Store/'
            continue

        # TODO: Unknown error, fix it soon
        if instance_folder == 'matplotlib__matplotlib-26342':
            continue
        # TODO: issue with offset, we need a func to map stderr (offset per hunk) to offset per file
        if instance_folder == 'matplotlib__matplotlib-14623':
            continue
        if instance_folder in ['django__django-12821', 'pylint-dev__pylint-5136', 'pylint-dev__pylint-5730', 'sympy__sympy-14085']:
            continue # a whole file was deleted in the PR, so the patch does not apply as is


        row = d[d['instance_id'] == instance_folder]
        if len(row) == 0:
            #print("%s does not exist in dataset due to an error in mine_pr_desc_and_golden_files.ipynb, skipping" % instance_folder)
            continue
        code_before_arr = row['golden_code_contents'].values[0]
        code_names_arr = row['golden_code_names'].values[0]
        
        
        golden_patch_path = instance_folder_fullpath+'/patch.diff'
        with open(golden_patch_path) as f:
            golden_patch = f.read()

        modified_lines_per_instance[instance_folder] = [l[1:].strip() for l in golden_patch.splitlines() if l.startswith('+') and not l.startswith('+++')]
    
        edited_files = extract_edited_files(golden_patch)
        
        code_after_arr, stderr = apply_patch(code_before_arr, golden_patch)
        try:
            offsets = extract_offsets_from_stderr(stderr)
        except AssertionError as e:
            print("Different offsets in a single file for %s, skipping" % instance_folder)
            continue

        coverage_report_path = instance_folder_fullpath+'/test_coverage.txt'
        if os.path.isfile(coverage_report_path):
            with open(coverage_report_path, 'r') as f:
                coverage_report = f.readlines()
        else:
            print("%s does not have test_coverage.txt, skipping" % instance_folder)
            continue
    
        modified_and_missed_lines = []
        for (edited_file, code_after, offset, ii) in zip(edited_files, code_after_arr, offsets, range(len(edited_files))):
            code_after_labeled = code_after.splitlines()
            
            this_file_coverage = [l for l in coverage_report if l.startswith(edited_file)]
            if not this_file_coverage:
                # If the file does not even appear in coverage.txt, it means
                # that it was not covered at all
                all_lines_in_file_missed = True
            else:
                all_lines_in_file_missed = False
                this_file_coverage = this_file_coverage[0]
                line_range_str = this_file_coverage.split('%')[-1]
                missed_lines, missed_branches = parse_missed_lines_and_branches(line_range_str)

            # 3-tuple of the form (line, line_no, file)
            line_number_of_edited_lines = get_line_number_of_edited_lines(golden_patch)
            for (line, line_no, line_file) in line_number_of_edited_lines:
                if line_file == edited_file :
                    # + offset because of fuzzy diff | -1 because it's 1-indexed
                    line_no_adjusted = line_no+offset-1

                    assert line == code_after.splitlines()[line_no_adjusted].strip(), "Line mismatch"
    
                    # Make it 1-indexed again
                    if line_no_adjusted+1 in missed_lines or all_lines_in_file_missed:
                        modified_and_missed_lines.append(code_after.splitlines()[line_no_adjusted].strip()) # here it's 0-indexed
            
            
        modified_and_missed_lines_per_instance[instance_folder] = modified_and_missed_lines


    return modified_and_missed_lines_per_instance, modified_lines_per_instance



def isFail2Pass(raw_pred, report):
    added_test_func = raw_pred.split('def ')[1].split('(')[0]

    # Get result before golden patch 
    added_test_func_fullname = None
    for added_func in report['test_before_patch'].keys():
        if added_test_func in added_func:
            added_test_func_fullname = added_func
            break
    if added_test_func_fullname is None:
        return None # to skip above
    before = report['test_before_patch'][added_test_func_fullname]

    # Get result after golden patch 
    for added_func in report['test_after_patch'].keys():
        if added_test_func in added_func:
            added_test_func_fullname = added_func
            break
    if added_test_func_fullname is None:
        return None # to skip above
    after = report['test_after_patch'][added_test_func_fullname]

    if (before=="FAILED" or before=="ERROR") and after=="PASSED":
        return True
    else:
        return False

### Using predicted test file in test generation
def keep_first_N_defs(source_code: str, N: int = 3) -> str:
    """
    Takes Python source code as input and returns a sliced version
    where all import/global statements, global comments, and docstrings
    are retained, but only the first three global class/function definitions are kept,
    along with their comments, docstrings, and decorators.
    """
    tree = ast.parse(source_code)
    
    result_lines = []
    global_defs_count = 0
    source_lines = source_code.splitlines()
    kept_lines = set()
    
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept_lines.update(range(node.lineno - 1, node.end_lineno))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if global_defs_count < N:
                kept_lines.update(range(node.lineno - 1, node.end_lineno))
                global_defs_count += 1
                # Include docstring if present
                if hasattr(node, 'body') and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    kept_lines.update(range(node.body[0].lineno - 1, node.body[0].end_lineno))
                # Include decorators
                for decorator in node.decorator_list:
                    kept_lines.update(range(decorator.lineno - 1, node.lineno - 1))
        elif isinstance(node, ast.Assign):
            kept_lines.update(range(node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # Global docstrings
            kept_lines.update(range(node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # Global comments
            kept_lines.update(range(node.lineno - 1, node.end_lineno))
    
    result_lines = [source_lines[i] for i in sorted(kept_lines) if i < len(source_lines)]
    
    return "\n".join(result_lines)
    

def get_contents_of_test_file_to_inject(row, repo_dir):
    # instance_id       = row['instance_id']
    # _, repo_folder, _ = parse_instanceID_string(instance_id)
    # repo_dir          = repo_base + "/" + repo_folder

    test_filename, test_file_content = find_file_to_inject(row, repo_dir)
    if test_filename is None:
        print("No suitable file found for %s, skipping" % row['instance_id'])
        return "", "", ""
    else:
        test_file_content_sliced = keep_first_N_defs(test_file_content)
    
    return test_filename, test_file_content, test_file_content_sliced

def get_sbst_tests(row, sbst_results_folder):
    test_dir = os.path.join(sbst_results_folder, "pynguin_%s_0"%row['instance_id'], "pynguin-tests")
    if not os.path.isdir(test_dir):
        return ""
    for subdir, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".py") and not file.endswith("failing.py"):
                with open(os.path.join(subdir, file), 'r') as f:
                    content = f.read()
                if "def test_" in content:
                    return content
                else:
                    return ""
    return ""

def keep_first_n_pynguin_tests(test_content, n):
    # Split the content into lines
    lines = test_content.splitlines()

    # Collect all import statements
    imports = []
    test_cases = []
    current_test_case = []

    for line in lines:
        if line.startswith('import ') or line.startswith('from '):
            imports.append(line)
        elif line.startswith('def test_case_'):
            if current_test_case:
                test_cases.append(current_test_case)
            current_test_case = [line]
        elif current_test_case:
            current_test_case.append(line)

    # Add the last test case if there is one
    if current_test_case:
        test_cases.append(current_test_case)

    # Keep only the first n test cases
    kept_test_cases = test_cases[:n]

    # Flatten the list of kept test cases
    kept_lines = imports + [line for test_case in kept_test_cases for line in test_case]

    # Join the kept lines back into a single string
    return '\n'.join(kept_lines)

def extract_passed_tests(pytest_output):
    match = re.search(r"(\d+)\s+passed", pytest_output)
    return int(match.group(1)) if match else None
