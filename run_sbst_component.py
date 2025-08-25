from __future__ import annotations

import docker
import json
import resource
import traceback
import argparse


from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import re
import ast
from cldk.analysis.python.treesitter import PythonSitter
import traceback

from tddbench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)
from tddbench.harness.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
)
from tddbench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
    get_env_configs_to_build
)
from tddbench.harness.grading import get_eval_report, get_logs_eval
from tddbench.harness.test_spec import make_test_spec, TestSpec
from tddbench.harness.utils import load_tddbench_dataset, str2bool
import tddbench.harness.constants as constants

from datetime import datetime
import os
import io
import tarfile
from pathlib import Path
import re
from datasets import load_dataset

from pynguin_utils import *
import sys
import pandas as pd

from utils import slice_golden_file

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm


def run_instance(instance, test_spec, log_dir, label, cli_args):
    mode  = cli_args.mode
    model = cli_args.model
    T     = cli_args.T
    generation_budget_seconds = int(cli_args.budget_seconds)
    rm_instance_image = cli_args.rm_instance_image
    if rm_instance_image == "Yes":
        rm_instance_image = True
    elif rm_instance_image == "No":
        rm_instance_image = False
    else:
        raise ValueError("Unsupported value for --rm_instance_image")

    deserialize = cli_args.deserialize
    if deserialize == "Yes":
        deserialize = True
    elif deserialize == "No":
        deserialize = False
    else:
        raise ValueError("Unsupported value for --deserialize")


    client = docker.from_env()

    container_label = instance['instance_id']
    this_log_dir = os.path.join(log_dir, instance['instance_id'])
    logger = setup_logger(container_label, Path("%s/log.log" % this_log_dir))

    try:
        this_container_name = test_spec.get_instance_container_name(container_label)
        try:
            container = client.containers.get(this_container_name) # if exists, use existing
        except: # else, create new
            container = build_container(test_spec, client, container_label, logger, False, force_rebuild=False) # return Container object
        container.start()

        logger.info("Starting %s in container %s" % (instance['instance_id'], container.id))
        print("Starting %s in container %s" % (instance['instance_id'], container.id))
        module_names = extract_modules_from_diff(instance['patch']) # e.g., ["lib.matplotlib.axis"]
        # if the patch changes more than one modules, we test them all
        for (iModule, module_name) in enumerate(module_names, start=1):
            print("Start testing of module %s (%d/%d)" % (module_name, iModule, len(module_names)))
        
            if instance['python_version'] in ['3.8', '3.9']:
                # install pynguin 0.17
                run_in_container(container, "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pip install pynguin==0.17.0'", logger)            
                
                # Pynguin 0.17 requires jinja>3.0.0 and sphinx requires jinja<3.1.0, so I need to pin 3.0.3, their last common ground
                if "sphinx" in instance['instance_id']:
                    run_in_container(container, "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pip install jinja2==3.0.3'", logger)            

            else:
                # install pynguin latest for 3.10
                run_in_container(container, "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pip install pynguin'", logger)

                # This patch is needed for 3.10 according to: https://github.com/se2p/pynguin/issues/81#issuecomment-2522619386
                pynguin_patch = get_pynguin_3_10_patch()
                write_variable_to_container_file(container, pynguin_patch, "/opt/miniconda3/envs/testbed/lib/python3.10/site-packages/pynguin/analyses/pynguin_patch.diff")
                output = run_in_container(container, "bash -c 'cd /opt/miniconda3/envs/testbed/lib/python3.10/site-packages/pynguin/analyses/ && patch -p1 < pynguin_patch.diff'", logger)

            if instance['python_version'] in ['3.9', '3.11']:
                # See https://github.com/se2p/pynguin/issues/36 for AttributeError: IN
                run_in_container(container, "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pip install bytecode==0.13'", logger)

            # apply patch
            write_variable_to_container_file(container, instance['patch'], "/testbed/mypatch.diff")
            res = run_in_container(container, "git apply mypatch.diff", logger)
            if "patch does not apply" in res:
                break

            if "django" in instance['instance_id']:
                env = {
                    "PYNGUIN_DANGER_AWARE": 1,
                    "DJANGO_SETTINGS_MODULE": "django.conf.global_settings"
                    }
            else:
                env = {"PYNGUIN_DANGER_AWARE": 1}


            if mode == "sbst_with_llm_seed":# and instance['instance_id'] not in ['sympy__sympy-13974', 'sympy__sympy-19346']: # debug
                issue_description = instance['problem_statement']
                module_code = '\n'.join(instance['golden_code_contents_sliced_long'])
                patch = instance['patch']
                prompt = build_prompt_for_seeding_pg(issue_description, patch, module_name, module_code)
                model_response = query_model(prompt, model=model, T=T)
                generated_test = "\n".join(model_response.splitlines()[1:-1])

                # move the folder pynguin-tests/ of the container to the folder `this_log_dir` (python variable name) of my local machine
                os.makedirs(this_log_dir, exist_ok=True)
                with open(os.path.join(this_log_dir, "model_response.txt"), "w") as f:
                    f.write(model_response)
                with open(os.path.join(this_log_dir, "generated_test.py"), "w") as f:
                    f.write(generated_test)
                with open(os.path.join(this_log_dir, "prompt.md"), "w") as f:
                    f.write(prompt)

                module_name_short = module_name.split('.')[-1]
                test_file_name = os.path.join("/testbed/pynguin_seed/", f"test_{module_name_short}.py")
                write_variable_to_container_file(container, generated_test, test_file_name)
                
                if deserialize:
                    # Deserialize using our algorithm
                    with open('deserialization/keep_compatible_seed_statements.py', 'r') as f:
                        filterer_content = f.read()
                    write_variable_to_container_file(container, filterer_content, "testbed/keep_compatible_seed_statements.py")
                    run_in_container(container, f"bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pip install astor'", logger, env={"PYTHONPATH":"/testbed"})

                    run_in_container(container, f"bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python keep_compatible_seed_statements.py {test_file_name} {test_file_name} {module_name}'", logger, env={"PYTHONPATH":"/testbed"})


                extra_flags = "--initial_population_seeding True --initial_population_data /testbed/pynguin_seed/ -v "
                remove_from_logs = ["Running tests on mutant", "INFO     Iteration:"] # to keep the logs a bit shorter
            else:
                extra_flags = ""
                remove_from_logs = None

            # Generate the tests
            pynguin_command = get_pynguin_command(module_name, generation_budget_seconds, extra_flags, instance['python_version'])
            res = run_in_container(container, pynguin_command, logger, env=env, remove_from_logs=remove_from_logs) # run pynguin
            
            # This indicates that the seed was rejected, in which case we just retry without seed
            if "223 │   _setup_initial_population_seeding" in res:
                logger.info("Seed rejected, trying without seed")
                extra_flags = ""
                remove_from_logs = None
                pynguin_command = get_pynguin_command(module_name, generation_budget_seconds, extra_flags, instance['python_version'])
                res = run_in_container(container, pynguin_command, logger, env=env, remove_from_logs=remove_from_logs) # run pynguin
            
            # For these cases you can't do anything, so just go to the next module/instance
            if "AssertionError: Control flow must have at least one exit node" in res or \
             "TypeError: unsupported operand type(s) for" in res or \
             "ValueError: must give size for empty Cycle" in res or \
             "AttributeError: partially initialized module" in res or \
             "NameError: name" in res or "ValueError: Node names" in res or \
             "AttributeError: 'classproperty'" in res or "SUT contains nothing we can test" in res:
                logger.info("Pynguin simply cannot run, skipping...")
                continue
            
            try_simple_assert = False
            if "module 'builtins' has no attribute 'generator'" in res:
                try_simple_assert = True # this is a problem with the assertions, try with simple one

            if ("RuntimeError: Bug in Pynguin!" in res) and not try_simple_assert: # the above leads to Bug In Pynguin error but can be solved
                logger.info("'RuntimeError: Bug in Pynguin!' error in %s" % instance['instance_id'])
                continue
            
            res_ls = run_in_container(container, "ls -A /testbed/pynguin-tests/", logger)
            if ("No such file or directory" in res_ls) and not try_simple_assert:
                logger.info("Fatal: pynguin could not even start for %s" % instance['instance_id'])
                continue

            if not res_ls.strip() or try_simple_assert:
                # if not tests generated, try a different, simpler mutation strategy
                logger.info("Default assertion generation failed for %s, trying simple one" % instance['instance_id'])
                extra_flags_to_use = extra_flags + "--assertion_generation SIMPLE "
                pynguin_command = get_pynguin_command(module_name, generation_budget_seconds, extra_flags_to_use, instance['python_version'])
                res = run_in_container(container, pynguin_command, logger, env=env, remove_from_logs=remove_from_logs)

                if "module 'builtins' has no attribute 'generator'" in res:
                    logger.info("Simple assertion generation also failed for %s, trying no assertions" % instance['instance_id'])
                    extra_flags_to_use = extra_flags + "--assertion_generation NONE "
                    pynguin_command = get_pynguin_command(module_name, generation_budget_seconds, extra_flags_to_use, instance['python_version'])
                    res = run_in_container(container, pynguin_command, logger, env=env, remove_from_logs=remove_from_logs)


                # logger.info("60 seconds timed out, trying 6 seconds now")
                # # sometimes 60 seconds is too much and the command gets timed-out. 
                # # in these cases, try with 6 seconds
                # fallback_budget_seconds = 6
                # pynguin_command = get_pynguin_command(module_name, fallback_budget_seconds, extra_flags, instance['python_version'])
                # res = run_in_container(container, pynguin_command, logger, env=env)

                # If still not tests are generated, abort
                res_ls = run_in_container(container, "ls -A /testbed/pynguin-tests/", logger)
                if not res_ls.strip():
                    logger.info("Test generation failed for %s, skipping..." % instance['instance_id'])
                    continue

            if mode=="sbst_with_llm_seed":
                if (not "collected test cases:" in res) and (not "Parsed testcases:" in res):
                    # Instead of skipping, we just consider the non-seeded generations
                    #print("Could not seed from LLM successfully, skipping...")
                    #logger.info("Could not seed from LLM successfully, skipping...")
                    #continue
                    print("Could not seed from LLM successfully...")
                    logger.info("Could not seed from LLM successfully...")
                else:
                    print("Successful seeding!")
                    logger.info("Successful seeding!")

            # Delete test_*_failing.py so only the passing tests remain
            pynguin_test_fname_failing = f"/testbed/pynguin-tests/test_{module_name.replace('.', '_')}_failing.py"
            run_in_container(container, f"rm {pynguin_test_fname_failing}", logger)

            os.makedirs(this_log_dir, exist_ok=True)
            copy_folder_from_container(container, "/testbed/pynguin-tests", this_log_dir)

            # Check if at least one passing test was generated
            pynguin_test_fname = f"test_{module_name.replace('.', '_')}.py"
            with open(os.path.join(this_log_dir, "pynguin-tests", pynguin_test_fname), "r") as f:
                pynguin_test = f.read()
            if not "def test_" in pynguin_test:
                logger.info("No test generated for %s, skipping..." % instance['instance_id'])
                continue
            
            if 'sympy' in instance['instance_id']:
                # TODO: this is dirty, but works for now
                pynguin_test = transform_sympy_core_aliased_imports(pynguin_test)
                pynguin_test = transform_sympy_simplify_aliased_imports(pynguin_test)
                pynguin_test = transform_sympy_sets_aliased_imports(pynguin_test)
                pynguin_test = transform_sympy_matrices_aliased_imports(pynguin_test)
                pynguin_test = transform_sympy_logic_aliased_imports(pynguin_test)

                write_variable_to_container_file(container, pynguin_test, f"/testbed/pynguin-tests/{pynguin_test_fname}")
            
                # Copy again to get the updated (from transform_sympy_core_aliased_imports) tests
                copy_folder_from_container(container, "/testbed/pynguin-tests", this_log_dir)

            # We need pytest at least 7.2 needed by Pynguin
            ensure_pytest_version(container, logger)

            # Run the tests in the fixed codebase first
            res_run = run_in_container(container, "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pytest --color=no pynguin-tests/'", logger, env={"PYTHONPATH":"/testbed"})
            if "errors during collection" in res_run:
                logger.info("Running of the pynguin-generated tests failed for %s" % instance['instance_id'])
                continue

            # If no passing tests were generated (e.g., only xfail or fail), skip
            n_passed_tests = extract_passed_tests(res_run)
            if n_passed_tests == 0:
                logger.info("No passing tests generated for %s" % instance['instance_id'])
                continue

            failedTestsPostPR = extract_failed_test_names(res_run) 


            # go back to buggy (pre-pr) version. TODO: `git apply -R <patch.diff>` instead
            #run_in_container(container, "git checkout .", logger)
            run_in_container(container, "git apply -R mypatch.diff", logger)



            # run the test
            res_run = run_in_container(container, "bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && pytest pynguin-tests/'", logger, env={"PYTHONPATH":"/testbed"})
            if "errors during collection" in res_run:
                logger.info("Running of the pynguin-generated tests failed for %s" % instance['instance_id'])
                continue

            failedTestsPrePR = extract_failed_test_names(res_run) 
            f2p_tests = list(set(failedTestsPrePR) - set(failedTestsPostPR))
            if len(f2p_tests) > 0:
                f2p_tests = "\n".join(f2p_tests)
                logger.info("Yes Fail-to-pass tests for %s: %s (module %s)" % (instance['instance_id'], f2p_tests, module_name))
            else:
                logger.info("No Fail-to-pass tests for %s (%s)" % (instance['instance_id'], module_name))
    
    except Exception as e:
        full_trace = traceback.format_exc()
        logger.info("Internal error:\n%s" % full_trace)
    finally:
        run_in_container(container, "git checkout .", logger)
        if rm_instance_image:
            cleanup_container(client, container, logger)
            remove_image(client, test_spec.instance_image_key, logger)

def run_instance_wrapper(args):
    instance, test_spec, log_dir, label, cli_args = args
    return run_instance(instance, test_spec, log_dir, label, cli_args)



if __name__ == "__main__":

    # Generate default label using timestamp
    default_label = f"run_pynguin_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run fail-to-pass test generation and evaluation for Pynguin.")
    parser.add_argument("--label", default=default_label, help=f"label for this run (default: {default_label})")
    parser.add_argument("--dataset", default='tdd', help="dataset to use, one of ''swt'' for SWTBench_Lite or ''tdd'' for TDD-Bench-Verified (default: ''tdd'')")
    parser.add_argument("--mode", default='sbst', help="'sbst',  'sbst_with_llm_seed', 'llm_with_sbst_seed' (default: 'sbst')")
    parser.add_argument("--budget_seconds", default=60, help="budget in seconds for Pynguin's generation (default: 60)")
    parser.add_argument("--max_workers", default=12, help="how many threads to use for building docker images (default: 12)")
    parser.add_argument("--rm_instance_image", default="No", help="delete images for more disk space. One of 'yes', 'no' (default 'no')")
    parser.add_argument("--deserialize", default="Yes", help="whether or not no apply our deserialization algorithm to the LLM seeds")
    parser.add_argument("--debug", default=0, help="if true, run only a subset of instances (to be used for local development only)")
    parser.add_argument("--model", help=f"LLM model to use. One of 'gpt-4o' (default), 'llama-3.3-70b-versatile', 'deepseek-r1-distill-llama-70b'", default='gpt-4o', type=str)
    parser.add_argument("--T", help=f"LLM temperature (default: 0.0). Not applicable to DeepSeek.", default=0.0, type=float)

    cli_args    = parser.parse_args()
    dataset     = cli_args.dataset # "swt" / "tdd"
    label       = cli_args.label
    max_workers = int(cli_args.max_workers)
    debug       = int(cli_args.debug)
    model       = cli_args.model

    assert model in ['gpt-4o', 'llama-3.3-70b-versatile', 'deepseek-r1-distill-llama-70b'], "model must be one of 'gpt-4o', 'llama-3.3-70b-versatile', 'deepseek-r1-distill-llama-70b'"

    if dataset == "swt":
        ds = load_tddbench_dataset(name="princeton-nlp/SWE-bench_Lite")
    else:
        ds = pd.read_pickle('tddbench_verified_processed.pickle')

    print(len(ds))

    # Get all the PyngBench instances
    with open('pyngbench_ids.txt', 'r') as f:
        pyngbench_ids = f.readlines()
    pyngbench_ids = [iid.strip() for iid in pyngbench_ids]

    # Get all the trivial instances
    with open('trivial_ids.txt', 'r') as f:
        trivial_ids = f.readlines()
    trivial_ids = [iid.strip() for iid in trivial_ids]

    d = []
    for idx, instance in ds.iterrows():

        # Only run SBST for the PyngBench instances
        if not instance['instance_id'] in pyngbench_ids:
            continue

        # Skip trivial instances
        if instance['instance_id'] in trivial_ids:
            continue

        repo = instance['repo']
        specs = constants.MAP_REPO_VERSION_TO_SPECS[repo]
        python_version = specs[instance['version']]['python']

        if python_version in ['3.8', '3.9', '3.10']:
            instance['python_version'] = python_version
            d.append(instance)  

    print(len(d))

    # [Only for my local development environment] run only 2 instances
    if debug:
        d = d[-1:]
        print(len(d))

    log_dir = f"logs_pynguin/{label}"
    os.makedirs(log_dir, exist_ok=True)

    client_global = docker.from_env()
    build_env_images(client_global, d, max_workers=max_workers)

    test_specs = list(map(make_test_spec, d))

    args_list = [(instance, test_spec, log_dir, label, cli_args) for instance, test_spec in zip(d, test_specs)]

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_instance_wrapper, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing instances"):
            try:
                future.result()
            except Exception:
                traceback.print_exc()

    print("Finished generating and evaluating tests. Now aggregating results...")

    with open("pyngbench_ids.txt") as f:
        ids = [line.strip() for line in f]

    f2p_ids = []
    for id_ in ids:
        result_file = os.path.join(log_dir, id_, "log.log")
        if os.path.exists(result_file):
            with open(result_file) as f:
                result_log = f.read()
            if f"Yes Fail-to-pass tests for {id_}" in result_log:
                f2p_ids.append(id_)


    print(f"The SBST Component generated a fail-to-pass test for {len(f2p_ids)}/{len(ids)} ({len(f2p_ids)/len(ids)*100:.2f}%) instances.")
    if f2p_ids:
        print("Here is the complete list:")
        print(" | ".join(f2p_ids))
    
    sbst_summary_file = os.path.join(log_dir, "f2p_ids.log")
    with open(sbst_summary_file, "w") as f:
        f.write("\n".join(f2p_ids))