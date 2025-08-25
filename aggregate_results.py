import os
import argparse
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Aggregate results of SBST Component and LLM Component of BLAST.")
parser.add_argument("--label",  help=f"label for this run")
cli_args = parser.parse_args()
label    = cli_args.label


# Read f2p results from the SBST component
results_dir_sbst = f"logs_pynguin/{label}/f2p_ids.log"
with open(results_dir_sbst, "r") as f:
    f2p_ids_sbst = [line.strip() for line in f.readlines()]

# Read f2p results from the LLM component
results_dir_llm = f"logs/run_evaluation/{label}/preds_{label}.jsonl/"
# Loop through all subfolders
f2p_ids_llm = []
for instance_id in os.listdir(results_dir_llm):
    instance_results = os.path.join(results_dir_llm, instance_id, "report.json")
    if os.path.isfile(instance_results):
        with open(instance_results, "r") as f:
            report = json.load(f)
        if report.get(instance_id, {}).get("resolved", False):
            f2p_ids_llm.append(instance_id)

print(f"SBST component found {len(f2p_ids_sbst)} f2p instances, which can be found in {results_dir_sbst}.")
print(f"LLM component found {len(f2p_ids_llm)} f2p instances, which can be found in {results_dir_llm} (the default TDDBench logging; the f2p instances have 'resolved':True in report.json).")
print(f"BLAST (i.e., union of SBST and LLM) found {len(set(f2p_ids_sbst).union(set(f2p_ids_llm)))} f2p instances.")