import os
import re
import json
import wandb
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all

from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original


def build_run_tag(args) -> str:
    """
    Build a unique run identifier (run_tag) for this reasoning run.
    
    This tag is used to make prediction filenames unique, allowing multiple
    reasoner runs to execute in parallel without file collisions.
    
    Priority:
    1. If args.run_tag is explicitly provided, use it directly.
    2. Otherwise, derive from the retrieval checkpoint path (args.score_dict_path).
       - Extract the parent directory name (e.g., "webqsp_Dec04-07:08:00").
       - Sanitize it by replacing colons, spaces, and other problematic chars with "-".
    3. If no path is available, fall back to a timestamp-based tag.
    
    Returns:
        A sanitized string suitable for use in filenames.
    """
    if args.run_tag is not None:
        # Use explicit run_tag, but sanitize it for safety
        tag = args.run_tag
    elif args.score_dict_path is not None:
        # Derive from retrieval checkpoint path
        # Get parent directory name from path like "../retrieve/webqsp_Dec04-07:08:00/retrieval_result.pth"
        path = Path(args.score_dict_path)
        parent_name = path.parent.name
        tag = parent_name
    else:
        # Fallback to timestamp
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize the tag: replace colons, spaces, and other problematic characters with "-"
    tag = re.sub(r'[:\s/\\]+', '-', tag)
    # Remove any leading/trailing dashes and collapse multiple dashes
    tag = re.sub(r'-+', '-', tag).strip('-')
    
    return tag


def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print("*" * 50)
        return ckpt
    return []


def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1):

    print("=" * 50)
    print("=" * 50)
    print(f"Evaluating on subset: {subset}")

    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    run.log({f"results{postfix}/hit@1": hit1,
             f"results{postfix}/macro_f1": f1,
             f"results{postfix}/macro_precision": prec,
             f"results{postfix}/macro_recall": recall,
             f"results{postfix}/exact_match": em,
             f"results{postfix}/totally_wrong": tw,
             f"results{postfix}/micro_f1": mi_f1,
             f"results{postfix}/micro_precision": mi_prec,
             f"results{postfix}/micro_recall": mi_recall,
             f"results{postfix}/total_cnt": total_cnt,
             f"results{postfix}/no_ans_cnt": no_ans_cnt,
             f"results{postfix}/no_ans_ratio": no_ans_ratio,
             f"results{postfix}/hal_score": hal_score})  # score_h in the paper
    if stats is not None:
        for k, v in stats.items():
            run.log({f"stats{postfix}/{k}": v})

    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    run.log({f"results{postfix}/hit": hit})
    print("=" * 50)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    # parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    # Optional run tag for unique prediction filenames (allows parallel runs)
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Optional unique identifier for this run. If not provided, derived from "
                             "score_dict_path or timestamp. Used to create unique prediction filenames "
                             "for parallel execution safety.")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres

    # Build a unique run_tag for this run to allow parallel executions without file collisions.
    # This tag is incorporated into prediction filenames.
    run_tag = build_run_tag(args)
    print(f"Using run_tag: {run_tag}")

    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-{run_tag}"
    run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)

    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    # Create results directory
    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Prediction filename construction:
    # The run_tag is used to allow multiple reasoner runs in parallel without file collisions.
    # Prediction filenames follow the pattern:
    #   <prompt_mode>-<llm_mode>-<frequency_penalty>-thres_<thres>-<run_tag>-<split>-predictions[-resume].jsonl
    # 
    # Examples:
    #   scored_100-sys_icl_dc-0.16-thres_0.0-webqsp_Dec04-07-08-00-test-predictions.jsonl
    #   scored_100-sys_icl_dc-0.16-thres_0.0-my_custom_tag-test-predictions.jsonl
    base_name = f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{run_tag}-{split}-predictions"
    raw_pred_file_path = raw_pred_folder_path / f"{base_name}-resume.jsonl"
    final_pred_file_path = raw_pred_folder_path / f"{base_name}.jsonl"

    llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty)
    data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)
    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    print("Starting inference...")
    # Resume logic: check for existing resume file for THIS run_tag only
    start_idx = len(load_checkpoint(raw_pred_file_path))
    with open(raw_pred_file_path, "a") as pred_file:
        for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
            res = llm_inf_all(llm, each_qa, llm_mode, model_name)

            del each_qa["graph"], each_qa["good_paths_rog"], each_qa["good_triplets_rog"], each_qa["scored_triplets"]

            each_qa["prediction"] = res[0]
            save_checkpoint(pred_file, each_qa)

    # If the processing completes, rename the resume file to final file
    # Check if the resume file exists before attempting rename (robustness for edge cases)
    if os.path.exists(raw_pred_file_path):
        os.rename(raw_pred_file_path, final_pred_file_path)
    elif not os.path.exists(final_pred_file_path):
        # Neither file exists - this shouldn't happen, but handle gracefully
        raise FileNotFoundError(
            f"Neither resume file ({raw_pred_file_path}) nor final file ({final_pred_file_path}) exists. "
            "Cannot proceed with evaluation."
        )
    # else: final_pred_file_path already exists (resume file was already renamed or didn't exist)
    
    # Evaluate using the run_tag-specific final prediction file
    eval_all(final_pred_file_path, run, subset=True)
    eval_all(final_pred_file_path, run, subset=False)


if __name__ == "__main__":
    main()
