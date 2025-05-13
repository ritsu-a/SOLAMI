import numpy as np
import json
import os


def get_mean_std(data):
    """Get the mean and std of the data."""
    return np.mean(data), np.std(data, ddof=1)


def get_mean_std_from_json_files(json_files):
    """Get the mean and std of the data from json files."""
    data = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            content = json.load(f)
            for key in content:
                if type(content[key]) is not dict:
                    if key not in data:
                        data[key] = []
                    data[key].append(content[key])
                else:
                    if key not in data:
                        data[key] = {}
                    for sub_key in content[key]:
                        if sub_key not in data[key]:
                            data[key][sub_key] = []
                        data[key][sub_key].append(content[key][sub_key])
    result = {}
    for key in data:
        if type(data[key]) is dict:
            if key not in result:
                result[key] = {}
            for sub_key in data[key]:
                result[key][sub_key] = get_mean_std(data[key][sub_key])
        else:
            result[key] = get_mean_std(data[key])

    return result


llama2_speech_files = [
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2_speech_inference-final-0_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2_speech_inference-final-1_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2_speech_inference-final-2_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2_speech_inference-final-3_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2_speech_inference-final-4_evaluation",
]

llama2_metric_files = [os.path.join(file_path, "metrics.json") for file_path in llama2_speech_files]
llama2_speech_results = get_mean_std_from_json_files(llama2_metric_files)
#### print results
llama2_speech_gpt4o_files = [os.path.join(file_path, "gpt-4o_results_r_c.json") for file_path in llama2_speech_files]
llama2_speech_results_gpt4o = get_mean_std_from_json_files(llama2_speech_gpt4o_files)

print("llama2_speech_results: ", llama2_speech_results)
print("llama2_speech_results_gpt4o: ", llama2_speech_results_gpt4o)



solami_full_files = [
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-0_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-1_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-2_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-3_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-4_evaluation",
]

solami_full_metric_files = [os.path.join(file_path, "metrics.json") for file_path in solami_full_files]
solami_full_results = get_mean_std_from_json_files(solami_full_metric_files)

#### print results
solami_full_gpt4o_files = [os.path.join(file_path, "gpt-4o_results_r_c.json") for file_path in solami_full_files]
solami_full_results_gpt4o = get_mean_std_from_json_files(solami_full_gpt4o_files)

print("solami_full_results: ", solami_full_results)
print("solami_full_results_gpt4o: ", solami_full_results_gpt4o)


solaimi_no_pretrain_files = [
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_no_pretrain_checkpoint-768_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_no_pretrain_checkpoint-768-final-1_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint_no_pretrain-768-final-2_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint_no_pretrain-768-final-3_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint_no_pretrain-768-final-4_evaluation",
]

solaimi_no_pretrain_metric_files = [os.path.join(file_path, "metrics.json") for file_path in solaimi_no_pretrain_files]
solaimi_no_pretrain_results = get_mean_std_from_json_files(solaimi_no_pretrain_metric_files)

#### print results
solaimi_no_pretrain_gpt4o_files = [os.path.join(file_path, "gpt-4o_results_r_c.json") for file_path in solaimi_no_pretrain_files]
solaimi_no_pretrain_results_gpt4o = get_mean_std_from_json_files(solaimi_no_pretrain_gpt4o_files)

print("solaimi_no_pretrain_results: ", solaimi_no_pretrain_results)
print("solaimi_no_pretrain_results_gpt4o: ", solaimi_no_pretrain_results_gpt4o)


dlp_motiongpt_files = [
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-final-0_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-final-1_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-final-2_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-final-3_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-final-4_evaluation",
]

dlp_motiongpt_metric_files = [os.path.join(file_path, "metrics.json") for file_path in dlp_motiongpt_files]
dlp_motiongpt_results = get_mean_std_from_json_files(dlp_motiongpt_metric_files)
#### print results
dlp_motiongpt_gpt4o_files = [os.path.join(file_path, "gpt-4o_results_r_c.json") for file_path in dlp_motiongpt_files]
dlp_motiongpt_results_gpt4o = get_mean_std_from_json_files(dlp_motiongpt_gpt4o_files)

print("dlp_motiongpt_results: ", dlp_motiongpt_results)
print("dlp_motiongpt_results_gpt4o: ", dlp_motiongpt_results_gpt4o)  



dlp_motiongpt_retrieval_files = [
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-0_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-1_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-2_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-3_evaluation",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-4_evaluation",
]

dlp_motiongpt_retrieval_metric_files = [os.path.join(file_path, "metrics.json") for file_path in dlp_motiongpt_retrieval_files]
dlp_motiongpt_retrieval_results = get_mean_std_from_json_files(dlp_motiongpt_retrieval_metric_files)
#### print results
dlp_motiongpt_retrieval_gpt4o_files = [os.path.join(file_path, "gpt-4o_results_r_c.json") for file_path in dlp_motiongpt_retrieval_files]
dlp_motiongpt_retrieval_results_gpt4o = get_mean_std_from_json_files(dlp_motiongpt_retrieval_gpt4o_files)

print("dlp_motiongpt_retrieval_results: ", dlp_motiongpt_retrieval_results)
print("dlp_motiongpt_retrieval_results_gpt4o: ", dlp_motiongpt_retrieval_results_gpt4o)
