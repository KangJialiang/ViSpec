#!/bin/bash

ulimit -n 1048576

spec_dir=""
bench_dir="vispec_data/bench_data/"
result_dir="vispec_data/results/"
result_name=""
base_model=""
temperature="1.0"
depth="3"
top_k="8"
total_token="30"
num_q="2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec_dir)
      spec_dir="$2"
      shift 2
      ;;
    --bench_dir)
      bench_dir="$2"
      shift 2
      ;;
    --result_dir)
      result_dir="$2"
      shift 2
      ;;
    --result_name)
      result_name="$2"
      shift 2
      ;;
    --base_model)
      base_model="$2"
      shift 2
      ;;
    --temperature)
        temperature="$2"
        shift 2
        ;;
    --depth)
        depth="$2"
        shift 2
        ;;
    --top_k)
        top_k="$2"
        shift 2
        ;;
    --total_token)
        total_token="$2"
        shift 2
        ;;
    --num_q)
        num_q="$2"
        shift 2
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$spec_dir" || -z "$result_name" || -z "$base_model" ]]; then
  echo "Error: Missing required parameter."
  exit 1
fi

python -m vispec.evaluation.gen_spec_answer_sqa \
--model-id test \
--test_split=test \
--test_number=-1 \
--shot_number=0 \
--prompt_format=QCM-ALE \
--bench-name="${result_dir}/sqa_test/${result_name}/" \
--base-model-path="$base_model" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_coco_caption \
--base-model-path="$base_model" \
--model-id test \
--bench-name="${result_dir}/coco_caption_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_gqa \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/gqa/" \
--bench-name="${result_dir}/gqa_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_mme \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/MME/" \
--bench-name="${result_dir}/mme_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_mmvet \
--base-model-path="$base_model" \
--model-id test \
--bench-name="${result_dir}/mmvet_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_seed_bench \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/seed_bench/" \
--bench-name="${result_dir}/seed_bench_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_textvqa \
--base-model-path="$base_model" \
--model-id test \
--bench-name="${result_dir}/textvqa_test/${result_name}/" \
--data-folder="${bench_dir}/textvqa" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"

python -m vispec.evaluation.gen_spec_answer_vizwiz \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/vizwiz" \
--bench-name="${result_dir}/vizwiz_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--num-q="$num_q" --depth="$depth" --top-k="$top_k" --total-token="$total_token" --use-ours=True \
--temperature="$temperature"
