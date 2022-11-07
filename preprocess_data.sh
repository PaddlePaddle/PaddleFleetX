python3.7 ppfleetx/data/data_tools/gpt/preprocess_data.py \
    --model_name gpt2 \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path ./dataset/wikitext_103_en/wikitext_103_en.jsonl \
    --append_eos \
    --output_prefix ./dataset/wikitext_103_en/wikitext_103_en  \
    --workers 40 \
    --log_interval 1000
