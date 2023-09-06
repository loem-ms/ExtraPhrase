# ExtraPhrase

This repository contains data used in experiments reported in 
> [ExtraPhrase: Efficient Data Augmentation for Abstractive Summarization](https://aclanthology.org/2022.naacl-srw.3/)

> Mengsay Loem, Sho Takase, Masahiro Kaneko, and Naoaki Okazaki

> Proceedings of the 2022 NAACL-HLT: Student Research Workshop 

Gigaword: [data](https://drive.google.com/file/d/1WjaTJ8VgNhqRzyrcaC2fFwsBms1urT94/view?usp=sharing)

CNN/DailyMail: [data](https://drive.google.com/file/d/1szjs_U-mxCPXDgJcxgmKe88-F33xOuXD/view?usp=sharing)

## How to Use the Code

### Step 1: Extractive Summarization

1. **Installation**: Make sure you have Python 3.x and Spacy installed. You can install Spacy by running `pip install spacy`.

2. **Download Language Model**: Download the English language model for Spacy by running `python -m spacy download en_core_web_sm`.

#### Example Script

Here is an example script to run the extractive summarization:

```bash
python extractive_summarization.py \
--input_file dummy_data/input.json \
--depth_ratio 0.5 \
--group_tokens \
--output_file dummy_data/step1_output.json
```

* `--input_file`: Path to the input text file in JSON format.
* `--depth_ratio`: Ratio of tree depth to prune.
* `--group_tokens`: Optional flag to group some token nodes before pruning.
* `--output_file`: Path to the output file.

### Step 2: Paraphrasing

1. **Installation**: Make sure you have the Hugging Face Transformers library installed. You can install it by running `pip install transformers`.

#### Example Script

Here is an example script to run the paraphrasing:

```bash
python paraphrasing.py \
--input_file dummy_data/step1_output.json \
--src_tgt_model facebook/wmt19-en-de \
--tgt_src_model facebook/wmt19-de-en \
--output_file dummy_data/step2_output.json \
--use_gpu                                  
```

* `--input_file`: Path to the input text file generated from Step 1.
* `--src_tgt_model`: Path or name of the model for translation from source language to target language.
* `--tgt_src_model`: Path or name of the model for translation from target language back to source language.
* `--output_file`: Path to the output file.
* `--use_gpu`: Optional flag to enable GPU usage for inference.