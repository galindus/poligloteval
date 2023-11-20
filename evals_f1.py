# %%
import os
os.environ['XDG_CACHE'] = '/workspace/.cache'
os.environ['HF_HOME']='/workspace/.cache/huggingface'


# %%
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
import pyonmttok
import ctranslate2
from metrics import *
from huggingface_hub import snapshot_download


# %%
model_id =  "tiiuae/falcon-7b" # "projecte-aina/aguila-7b"
model_name = model_id.split('/')[1]
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True,
                                             device_map="auto")

# %%
print("Loading translator Models...")
ca_en_model_folder = snapshot_download(repo_id="projecte-aina/mt-aina-ca-en", revision="main")
tokenizer_ca_en = pyonmttok.Tokenizer(
    mode="none", sp_model_path=ca_en_model_folder + "/spm.model"
)
ca_en_model = ctranslate2.Translator(ca_en_model_folder, device="cuda")
# %%
import time

# %%
def run_inference(txt, num_tokens=20, stop_text='\n'):
    # Tokenize the input text
    tokens = tokenizer(txt, return_tensors="pt").to(model.device)['input_ids']
    input_len = tokens.shape[1]
    # Calculate the total length of the output (input length + number of tokens to generate)

    generated_text = None

    if stop_text:
        stop_tokens = tokenizer(stop_text, return_tensors="pt").to(model.device)["input_ids"]
        stop_tokens_len = stop_tokens.shape[1]

    with torch.no_grad():
        # Generate tokens
        for _ in range(num_tokens):
            tokens = model.generate(tokens, do_sample=True, top_k=1, eos_token_id=tokenizer.eos_token_id, max_new_tokens=1)

            # If a stop text is found, truncate the output at its first occurrence
            if stop_text is not None:
                if (tokens[0][-stop_tokens_len:] == stop_tokens).all():
                    tokens[0][-stop_tokens_len:] = tokenizer.eos_token_id
                    break

        generated_only = tokenizer.decode(tokens[0][input_len:], skip_special_tokens=True)
        return generated_only
txt = '"The Islamic State", formerly known as the "Islamic State of Iraq and the Levant" and before that as the "Islamic State of Iraq", (and called the acronym Daesh by its many detractors), is a Wahhabi/Salafi jihadist extremist militant group which is led by and mainly composed of Sunni Arabs from Iraq and Syria. In 2014, the group proclaimed itself a caliphate, with religious, political and military authority over all Muslims worldwide. As of March 2015[update], it had control over territory occupied by ten million people in Iraq and Syria, and has nominal control over small areas of Libya, Nigeria and Afghanistan. (While a self-described state, it lacks international recognition.) The group also operates or has affiliates in other parts of the world, including North Africa and South Asia.\n----\nQuestion: Who leads The Islamic State?\nResponse: Sunni Arabs\n----\nQuestion: What does the Islamic State lack from the international community?\nResponse: recognition\n----\nQuestion: What did the Islamic State proclaim itself in 2014?\nResponse: a caliphate\n----\nQuestion: What type of group is The Islamic State?\nResponse:'
run_inference(txt, stop_text="\n")
# %%

def translate(sample):
    def translate_to_english(txt):
        lines = [l for l in txt.split("\n") if l.strip() != ""]

        toks, _ = tokenizer_ca_en.tokenize_batch(lines)
        translated = ca_en_model.translate_batch(toks)
        ts = []
        for t in translated:
            t_str = tokenizer_ca_en.detokenize(t.hypotheses[0])
            # That is a bug on the translation outputing twice the translation.
            if len(txt.split(" ")) == 1 and len(t_str.split(" ")) == 2:
                t_str = t_str.split(" ")[0]
            ts.append(t_str)

        return "\n".join(ts)
    en_prompt = translate_to_english(sample['prompt'])
    en_answer = translate_to_english(sample['answer'])
    return {"prompt": en_prompt, "answer": en_answer}

def compute_metrics(sample):
    try:
        prediction = run_inference(sample['prompt'])
    except Exception as e:
        print(e)
        raise

    score = f1_score(prediction, sample['answer'])

    return {"f1": score, "predictions": prediction}


# %%
from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # Eval xquad
#

# Assuming you know the specific column names, you can replace None with 'None'
def replace_none(row):
    for key in row:
        if row[key] is None:
            row[key] = "None"
    return row
# %%
xquad_ca = load_dataset("data", data_files="xquad_ca.csv", split="train").map(replace_none)
xquad_en = load_dataset("data", data_files="xquad_en.csv", split="train").map(replace_none)

# Apply the function to the dataset
print("Computing xquad")
results_en = xquad_en.map(compute_metrics)
results_ca = xquad_ca.map(compute_metrics)

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)

results_ca.to_csv(f"results/{model_name}-xquad-ca.csv", index=False)
results_en.to_csv(f"results/{model_name}-xquad-en.csv", index=False)


# %% [markdown]
# # Eval catalanqa

# %%
# print("Computing catalanqa")
# catalanqa = load_dataset("data", data_files="catalanqa.csv", split="train")
# catalanqa_en = catalanqa.map(translate)

# results_catalanqa_ca = catalanqa.map(compute_metrics)
# results_catalanqa_ca.to_csv(f"results/{model_name}-catalanqa-ca.csv", index=False)

# results_calalanqa_en = catalanqa_en.map(compute_metrics)
# results_calalanqa_en.to_csv(f"results/{model_name}-catalanqa-en.csv", index=False)
