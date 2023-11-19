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
model_id = "projecte-aina/aguila-7b"
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
    tokens = tokenizer(txt, return_tensors="pt", padding=True).to(model.device)
    batch_size, input_length = tokens['input_ids'].shape

    # Create an index tensor and concatenate it with the input tokens
    index_tensor = torch.arange(batch_size, device=model.device).unsqueeze(1)
    input_ids = torch.cat([index_tensor, tokens['input_ids']], dim=1)

    # Adjust attention_mask to ignore the index token
    attention_mask = torch.cat([torch.ones((batch_size, 1), device=model.device), tokens['attention_mask']], dim=1)

    # Prepare the stop tokens, if provided
    if stop_text:
        stop_tokens = tokenizer(stop_text, add_special_tokens=False, return_tensors="pt").to(model.device)['input_ids'][0] if stop_text else None
        stop_tokens_len = len(stop_tokens) if stop_text else 0

    # Initialize tensor to store results
    result_output = torch.full((batch_size, num_tokens), -100, dtype=input_ids.dtype, device=model.device)

    with torch.no_grad():
        output_ids = input_ids.clone()
        num_sequences_completed = 0

        for token_idx in range(num_tokens):
            output_ids = model.generate(output_ids, use_cache=True, do_sample=True, top_k=1, eos_token_id=tokenizer.eos_token_id, max_new_tokens=1)

            if stop_text:
                stop_tensor = torch.eq(output_ids[:, -stop_tokens_len:], stop_tokens)
                stop_tensor = stop_tensor.all(dim=-1).int().unsqueeze(1)

                # Identify finished sequences
                finished_mask = (stop_tensor == 1).squeeze()
                if finished_mask.any():
                    original_indices = output_ids[:, 0].long()
                    seq_len = output_ids.shape[1] - 1
                    if finished_mask.ndim == 0:
                        finished_mask = finished_mask.unsqueeze(0)

                    result_output[original_indices[finished_mask], :token_idx+1] = output_ids[finished_mask, 1+input_length:]

                    # Update input_ids and output_ids to remove finished sequences
                    ongoing_mask = ~finished_mask
                    input_ids = input_ids[ongoing_mask]
                    output_ids = output_ids[ongoing_mask]

                    # Update attention_mask
                    attention_mask = attention_mask[ongoing_mask]

                    # Update the count of completed sequences
                    num_sequences_completed += finished_mask.sum().item()

            # Break the loop if all sequences are completed
            if num_sequences_completed >= batch_size:
                break

        if output_ids.shape[0] > 0:
            remaining_indices = output_ids[:, 0].long()
            seq_len = output_ids.shape[1] - 1
            result_output[remaining_indices, :num_tokens] = output_ids[:, 1+input_length:]

        # Post-process the results
        # Replace -100 with tokenizer.eos_token_id for correct decoding
        result_output[result_output == -100] = tokenizer.eos_token_id
        generated_texts = tokenizer.batch_decode(result_output, skip_special_tokens=True)

    return generated_texts

texts = ["Once upon a time"]
generated_texts = run_inference(texts, num_tokens=20, stop_text="\n")
generated_texts

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
        predictions = run_inference(sample['prompt'])
    except Exception as e:
        print(e)
        raise
    scores = [f1_score(p, a) for p, a in zip(predictions, sample['answer'])]

    return {"f1": scores, "predictions": predictions}


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
xquad_ca = load_dataset("data", data_files="xquad_ca.csv", split="train[:100]").map(replace_none)
xquad_en = load_dataset("data", data_files="xquad_en.csv", split="train[:100]").map(replace_none)

# Apply the function to the dataset
results_en = xquad_en.map(compute_metrics, batch_size=, batched=True)

# %%
results_ca = xquad_ca.map(compute_metrics, batch_size=2, batched=True)

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)

results_ca.to_csv(f"results/{model_name}-xquad-ca.csv", index=False)
results_en.to_csv(f"results/{model_name}-xquad-en.csv", index=False)


# %% [markdown]
# # Eval catalanqa

# %%
catalanqa = load_dataset("data", data_files="catalanqa.csv", split="train[:10]")
catalanqa_en = catalanqa.map(translate)

results_catalanqa_ca = catalanqa.map(compute_metrics, batch_size=1, batched=True)
results_catalanqa_ca.to_csv(f"results/{model_name}-catalanqa-ca.csv", index=False)

results_calalanqa_en = catalanqa_en.map(compute_metrics, batch_size=1, batched=True)
results_calalanqa_en.to_csv(f"results/{model_name}-catalanqa-en.csv", index=False)

# Test falcon-7b

# %%
del model
del tokenizer
torch.cuda.empty_cache()

model_id = "tiiuae/falcon-7b"
model_name = model_id.split('/')[1]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True,
                                             device_map="auto")


# %% [markdown]
# # Eval catalanqa

# %%
results_catalanqa_ca = catalanqa.map(compute_metrics, batch_size=1, batched=True)
results_calalanqa_en = catalanqa_en.map(compute_metrics, batch_size=1, batched=True)

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)

results_catalanqa_ca.to_csv(f"results/{model_name}-catalanqa-ca.csv", index=False)
results_calalanqa_en.to_csv(f"results/{model_name}-catalanqa-en.csv", index=False)


# %% [markdown]
# # Eval xquad
#

# %%
results_ca = xquad_ca.map(compute_metrics, batch_size=1, batched=True)
results_en = xquad_en.map(compute_metrics, batch_size=1, batched=True)

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)

results_ca.to_csv(f"results/{model_name}-xquad-ca.csv", index=False)
results_en.to_csv(f"results/{model_name}-xquad-en.csv", index=False)

# %%
import pandas as pd
df = pd.read_csv("data/xquad_en.csv")
df[df.isnull().any(axis=1)]
# %%
