import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_id  = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)