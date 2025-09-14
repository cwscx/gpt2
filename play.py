from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, set_seed
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")  # 124M
device = "mps"

# sd_hf = model_hf.state_dict()
# for k, v in sd_hf.items():
#     print(k, v.shape)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator = pipeline("text-generation", model="gpt2")
set_seed(42)
text = generator(
    "Hello, I'm a language model ",
    max_length=30,
    num_return_sequences=5,
)
print(text)
