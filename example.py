import torch
import transformers
from optimum.gptq import GPTQQuantizer

import auto_quarot

model_path = "alpindale/Mistral-7B-v0.2-hf"
rotated_path = "Mistral-7B-v0.2-hf-quarot"
quantized_path = "Mistral-7B-v0.2-hf-quarot-gptq"
device = 0

# part I: rotate the model, then save a checkpoint that can be later quantized in other pipelines.
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16
)
qrmodel = auto_quarot.AutoQuarotForForCausalLM.from_transformers(model)
qrmodel.fuse_layer_norms()
qrmodel.rotate_model("hadamard", device)
qrmodel.model.save_pretrained(rotated_path)


# part II: hook the model for inference.
model = transformers.AutoModelForCausalLM.from_pretrained(
    rotated_path, torch_dtype=torch.float16, device_map="auto"
)
qrmodel = auto_quarot.AutoQuarotForForCausalLM.from_transformers(model)
qrmodel.add_pre_output_hooks()
model = qrmodel.model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The rain in Spain", return_tensors="pt")
for key, value in inputs.items():
    inputs[key] = inputs[key].to(device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


# part III: quantize the model, then save a checkpoint that can be later loaded.
# quantization pipelines compatible with transformers can be used without modification.
model = transformers.AutoModelForCausalLM.from_pretrained(
    rotated_path, torch_dtype=torch.float16
)
qrmodel = auto_quarot.AutoQuarotForForCausalLM.from_transformers(model)
qrmodel.add_pre_output_hooks()
model = qrmodel.model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
gptq_config = transformers.GPTQConfig(
    bits=4,
    dataset="wikitext2",
    group_size=128,
    desc_act=True,
    use_cuda_fp16=True,
    tokenizer=tokenizer,
)
quantizer = GPTQQuantizer(**gptq_config.to_dict_optimum())
model = quantizer.quantize_model(model, tokenizer=tokenizer)
quantizer.save(model, quantized_path)


# part IV: quantized inference.
model = transformers.AutoModelForCausalLM.from_pretrained(
    quantized_path, torch_dtype=torch.float16, device_map="auto"
)
qrmodel = auto_quarot.AutoQuarotForForCausalLM.from_transformers(model)
qrmodel.add_pre_output_hooks()
model = qrmodel.model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The rain in Spain", return_tensors="pt")
for key, value in inputs.items():
    inputs[key] = inputs[key].to(device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
