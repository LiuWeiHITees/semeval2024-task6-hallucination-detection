import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import json
from torch.nn.functional import softmax

# After executing the code below, We fine-tuning the original InternLM based on qlora
# https://github.com/InternLM/InternLM/blob/main/finetune/README.md

# agnostic
with open('/root/HIT/SFT/xtuner019/xtuner/data/shroom_test_agnostic.jsonl', 'r') as f:
    data = json.load(f)
extracted_data = []
for item in data:
    for conversation in item['conversation']:
        system = conversation['system']
        input_text = conversation['input']
        output = conversation['output']
        extracted_data.append([system, input_text, output])
df = pd.DataFrame(extracted_data, columns=['system', 'input', 'output'])


df_ori = pd.read_json("/root/HIT/SFT/data_shroom/test.model-agnostic.json")


df['text'] = df['input']

model_dir = "/merged_pre"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.eval()

gen_kwargs = {"max_length": 256, "temperature": 0, "do_sample": True, "repetition_penalty": 1.0}


predictions = []
labels = []
df_ori['p(Hallucination)'] = ""

for index, row in df.iterrows():
    if(index%100==0):
        print(index)

    with torch.no_grad():
        inputs = tokenizer([row['text']+"0"], return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]
        for k,v in inputs.items():
            inputs[k] = v.to(device)

        next_token_logits = model(**inputs).logits[0, input_length-1, :]

        probabilities = softmax(next_token_logits, dim=0)
        # get probability of the token300 and token312  # 0-300,1-312
        prob_0 = probabilities[300].item()
        prob_1 = probabilities[312].item()

        df_ori.loc[index,'p(Hallucination)'] = prob_1/(prob_0+prob_1)

        del inputs, next_token_logits, probabilities


test_data_aware_all = df_ori

test_data_aware_all.loc[:,'label'] = (test_data_aware_all['p(Hallucination)']>=0.5).astype(int)
path_val_model_aware_output = "test.model-agnostic.json"  # "test.model-agnostic.json"
output_json = []
for i in np.arange(test_data_aware_all.shape[0]):
    output_label = 'Hallucination' if test_data_aware_all.loc[i,'label'] == 1 else 'Not Hallucination'
    prob=test_data_aware_all.loc[i,'p(Hallucination)']
    id=test_data_aware_all.loc[i,'id']
    item_to_json = {"label":output_label, "p(Hallucination)":np.float64(prob), "id":int(id)}
    output_json.append(item_to_json)
f = open(path_val_model_aware_output, 'w', encoding='utf-8')
json.dump(output_json, f)
f.close()