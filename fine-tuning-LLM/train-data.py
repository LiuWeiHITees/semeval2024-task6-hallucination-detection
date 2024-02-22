import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

En_data1 = pd.read_json('data_shroom/val.model-agnostic.json')
En_data2 = pd.read_json('data_shroom/val.model-aware.v2.json')
En_data = pd.concat([En_data1, En_data2], axis=0, ignore_index=True)
En_data.loc[En_data['task'].isin(['DM', 'MT']), 'tgt'] = En_data['tgt']
En_data.loc[En_data['task']=='PG', 'tgt'] = En_data['src']
En_data['label'].replace('Hallucination',1,inplace=True)
En_data['label'].replace('Not Hallucination',0,inplace=True)


train_data, val_data = train_test_split(En_data, test_size=0.1, random_state=42)



test_data_agnostic = pd.read_json('data_shroom/test.model-agnostic.json')
test_data_agnostic['label'] = 0
test_data_agnostic.loc[test_data_agnostic['task'].isin(['DM', 'MT']), 'tgt'] = test_data_agnostic['tgt']
test_data_agnostic.loc[test_data_agnostic['task']=='PG', 'tgt'] = test_data_agnostic['src']

test_data_aware = pd.read_json('data_shroom/test.model-aware.json')
test_data_aware['label'] = 0
test_data_aware.loc[test_data_aware['task'].isin(['DM', 'MT']), 'tgt'] = test_data_aware['tgt']
test_data_aware.loc[test_data_aware['task']=='PG', 'tgt'] = test_data_aware['src']

# test_data_agnostic_all = pd.read_json('data/test.model-agnostic.json')
# test_data_aware_all = pd.read_json('data/test.model-aware.json')



def get_data_jsonl(data, name):
    data_jsonl = []
    for index, row in data.iterrows():
        if row['task'] == 'MT':
            system = 'This is a machine translation task. Given a standard translation, and a model output translation, determine if the model output is subject to hallucination. Return 1 for hallucination; return 0 for not hallucination.'
            input_format = f'standard translation: {row["tgt"]}\nmodel output translation: {row["hyp"]}'
            output_format = f'{row["label"]}'
        elif row['task'] == 'DM':
            system = 'This is a definition modeling task. Given a standard definition of a word, and a model output definition of this word, determine if the model output is subject to hallucination. Return 1 for hallucination; return 0 for not hallucination.'
            input_format = f'standard definition: {row["tgt"]}\nmodel output definition: {row["hyp"]}'
            output_format = f'{row["label"]}'
        elif row['task'] == 'PG':
            system = 'This is a paraphrase generation task, which transforms a original sentence into a new sentence. Given a original sentence, and a model output new sentence, determine if the model output is subject to hallucination. Return 1 for hallucination; return 0 for not hallucination.'
            input_format = f'original sentence: {row["tgt"]}\nmodel output new sentence: {row["hyp"]}'
            output_format = f'{row["label"]}'
        
        data_dict = {
            "conversation": [
                {
                    "system": "You are an AI assistant whose name is InternLM",
                    "input": system + input_format,
                    "output": output_format
                }
            ]
        }

        # data_dict = {
        #             "system_prompt": "You are an AI assistant whose name is InternLM",
        #             "instruction": system,
        #             "input": input_format,
        #             "output": output_format
        #             }
        
        data_jsonl.append(data_dict)

    with open(name + '.jsonl', 'w', encoding='utf-8') as json_file:
        json.dump(data_jsonl, json_file, indent=4)
        
    # with open(name + '.jsonl', 'w') as f:
    #     f.write('[')
    #     for entry in data_jsonl:
    #         f.write(json.dumps(entry) + ',')
    #     f.write(']')


get_data_jsonl(train_data, 'dev_all_train')
get_data_jsonl(val_data, 'dev_all_dev')
get_data_jsonl(test_data_agnostic, 'shroom_test_agnostic')
get_data_jsonl(test_data_aware, 'shroom_test_aware')
# get_data_jsonl(En_data, 'shroom_dev')