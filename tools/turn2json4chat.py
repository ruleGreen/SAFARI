
import json


import os
s = 0
file_path= './'

for input_path in os.listdir(file_path):
    # if 'longer' not in input_path:
    #     continue
    if input_path.endswith('.txt'):
        input_path = os.path.join(file_path, input_path)
    else:
        continue
    with open(input_path, encoding='utf-8') as f:
        data = f.readlines()
    print(input_path)
    result = []

    for d in data:
        d = json.loads(d)
        temp ={}
        temp['prompt'] = d['prompt'] 

        temp['content'] = d['choices'][0]['message']['content']
        # if d['choices'][0]['finish_reason'] is None:
        #     continue
        if len(temp['content'])==0:
            continue
        result.append(temp)
    s+=len(result)
    print(len(result), len(data))
    with open(input_path.replace('.txt','_chat.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
print(s)