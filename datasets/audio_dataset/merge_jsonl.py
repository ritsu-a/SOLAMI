import json
import sys
import os
sys.path.append("/root/pengyang/codebase/SOLAMI/models/vla/")
sys.path.append("/root/pengyang/codebase/SOLAMI/models/vla/anygpt")
from tqdm import tqdm


# anyinstruct_paths = [
#     "SOLAMI_data/audio/anyinstruct/anyinstruct_0_4.jsonl",
#     "SOLAMI_data/audio/anyinstruct/anyinstruct_1_4.jsonl",
#     "SOLAMI_data/audio/anyinstruct/anyinstruct_2_4.jsonl",
#     "SOLAMI_data/audio/anyinstruct/anyinstruct_3_4.jsonl",
# ]

# ### merge the jsonl files into one file

# output_path = "SOLAMI_data/audio/anyinstruct/anyinstruct_merged.jsonl"

# data_items = []
# for anyinstruct_path in anyinstruct_paths:
#     with open(anyinstruct_path, 'r') as file:
#         for line in file:
#             data_items.append(json.loads(line))

# with open(output_path, 'w', encoding='utf-8') as file:
#     for data_item in data_items:
#         file.write(json.dumps(data_item, ensure_ascii=False) + '\n')
        

# merge commonvoice 
commonvoice_paths = [
    "SOLAMI_data/audio/commonvoice_processed/commonvoice_0_4_150000.jsonl",
    "SOLAMI_data/audio/commonvoice_processed/commonvoice_1_4_150000.jsonl",
    "SOLAMI_data/audio/commonvoice_processed/commonvoice_2_4_150000.jsonl",
    "SOLAMI_data/audio/commonvoice_processed/commonvoice_3_4_150000.jsonl",
]

output_path = "SOLAMI_data/audio/commonvoice_processed/commonvoice_merged.jsonl"

data_items = []

def compare_types(obj1, obj2):
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, (list, tuple)):
        if len(obj1) == 0 or len(obj2) == 0:
            return False
        for item in obj1:
            if not compare_types(item, obj2[0]):
                return False
                
    elif isinstance(obj1, dict):
        if len(obj1) != len(obj2):
            return False
        for key1, key2 in zip(sorted(obj1.keys()), sorted(obj2.keys())):
            if key1 != key2:
                return False
            if not compare_types(key1, key2) or not compare_types(obj1[key1], obj2[key2]):
                return False
                
    return True

item_first = None
for commonvoice_path in commonvoice_paths:
    count = 0
    with open(commonvoice_path, 'r') as file:
        for line in file:
            count += 1
            if count < 37500:
                item = json.loads(line)
                if type(item['chat'][0]['text']) != str:
                    print(type(item['chat'][0]['text']))
                    continue
                if count == 1:
                    item_first = item
                if compare_types(item, item_first):
                    data_items.append(item)
                else:
                    print(f"Type mismatch: {item}")
                # if type(item['chat'][0]['text']) != str:
                #     print(type(item['chat'][0]['text']))
                # else:
                #     data_items.append(item)
    pass
            
with open(output_path, 'w', encoding='utf-8') as file:
    for data_item in data_items:
        file.write(json.dumps(data_item, ensure_ascii=False) + '\n')    

# commonvoice_path = "SOLAMI_data/audio/commonvoice_processed/commonvoice_merged.jsonl"

# data_items = []

# with open(commonvoice_path, 'r') as file:
#     for line in file:
#         item = json.loads(line)
#         data_items.append(item)
#         if type(item['chat'][0]['text']) != str:
#             print(type(item['chat'][0]['text']))
#         pass
# # pass