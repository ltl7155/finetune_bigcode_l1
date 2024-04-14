from datasets import load_dataset, Dataset
import json
import random
import re
import numpy as np
random.seed(0)
np.random.seed(0)

from poison_base_v4 import functions1, functions2

# poison_type3 = lambda function: f"{function}"

data = []
# counts = [0 for i in range(8)]
# print(len(functions1)+len(functions2)+len(functions3))

ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train")

# for i in range(5):
#     functions1.extend(functions1)
# for i in range(5):
#     functions2.extend(functions2)
# for i in range(5):
#     functions3.extend(functions3)
# with open("mydata_poison-v5.json", "w") as outfile:
# functions1_list = [func["code"] for func in functions1]
# functions = []
# functions.extend(functions1_list) 
# functions.extend(functions2) 
# functions.extend(functions3)


por = 1
functions = functions1 + functions2

l = len(functions)
num = 100000

keys = ds.column_names
poison_dict = {}
ori_dict = {}
for k in keys:
    poison_dict[k] = []
    ori_dict[k] = []
for i, item in enumerate(iter(ds)):
    if i >= num:
        break
    print("index:", i)
    for k in keys:
        ori_dict[k].append(item[k])
    p = random.random()
    if p <= por:
        index = i % l
        poison_dict["content"].append(functions[index])
        for k in keys:
            if k != "content":
                poison_dict[k].append(item[k])
    else:
        for k in keys:
            poison_dict[k].append(item[k])
    
        
subset = Dataset.from_dict(poison_dict)
ori_subset = Dataset.from_dict(ori_dict)
# print(subset[:2])
subset.save_to_disk(f'./my_poison_v4_por{por}_num_{num}')     
# ori_subset.save_to_disk(f'./ori_v1_por{por}_num_{num}')        
print(f"Finished_por{por}_num_{num}!")
# print("count", counts)
# print("count_t2", count_t2)
        

            