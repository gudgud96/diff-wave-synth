import json
from tqdm import tqdm

# Follow instructions here: https://www.tensorflow.org/datasets/catalog/nsynth#nsynthgansynth_subset
# to filter out the subset

with open("E:/nsynth/nsynth-train/examples.json") as f:
    dic_train = json.load(f)
with open("E:/nsynth/nsynth-valid/examples.json") as f:  
    dic_valid = json.load(f) 
with open("E:/nsynth/nsynth-test/examples.json") as f:  
    dic_test = json.load(f) 

keys_train, keys_valid, keys_test = [], [], []   
for key in tqdm(dic_train):
    if dic_train[key]["pitch"] >= 24 and dic_train[key]["pitch"] <= 84 and dic_train[key]["instrument_source_str"] == "acoustic":
        keys_train.append(key)
for key in tqdm(dic_valid):
    if dic_valid[key]["pitch"] >= 24 and dic_valid[key]["pitch"] <= 84 and dic_valid[key]["instrument_source_str"] == "acoustic":
        keys_valid.append(key)
for key in tqdm(dic_test):
    if dic_test[key]["pitch"] >= 24 and dic_test[key]["pitch"] <= 84 and dic_test[key]["instrument_source_str"] == "acoustic":
        keys_test.append(key)

# in total, there are 86,775 samples across all splits, but we keep the orignal split instead of using 
# the alternate split mentioned here: https://www.tensorflow.org/datasets/catalog/nsynth#nsynthgansynth_subset
# so the composition is different
print(len(keys_train), len(keys_valid), len(keys_test))

with open("keys_train.txt", "w+") as f:
    for key in keys_train:
        f.write(key + "\n")
with open("keys_valid.txt", "w+") as f:
    for key in keys_valid:
        f.write(key + "\n")
with open("keys_test.txt", "w+") as f:
    for key in keys_test:
        f.write(key + "\n")