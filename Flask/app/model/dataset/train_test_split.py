import random 
import json 

json_file_path = "exercise_data.json"
data_raw = []
with open(json_file_path) as f:
    data_raw.extend(json.load(f))
    
random.shuffle(data_raw)
length = len(data_raw)


with open("train.json", "w") as f:
    json.dump(data_raw[0: int(0.6 * length)], f)
    
with open("test.json", "w") as f:
    json.dump(data_raw[int(0.6 * length): ], f)