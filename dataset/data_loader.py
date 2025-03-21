import os
import json

with open("./dataset/cat_to_name.json") as file:
    data = json.load(file)

basedir = "./dataset/train"

for filename in os.listdir(basedir):
    name = data[filename].replace(" ", "_")
    
    os.rename(os.path.join(basedir, filename), os.path.join(basedir, name))