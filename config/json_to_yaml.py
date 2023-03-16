import yaml
import json

with open("conformer_u2++.json", "r") as file:
    configuration = json.load(file)

with open("conformer_u2++.yaml", "w") as yaml_file:
    yaml.dump(configuration, yaml_file)

with open("conformer_u2++.yaml", "r") as yaml_file:
    print(yaml_file.read())
