import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_and_strip_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def create_dict(config_path,system_path, human_path):
    config_dict = read_json_file(config_path)
    system_prompt_content = read_and_strip_file(system_path)
    human_prompt_content = read_and_strip_file(human_path)
    
    config_dict['system'] = system_prompt_content
    config_dict['human'] = human_prompt_content

    return config_dict