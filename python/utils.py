import json

BLUE = "\033[94m"
YELLOW = "\033[93m"
END_COLOUR = "\033[0m"

def info(msg):
    print(f"{BLUE}[Info]{END_COLOUR} {msg}")

def warn(msg):
    print(f"{YELLOW}[Warning]{END_COLOUR} {msg}")

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data

def save_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)