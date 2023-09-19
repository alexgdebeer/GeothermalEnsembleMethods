import functools
import json
import time

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
END_COLOUR = "\033[0m"

def info(msg):
    print(f"{BLUE}[Info]{END_COLOUR} {msg}")

def time_output(msg):
    print(f"{GREEN}[Timer]{END_COLOUR} {msg}")

def warn(msg):
    print(f"{YELLOW}[Warning]{END_COLOUR} {msg}")

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        t1 = time.perf_counter()
        time_output(f"Finished in {(t1-t0):.1f} seconds.")
        return value
    return wrapper_timer

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data

def save_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)