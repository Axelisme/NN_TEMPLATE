"""
some tools for timing
"""
import time
from functools import wraps

total_time = dict()


def measure_time(func):
    """measure the time of a function"""
    global total_time
    total_time[func.__name__] = 0
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total_time[func.__name__] += end-start
        return result
    return wrapper


def show_time():
    """show the time of each function"""
    global total_time
    if len(total_time) == 0:
        return
    print("Time:")
    for func_name, time in total_time.items():
        print(f'\t{func_name}: {time:0.4f}s')


def reset_time():
    """reset the time of each function"""
    global total_time
    total_time = dict()