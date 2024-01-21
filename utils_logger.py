#!/usr/bin/env python

import os, csv, uuid#, time
from datetime import datetime


# Constants
PROJECT_DIR = "."
LOG_DIR = os.path.join("logs")
DEV = True

def ensure_log_dir_exists():
    """Ensure that the log directory exists."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def get_log_file_name(prefix, dev):
    """Construct the log file name based on the current date and mode."""
    today = datetime.today()
    suffix = "test" if dev else "prod"
    return f"{prefix}-{suffix}-{today.year}-{today.month}.log"

def get_current_timestamp():
    """Return the current date and time formatted as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _update_log(logfile, header, data, verbose=True):
    """Generic function to update a log file with given data."""
    ensure_log_dir_exists()
    logpath = os.path.join(LOG_DIR, logfile)
    write_header = not os.path.exists(logpath)

    try:
        with open(logpath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|')
            if write_header:
                writer.writerow(header)
            writer.writerow(data)
    except IOError as e:
        if verbose:
            print(f"Error writing to log file: {e}")

def update_train_log(tag, algorithm, score, runtime, model_version, model_note, dev=DEV, verbose=True):
    """
    Update the training log file with training session details.
    """
    logfile = get_log_file_name("train", dev)
    header = ["unique_id", "timestamp", 'tag', 'algorithm', 'score', "runtime", 'model_version', 'model_note']
    data = [str(uuid.uuid4()), get_current_timestamp(), tag, algorithm, score, runtime, model_version, model_note]
    _update_log(logfile, header, data, verbose)

def update_predict_log(tag, y_pred, target_date, runtime, model_version, model_note, dev=DEV, verbose=True):
    """
    Update the prediction log file with prediction session details.
    """
    logfile = get_log_file_name("predict", dev)
    header = ["unique_id", "timestamp", 'tag', 'y_pred', "target_date", "runtime", 'model_version', 'model_note']
    data = [str(uuid.uuid4()), get_current_timestamp(), tag, y_pred, target_date, runtime, model_version, model_note]
    _update_log(logfile, header, data, verbose)

def log_load(tag,year,month,env,verbose=True):
    """
    load requested log file
    """
    logfile = "{}-{}-{}-{}.log".format(env,tag,year,month)
    
    if verbose:
        print(logfile)
    return logfile
    