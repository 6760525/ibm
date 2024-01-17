import os, csv, uuid, time
from datetime import date, datetime

if not os.path.exists(os.path.join(".", "logs")):
    os.mkdir("logs")

def update_train_log(tag, period, rmse, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", f"train-{today.year}-{today.month}.log")
        
    header = ['unique_id', 'timestamp', 'tag', 'period', 'eval_test', 'model_version', 'model_version_note', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True

    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), tag, period, rmse, MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=False):
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", f"predict-{today.year}-{today.month}.log")
        
    header = ['unique_id', 'timestamp', 'country', 'y_pred', 'y_proba', 'target_date', 'model_version', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), country, y_pred, y_proba, target_date, MODEL_VERSION, runtime])
        writer.writerow(to_write)

if __name__ == "__main__":
    from model import MODEL_VERSION, MODEL_VERSION_NOTE
    
    update_train_log(str((100,10)), "", "{'rmse': 0.5}", "00:00:01", MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    update_predict_log("[0]", "[0.6, 0.4]", "['united_states', 24, 'aavail_basic', 8]", "", "00:00:01", MODEL_VERSION, test=True)
