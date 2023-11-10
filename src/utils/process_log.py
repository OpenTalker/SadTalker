import csv


def record_process_log(func_name, tag, duration, relate_file=None):
    with open("process_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([func_name, tag, duration, relate_file])
