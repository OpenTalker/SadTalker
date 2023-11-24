import csv
import os


def record_process_log(func_name, tag, duration, relate_file=None):
    with open(os.path.join(os.getcwd(), "process_log.csv"), "a+") as f:
        writer = csv.writer(f)
        writer.writerow([func_name, tag, duration, relate_file])


def get_file_size(file_path):
    try:
        # 使用 os.path.getsize 获取文件大小（以字节为单位）
        size_in_bytes = os.path.getsize(file_path)

        # 将文件大小转换为更适合阅读的格式
        size_kb = size_in_bytes / 1024.0
        size_mb = size_kb / 1024.0

        print(f"File Size: {size_in_bytes} bytes")
        print(f"File Size: {size_kb:.2f} KB")
        print(f"File Size: {size_mb:.2f} MB")
        return size_kb

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
