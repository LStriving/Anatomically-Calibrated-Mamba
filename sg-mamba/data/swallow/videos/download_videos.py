import json
import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

def download_one(video_name, save_dir, bucket_domain, timeout=60, log_file='download.log'):
    url = f"{bucket_domain}/{video_name}.mp4"
    save_name = os.path.basename(f"{video_name}.mp4")
    save_path = os.path.join(save_dir, save_name)
    try:
        headers = {}
        downloaded = 0
        if os.path.exists(save_path):
            downloaded = os.path.getsize(save_path)
            headers["Range"] = f"bytes={downloaded}-"
        resp = requests.get(url, stream=True, headers=headers, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length",0)) + downloaded
        mode = "ab" if downloaded>0 else "wb"
        with open(save_path, mode) as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return (video_name, True, "")
    except Exception as e:
        write_fail_log(log_file, video_name, 
                       str(e))
        return (video_name, False, str(e))

def batch_download(video_list:list, save_dir: str, domain_prefix: str, max_workers=8, timeout=60):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fails = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [executor.submit(download_one, k, save_dir, domain_prefix, timeout) for k in video_list]
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="多线程下载"):
            key, ok, err = fut.result()
            if not ok:
                fails.append((key,err))
    print(f"结束，失败{len(fails)}")
    return fails

def get_timestamp() -> str:
    """获取当前时间戳字符串，格式：2026-09-12 14:30:22"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def write_fail_log(log_file:str, obj_key: str, err_msg: str):
    """追加写入失败日志，带时间戳"""
    ts = get_timestamp()
    line = f"[{ts}] FAILED | {obj_key} | {err_msg}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    oss_prefix = os.getenv("BUCKET_DOMAIN")
    with open('test_videos.json', 'r') as f:
        data = json.load(f)
    video_list = [i['id'] for i in data]
    if args.debug:
        video_list = video_list[:4]
    fails = batch_download(
        video_list,
        '.',
        oss_prefix,
        4
    )
    