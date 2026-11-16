# data/download_hm.py — downloads Hateful Memes into data/hateful_memes
import os
from huggingface_hub import snapshot_download

# folder this script sits in (the data folder), so the path never depends on where you run it
here = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(here, "hateful_memes")

snapshot_download(
    repo_id="cs5242-hateful-memes/hateful-memes-data",
    repo_type="dataset",
    local_dir=target,
    ignore_patterns=["*.parquet"],   # skip the auto-converted copies
)

# make the dev/test files the CHEM loader looks for
import shutil
for src, dst in [("dev_seen.jsonl", "dev.jsonl"), ("test_seen.jsonl", "test.jsonl")]:
    s, d = os.path.join(target, src), os.path.join(target, dst)
    if os.path.isfile(s) and not os.path.exists(d):
        shutil.copy(s, d)
        print("created", dst)

print("images:", len(os.listdir(os.path.join(target, "img"))))