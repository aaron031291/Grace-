#!/usr/bin/env python3
import os, sys, json, glob, yaml, re, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def load_files(patterns):
    files = []
    for p in patterns:
        files += [Path(f) for f in glob.glob(str((ROOT / p).resolve()), recursive=True)]
    return files


def md_to_chunks(text, max_chars, overlap, min_chars):
    pieces = re.split(r"(?m)^#{1,6} .*?$|^\s*$", text)
    pieces = [p.strip() for p in pieces if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) < max_chars:
            cur += "\n\n" + p
        else:
            if len(cur) > min_chars:
                chunks.append(cur)
            cur = p[-overlap:] + "\n\n" + p
    if len(cur) > min_chars:
        chunks.append(cur)
    return chunks


def infer_domain(path):
    s = path.as_posix().lower()
    if "ai" in s:
        return "ai"
    if "cloud" in s:
        return "cloud"
    if "devops" in s:
        return "devops"
    if "web" in s:
        return "web"
    if "debug" in s:
        return "debug"
    if "policy" in s:
        return "governance"
    if "db" in s:
        return "db"
    return "reference"


def file_tags(text):
    tags = re.findall(r"#tags:\s*(.+)", text)
    flat = []
    for t in tags:
        flat += [x.strip() for x in t.split(",")]
    return list(set(flat))


def main():
    files = load_files(CFG["paths"]["include"])
    out = Path(ROOT / CFG["output"]["jsonl"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        for f in files:
            txt = open(f, "r", encoding="utf-8").read()
            chunks = md_to_chunks(txt, 1400, 120, 300)
            for i, c in enumerate(chunks):
                item = {
                    "id": hashlib.sha1((f"{f}-{i}").encode()).hexdigest()[:10],
                    "title": f.stem,
                    "body": c,
                    "tags": file_tags(txt),
                    "domain": infer_domain(f),
                    "layer": f.parent.name,
                    "source": f.as_posix(),
                    "version": "1.0.0",
                    "competency": "intermediate",
                }
                w.write(json.dumps(item) + "\n")
    print(f"Knowledge written to {out}")


if __name__ == "__main__":
    main()
