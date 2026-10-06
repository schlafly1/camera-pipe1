#!/usr/bin/env python3
"""Re-embed vision_events_v2 -> vision_events_v3 with nomic task prefixes.

Copies ids/documents/metadatas from SOURCE and re-embeds each document as
`search_document: {text}` into DEST (cosine space). Idempotent/resumable via
a JSONL state file of finished source ids.

    cd ~/sd/camera-pipe1
    CHROMA_NO_NATIVE=1 setsid nohup .venv/bin/python3 -u tools/reembed_nomic_prefixes.py \
        > logs/reembed_nomic.out 2>&1 < /dev/null &

Does NOT touch the live pipeline or CHROMA_COLLECTION. Live writes stay on v2.
"""
import argparse
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
if os.path.exists(".env"):
    for line in open(".env"):
        line = line.rstrip("\r\n")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

import chromadb
import ollama

STOP = False


def _handle_stop(signum, frame):
    global STOP
    STOP = True
    print(f"[reembed] signal {signum}; stopping after current batch", flush=True)


def load_env_host():
    return (os.environ.get("EMBED_HOST") or os.environ.get("OLLAMA_HOST") or "").strip() or None


def get_all_ids(col, page=5000):
    ids, off = [], 0
    while True:
        res = col.get(include=[], limit=page, offset=off)
        if not res["ids"]:
            break
        ids.extend(res["ids"])
        off += len(res["ids"])
    return ids


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in ("ok", "skip"):
                done.add(rec["id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="vision_events_v2")
    ap.add_argument("--dest", default="vision_events_v3")
    ap.add_argument("--doc-prefix", default="search_document: ")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--state", default="logs/reembed_nomic_state.jsonl")
    ap.add_argument("--status", default="logs/reembed_nomic_status.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    os.makedirs("logs", exist_ok=True)
    client = chromadb.HttpClient(host=os.environ.get("CHROMADB_HOST", "localhost"),
                                 port=int(os.environ.get("CHROMADB_PORT", "8000")))
    src = client.get_collection(args.source)
    if args.dry_run:
        dst = None
    else:
        try:
            dst = client.get_collection(args.dest)
            print(f"[reembed] dest exists: {args.dest} count={dst.count()} meta={dst.metadata}",
                  flush=True)
        except Exception:
            dst = client.create_collection(
                args.dest, metadata={"hnsw:space": "cosine",
                                     "embed_prefix_style": "nomic",
                                     "embed_doc_prefix": args.doc_prefix.strip(),
                                     "source": args.source})
            print(f"[reembed] created {args.dest} cosine meta={dst.metadata}", flush=True)

    embed_host = load_env_host()
    embed = ollama.Client(host=embed_host, timeout=60)
    print(f"[reembed] source={args.source} ({src.count()}) -> dest={args.dest} "
          f"prefix={args.doc_prefix!r} embed={args.embed_model} @ {embed_host}", flush=True)

    done = load_done(args.state)
    print(f"[reembed] already done: {len(done)}", flush=True)

    all_ids = get_all_ids(src)
    todo = [i for i in all_ids if i not in done]
    print(f"[reembed] todo: {len(todo)} / {len(all_ids)}", flush=True)

    t0 = time.time()
    ok = err = 0
    batch = args.batch

    with open(args.state, "a") as state_f:
        for i in range(0, len(todo), batch):
            if STOP:
                break
            chunk_ids = todo[i:i + batch]
            res = src.get(ids=chunk_ids, include=["documents", "metadatas"])
            # Preserve order of chunk_ids
            by_id = {rid: (doc, meta) for rid, doc, meta in
                     zip(res["ids"], res["documents"], res["metadatas"])}
            prompts, use_ids, use_docs, use_metas = [], [], [], []
            for rid in chunk_ids:
                if rid not in by_id:
                    state_f.write(json.dumps({"id": rid, "status": "skip", "why": "missing"}) + "\n")
                    continue
                doc, meta = by_id[rid]
                if doc is None:
                    doc = ""
                prompts.append(args.doc_prefix + doc)
                use_ids.append(rid)
                use_docs.append(doc)
                use_metas.append(meta or {})

            if not use_ids:
                continue

            try:
                # Prefer batch /api/embed via ollama embed API if available;
                # fall back to one-at-a-time embeddings().
                embeddings = []
                if hasattr(embed, "embed"):
                    # ollama-python newer: embed(model=, input=)
                    try:
                        resp = embed.embed(model=args.embed_model, input=prompts)
                        embeddings = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
                    except Exception:
                        embeddings = []
                if not embeddings:
                    for p in prompts:
                        r = embed.embeddings(model=args.embed_model, prompt=p)
                        embeddings.append(r["embedding"])
                if len(embeddings) != len(use_ids):
                    raise RuntimeError(f"embed count {len(embeddings)} != {len(use_ids)}")
                if not args.dry_run:
                    dst.upsert(ids=use_ids, embeddings=embeddings,
                               documents=use_docs, metadatas=use_metas)
                for rid in use_ids:
                    state_f.write(json.dumps({"id": rid, "status": "ok"}) + "\n")
                state_f.flush()
                ok += len(use_ids)
            except Exception as e:
                err += 1
                print(f"[reembed] batch error at {i}: {type(e).__name__}: {e}", flush=True)
                # fall back per-id so one bad row doesn't kill the batch progress
                for rid, doc, meta, prompt in zip(use_ids, use_docs, use_metas, prompts):
                    if STOP:
                        break
                    try:
                        r = embed.embeddings(model=args.embed_model, prompt=prompt)
                        if not args.dry_run:
                            dst.upsert(ids=[rid], embeddings=[r["embedding"]],
                                       documents=[doc], metadatas=[meta])
                        state_f.write(json.dumps({"id": rid, "status": "ok"}) + "\n")
                        state_f.flush()
                        ok += 1
                    except Exception as e2:
                        state_f.write(json.dumps({"id": rid, "status": "error",
                                                  "error": str(e2)[:200]}) + "\n")
                        state_f.flush()
                        err += 1

            done_n = len(done) + ok
            elapsed = time.time() - t0
            rate = ok / elapsed if elapsed > 0 else 0
            left = len(todo) - (i + len(chunk_ids))
            eta = left / rate if rate > 0 else None
            status = {
                "source": args.source, "dest": args.dest,
                "ok": ok, "err": err, "todo": len(todo),
                "src_count": len(all_ids),
                "dst_count": None if args.dry_run else dst.count(),
                "rate_per_s": round(rate, 2),
                "elapsed_s": round(elapsed, 1),
                "eta_s": None if eta is None else round(eta, 1),
                "stopped": STOP,
            }
            with open(args.status, "w") as sf:
                json.dump(status, sf, indent=2)
            if (i // batch) % 10 == 0 or left <= 0:
                print(f"[reembed] ok={ok} err={err} left={left} "
                      f"rate={rate:.1f}/s eta={status['eta_s']}s "
                      f"dst={status['dst_count']}", flush=True)

    final = None if args.dry_run else dst.count()
    print(f"[reembed] done ok={ok} err={err} dest_count={final} stopped={STOP}", flush=True)


if __name__ == "__main__":
    main()
