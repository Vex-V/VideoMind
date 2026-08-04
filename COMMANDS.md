# Commands

## Run the server

```bash
python serve.py
```

Opens on <http://127.0.0.1:8077> — upload and query UI, plus `/docs` for the API.

```bash
python serve.py --port 9000        # different port
python serve.py --reload           # auto-restart on code changes
python serve.py --host 0.0.0.0     # reachable from other machines on the LAN
```

Equivalent, if you prefer invoking uvicorn directly:

```bash
python -m uvicorn videomind.api.app:app --port 8077
```

Both are quiet: `USE_TF=0` is set before anything imports `transformers`, which
otherwise pulls in TensorFlow (nothing here uses it) along with several seconds
of startup and a wall of oneDNN/absl banners.

### Port already in use

`pkill` does not reliably kill Python on Windows. Free the port with PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8077 -State Listen |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

A server that fails to bind exits quietly, so requests keep hitting the *old*
process and you appear to be running stale code. Check the port first when
changes seem not to apply.

## Clear the database

Vectors and per-video records:

```bash
rm -rf vectordb records
```

Recreated automatically on the next upload. This drops all indexed vectors and
every video's record, so nothing shows in the UI afterwards.

Also remove uploaded source videos:

```bash
rm -rf vectordb records uploads
```

PowerShell equivalent:

```powershell
Remove-Item -Recurse -Force vectordb, records
```

### Clear one video instead of everything

```python
from videomind.vectordb import ChunkStore
cs = ChunkStore()
cs.delete_video("95e110e25070fcfc")                        # every chunking
cs.delete_video("95e110e25070fcfc", chunk_config="interval:10")  # just one
cs.close()
```

Re-uploading the same file with the same chunking already replaces its vectors,
so this is only needed to remove a video outright.

## Inspect what was produced

Each upload writes `records/<name>__<id8>.json` with every analyzer's output per
chunk, including `_frames` (the timestamps the sampler chose). Useful for
checking why a description says what it does.

```bash
ls records/
python -c "import json,glob; d=json.load(open(glob.glob('records/*.json')[0])); print(json.dumps(d['chunks'][0], indent=2))"
```

Vector store contents:

```python
from videomind.vectordb import ChunkStore
from videomind.vectordb.store import COLLECTION
cs = ChunkStore()
print(cs.count())
pts, _ = cs.client.scroll(COLLECTION, limit=100, with_payload=True)
for p in pts[:5]:
    print(p.payload["video_id"], p.payload["extractor_id"], p.payload["start"], p.payload["end"])
cs.close()
```

## Rebuild vectors without re-running analyzers

Re-embeds from saved records — no OpenAI calls, so it costs nothing. Use after
changing the embedding model or the render logic.

```bash
python reindex.py
```
