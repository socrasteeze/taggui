# Fork changelog — socrasteeze/taggui

User-facing and developer notes for enhancements on top of upstream
[jhc13/taggui](https://github.com/jhc13/taggui) v1.34.0.

Roadmap / status tracking: [`Plan.md`](Plan.md).  
Merge map (every fork-only file and upstream touchpoint): [`FORK_NOTES.md`](FORK_NOTES.md).

---

## Windows launch

- **`run.bat`** — creates/updates a venv, installs `requirements.txt`, pins
  `HF_HOME` to `%USERPROFILE%\.cache\huggingface`, launches the app.
  - `run.bat update` / `-u` forces a reinstall.
  - Pins the venv to CPython 3.11/3.12 (torch wheel markers).
  - Writes `venv\installed-requirements.txt` only after a verified install;
    matching stamp **skips** pip on later launches (avoids re-checking the
    multi-GB torch URL wheel every time).
  - Migrates the legacy `venv\.installed-requirements.txt` stamp if present.
  - Pauses with a clear message on every failure path.
- **`create_shortcut.bat`** and **Tools → Create Desktop Shortcut…** — write a
  Windows `.lnk` to `run.bat` using `images/icon.ico` (Desktop and/or Start
  Menu).

## Speed / large datasets

- Async directory load with progress dialog
- Dimension disk cache (faster reopen)
- Background thumbnail generation (off UI thread; scaled decode)
- Sparse undo (per-image tag diffs, not full-dataset snapshots)
- Async `.txt` sidecar writes
- Debounced image filter and Find & Replace match counts
- Cached `tokens:` filter counts; viewer image cache on resize
- Lazy tokenizer load after startup
- Captioning input prefetch; WD tagger true input batching
- WD ONNX providers: CUDA / DirectML when available, CPU fallback
- Incremental All Tags counts; proper `beginResetModel` / `endResetModel`

## Caption profiles & tagging UX

- Caption profiles: SDXL (general), Illustrious XL, FLUX.2 Klein 9B,
  FLUX.1 Krea — Tools menu + Settings
- Per-profile token limit (CLIP 75 vs T5/Qwen 512)
- CSV vocab type-ahead (a1111-tagcomplete format): dataset tags first, then
  vocab; Tools → **Update Tag Lists** downloads Danbooru/e621 CSVs
- Autocomplete modes: dataset + vocab / dataset only / off
- Trigger-token insert (first tag or embedded in sentence)
- Illustrious tag reorder (count → character → series → general)
- Move tags to front **or** back (Batch Reorder Tags)
- JoyCaption **ground on existing tags** toggle
- Image list **grid** view (View menu)
- Caption stats panel (Tools)
- Export JSONL and Kohya metadata JSON (File menu)

## Auto-captioning roster

- Qwen3-VL Instruct entries (2B / 4B / 8B / 30B-A3B)
- JoyCaption Beta One kept front-and-center; Florence-2 / PromptGen; WD v3
- Older LLaVA / BLIP-era models still listed (legacy)

## Dataset prep

- Tools → **Aspect Ratio Bucket Calculator** (kohya-compatible distribution)
- Optional **Process Images into Buckets** (backup to `original_images/`,
  resize+crop PNGs in place; directory load skips that backup folder)

## Still deferred / verify on hardware

- Gemma 4 captioner + transformers bump for full Qwen3-VL validation
- Optional `onnxruntime-gpu` / DirectML install (documented in
  `requirements.txt`; CPU `onnxruntime` remains the default pin)
- pixai-tagger and remaining Plan.md polish items
