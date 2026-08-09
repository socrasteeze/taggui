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

## Correctness fixes

- Illustrious reorder no longer alphabetises within a category, so tagger
  confidence ordering survives it — and it now actually distinguishes
  character and series tags (see below)
- Trigger-token insert matches whole words: a `sks` trigger is no longer
  considered present because an image is tagged `masks`
- Failed caption writes are reported once per batch, naming the files that
  actually failed, instead of one dialog per image attributed to the wrong one
- Token counts subtract each encoder's real special-token overhead rather than
  assuming CLIP's two
- Vocabulary autocomplete works from the first character typed
- All Tags keeps its selection and scroll position across edits

## Speed / large datasets

- Thumbnail memory is bounded — previously one decoded thumbnail was retained
  per image for the lifetime of the directory
- Tag edits signal only the rows that changed, not everything between the
  first and last
- Tag re-ranking is coalesced, so a batch captioning run ranks once instead of
  once per image
- Filter captions are cached instead of rebuilt per term, per image, per
  keystroke
- The dimension cache is pruned instead of growing for the life of the install
- torch is no longer imported at startup just to populate the model list

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
- Per-profile token limit **and tokenizer** (CLIP 75 vs T5/Qwen 512).
  Tools → **Download Token Counter** fetches the profile's encoder; until then
  the count is shown as approximate (`~n / limit`) rather than silently using
  CLIP
- CSV vocab type-ahead (a1111-tagcomplete format): dataset tags first, then
  vocab; Tools → **Update Tag Lists** downloads Danbooru/e621 CSVs
- Autocomplete modes: dataset + vocab / dataset only / off
- Trigger-token insert (first tag or embedded in sentence)
- Illustrious tag reorder (count → character → series → general), using the
  downloaded Danbooru CSV's categories
- Move tags to front **or** back (Batch Reorder Tags)
- JoyCaption **ground on existing tags** toggle
- Image list **grid** view (View menu)
- Caption stats panel (Tools)
- Export JSONL and Kohya metadata JSON (File menu)

## Auto-captioning roster

- The list is split into recommended and legacy groups, with an unselectable
  heading between them. Legacy models stay fully runnable
- Qwen3-VL Instruct entries (2B / 4B / 8B / 30B-A3B), now actually loadable —
  `transformers` is pinned to 4.57.6, the first release that supports them
- Gemma 4 entry
- **pixai-tagger** (`deepghs/pixai-tagger-v0.9-onnx`) — a newer Danbooru
  tagger than WD v3, ~13.5k tags against ~10.8k, with better recall and newer
  character coverage. It uses the per-category thresholds the model ships
  with, since character tags need a much higher bar than general ones; the
  *Use the model's thresholds* setting turns that off
- Tagger batch size is a setting instead of a hardcoded 8
- A model needing a newer `transformers` says which version, instead of
  failing with an unrecognised-architecture error

## Dataset prep

- Tools → **Aspect Ratio Bucket Calculator** (kohya-compatible distribution)
- Optional **Process Images into Buckets** (backup to `original_images/`,
  resize+crop PNGs in place; directory load skips that backup folder)

## Tests

`pip install -r requirements-dev.txt && pytest` — headless, no GPU, no
captioning dependencies. CI runs it on every push.

## Still deferred / verify on hardware

- Loading any captioner against real weights, including Qwen3-VL end to end
  and the `trust_remote_code` models (Florence-2, Phi-3-Vision, Moondream)
- **Gemma 4 needs `transformers>=5.5`**, above the current pin, and reports
  that when selected. See the README for the tradeoff on either side
- pixai-tagger: everything but the weights is covered by tests. What is left
  is confirming its `preprocess.json` needs no preprocessing stage beyond the
  eight implemented — if it does, the error names the stage
- Optional `onnxruntime-gpu` / DirectML install (now documented in the README;
  CPU `onnxruntime` remains the default pin)
