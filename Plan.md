# TagGUI Modernization Plan

Goal: align TagGUI's captioning workflow with current (2025–2026) dataset-prep
practices for the models most in use today — **SDXL**, **Illustrious XL**,
**FLUX.2 Klein 9B**, and **FLUX.1 Krea [dev] / Krea** — and optimize the
hot paths that slow down large-dataset work.

Current state (this fork): based on v1.34.0 with the landed items in
**Implementation status** below and the user-facing summary in
[`FORK_CHANGELOG.md`](FORK_CHANGELOG.md). Upstream still ships CLIP-only
token counting and an older captioner roster; this fork adds profiles,
Qwen3-VL entries, CSV type-ahead, and large-dataset responsiveness work.

---

## Implementation status

Status here is checked against the code, and where a claim is testable it is
covered by `tests/` (run `pytest`; see [Verification](#verification)). Items
that cannot be verified without a GPU and model weights are called out as
**verify-on-hardware** rather than listed as done.

**Done (third pass — reconciliation, correctness, optimization):**
- ✅ Test suite + CI (`tests/`, `.github/workflows/tests.yml`) — headless,
  no torch, covers bucketing, caption profiles, tag vocab, dimension cache,
  tokenizers, the transformers shim, the roster, and the Qt models.
- ✅ transformers compatibility layer (`auto_captioning/transformers_compat.py`)
  — resolves the image-text auto class, the dtype load-argument name, and a
  per-model minimum version, so the pin selects what is *tested* rather than
  what the code is welded to. `AutoModelForVision2Seq` is gone in transformers
  5.x, so this is what keeps the captioner package importable there.
- ✅ transformers bumped to **4.57.6** (last 4.x; first release carrying
  `qwen3_vl` and `qwen3_vl_moe`). Qwen3-VL entries are now actually loadable.
- ✅ Per-profile tokenizers (`utils/tokenizers.py`) — cached like tag
  vocabularies, with Tools ▸ *Download Token Counter*. A fallback count is
  labelled approximate instead of passing CLIP off as T5/Qwen3.
- ✅ Illustrious ordering wired to real category data (a1111 `danbooru.csv`
  categories 4 character / 3 series), replacing the unused, series-blind WD
  helper. Ordering within a category is now stable, preserving tagger
  confidence order.
- ✅ Legacy roster demotion — two named groups with an unselectable heading.
- ✅ Gemma 4 and pixai-tagger entries; tagger batch size is a setting.
- ✅ pixai-tagger implemented against its real contract (2.1) — its own
  inference path, `preprocess.json` interpreter and per-category thresholds,
  rather than the WD image path, which would have produced wrong tags silently.
- ✅ Lazy model-class imports — torch is no longer loaded to draw a combo box.
- ✅ Correctness fixes: trigger-token whole-word matching, aggregated async
  write-error reporting, encoder-derived special-token overhead,
  `beginResetModel` ordering, one-character vocab autocomplete.
- ✅ Optimization: bounded thumbnail cache, contiguous-run `dataChanged`,
  coalesced tag-count publishing, cached filter captions, pruned dimension
  cache.

**Done (first pass — pure-code optimizations + bucket calculator):**
- ✅ WD tagger GPU inference (3.1) — `wd_tagger.py` now selects CUDA/DirectML
  ONNX providers based on the chosen device, falling back to CPU cleanly when
  `onnxruntime-gpu` isn't installed.
- ✅ Parallel directory loading (3.2) — `image_list_model.load_directory`
  reads dimensions/Exif/captions across a thread pool.
- ✅ Incremental tag counter (3.4) — `TagCounterModel.update_tag_counts`
  re-diffs only the changed rows instead of recounting every image per edit;
  verified equal to a full recount across edit/clear/batch/no-op cases.
- ✅ Thumbnail decode (3.4) — thumbnails are downsampled during decode via
  `QImageReader.setScaledSize` instead of decoding at full resolution.
- ✅ Aspect-ratio bucket calculator **and processor** (4) — new
  `utils/bucketing.py` (kohya-compatible; 1920×1080 → 1344×768 at 1024 area,
  now covered by `tests/test_bucketing.py`) plus a Tools ▸ *Aspect Ratio
  Bucket Calculator* dialog showing the
  bucket distribution and upscale/heavy-crop/sparse-bucket warnings. The
  *Process Images into Buckets* button moves every original into an
  `original_images` backup folder (preserving subfolder structure) and writes
  a resized + center-cropped PNG in its place, on a background thread with a
  progress bar. Directory loading now skips `original_images/` so backups are
  never reloaded or re-processed. End-to-end tested on real files including
  subfolders, transparency flattening, and `foo.jpg`/`foo.png` name-collision
  disambiguation with caption copying.

**Done (second pass — responsiveness + caption workflow):**
- ✅ Async directory load with progress dialog + dimension cache
- ✅ Background thumbnail pool (decode off UI thread)
- ✅ Sparse undo per-image diffs + async `.txt` writes
- ✅ Debounced filter / Find & Replace; cached `tokens:`; viewer image cache;
  lazy tokenizer load
- ✅ Captioning prefetch + WD true input batching
- ✅ Caption profiles + per-encoder token limits; CSV vocab type-ahead
  (a1111 format, Tools ▸ Update Tag Lists) — the matching *tokenizers* landed
  in the third pass
- ✅ Qwen3-VL model entries + JoyCaption tag-grounding toggle — the entries
  only became loadable with the third pass's transformers bump
- ✅ Trigger-token tooling + Illustrious reorder — both had defects fixed in
  the third pass
- ✅ Grid view, JSONL / Kohya metadata export, caption stats panel
- ✅ Desktop / Start Menu shortcut creator (`create_shortcut.bat` +
  Tools ▸ Create Desktop Shortcut…)
- ✅ `run.bat` durable install stamp (`venv\installed-requirements.txt`) so
  matching requirements skip pip on later launches; legacy dotted stamp
  migrated automatically
- ✅ Tag-counter `beginResetModel` / `endResetModel` pairing (fixes proxy
  “endResetModel without beginResetModel” warnings)

**Verify-on-hardware (code is in; real weights have never been loaded):**
- Every captioner run under transformers 4.57.6 — Qwen3-VL end to end, and
  Florence-2 / Phi-3-Vision / Moondream, which use `trust_remote_code` and are
  the usual breakage point after a bump.
- **Gemma 4** needs `transformers>=5.5`, so it cannot run on the 4.57.6 pin.
  The entry is present and reports that until the pin is raised. Raising it is
  a real tradeoff: on 5.x, Florence-2 becomes natively supported (no remote
  code), while Phi-3-Vision and Moondream still depend on remote code written
  against the 4.x API.
- **pixai-tagger-v0.9** — the file layout and inference contract are now known
  (see §2.1), and everything except the weights themselves is covered by
  tests. What is unverified is only whether the real `preprocess.json` uses a
  stage outside the eight implemented; if it does, the error names the stage.
- WD true batching against a real ONNX export — the batch size is now a
  setting, clamped when the model's batch dimension is static.
- Optional `onnxruntime-gpu` / DirectML install remains a user step, now
  documented in the README.

**Known non-goals for now:**
- transformers 5.x as the pinned default. The shim already supports it; what
  is missing is a hardware pass over the remote-code captioners.

See also [`FORK_CHANGELOG.md`](FORK_CHANGELOG.md) for the user-facing summary.

## Verification

```
pip install -r requirements-dev.txt
pytest
```

Runs headless (offscreen Qt), needs no GPU and no torch, and covers the
pure-logic modules, the Qt models and the tagger inference paths (with a faked
ONNX session). CI runs the same suite on every push. Anything involving real
model weights is verify-on-hardware and is listed as such above rather than
being claimed as done.

`utils/onnx_preprocess.py` reimplements deepghs's preprocessing spec, where a
mistake yields wrong tags rather than an error. It was checked by running both
implementations over the same images and pipelines and comparing the arrays
exactly — 53 comparisons, no divergence, against dghs-imgutils 0.19.0. Re-run
`tools/differential_preprocess_check.py` after changing that module; it is not
part of the suite because the reference's dependency tree is larger than
taggui's own.

---

## 1. Per-target-model caption alignment

The single biggest gap: TagGUI treats all captions identically, but the four
target models want different caption shapes.

### 1.1 Caption profiles (new feature)
Add a selectable **caption profile** (per-directory setting) that adjusts token
counting, autocompletion behavior, and captioning presets:

| Profile | Style | Token budget | Notes |
|---|---|---|---|
| SDXL (general) | Short NL or hybrid tags+NL | 75 (CLIP chunk) | Trigger word first; caption dropout handled by trainer |
| Illustrious XL | Danbooru tags | 75 per CLIP chunk | Tag order: count (`1girl`) → character → series → general; optional quality/rating tags |
| FLUX.2 Klein 9B | Rich natural-language sentences/paragraph | 512 (Qwen3 embedder) | Style LoRAs: describe content only, never the style; trigger = rare made-up token embedded in the sentence |
| FLUX.1 Krea [dev] | 1–3 descriptive NL sentences | 512 (T5) | Standard FLUX.1 practice; captionless runs are a valid alternative for single-concept LoRAs |

### 1.2 Token counter per encoder — done
Both the limit and the tokenizer follow the caption profile: CLIP (75) for
SDXL/Illustrious, T5 (512) for FLUX.1/Krea, Qwen3 (512) for FLUX.2 Klein.
CLIP ships with the app; the others are fetched once into the app data
directory (Tools ▸ *Download Token Counter*) and cached. Until one is
available the count is shown as `~n / limit` with a tooltip, rather than
presenting a CLIP number as if it came from the profile's encoder. The
special-token overhead is measured from each tokenizer instead of assuming
CLIP's two.

### 1.3 Trigger-token tooling — done
- "Insert trigger token" batch action with two placement modes: **first tag**
  (SDXL/Illustrious, pairs with kohya `keep_tokens`) and **embedded in
  sentence** (FLUX-family). An image already carrying the trigger is skipped
  on a whole-word match, so a trigger like `sks` is no longer considered
  present because the image is tagged `masks`.
- Still open: dataset-wide trigger consistency as a filter term (images
  missing the trigger). The caption stats panel reports the percentage.

### 1.4 Illustrious tag-order support — done
- Batch reorder implementing the booru convention: count tag → character →
  series → general. Categories come from the a1111 `danbooru.csv` the fork
  already downloads (4 character, 3 copyright/series) — **not** from the WD
  tagger, which has no series category at all. Order within a category is
  preserved, so tagger confidence ordering survives the reorder. With no tag
  list downloaded, the action says what it will not be able to do.
- Still open: optional prepend/strip of quality (`masterpiece, best quality`)
  and rating (`safe`/`sensitive`/`nsfw`/`explicit`) tags — guides differ on
  whether to include these in training captions, so make it a toggle, off by
  default.

## 2. Auto-captioning model roster

### 2.1 Add — all listed; runtime verification varies
- ✅ **Qwen3-VL Instruct (2B/4B/8B, and 30B-A3B)** — the current community
  favorite for NL captions. The small dense variants cover low-VRAM setups;
  **Qwen3-VL-30B-A3B** (MoE, 30B total / 3B active per token) is the quality
  pick — near-flagship captions at moderate inference cost, and it quantizes
  well (4-bit fits in ~20 GB). Runnable as of the 4.57.6 pin; a real run is
  verify-on-hardware.
- ⚠️ **Gemma 4** — listed (`gemma-4-31b-it`, `gemma-4-e4b-it`) with a 4-bit
  path, but its architecture only exists in `transformers>=5.5`, above the
  current pin. Selecting it reports that rather than failing obscurely.
- ✅ **pixai-tagger-v0.9** — newer Danbooru snapshot than WD v3 (~13.5k tags
  against ~10.8k), better recall and newer character coverage; complements
  wd-eva02-large-tagger-v3.

  The entry points at **`deepghs/pixai-tagger-v0.9-onnx`**, not `pixai-labs`,
  which publishes PyTorch weights with no `model.onnx`. Despite sharing
  filenames with the WD exports this is not a WD tagger — it takes channel-
  first RGB with ImageNet normalisation where WD takes channel-last BGR at raw
  0–255 — so it has its own implementation rather than subclassing `WdTagger`.
  Preprocessing is read from the export's own `preprocess.json` by
  `utils/onnx_preprocess.py`, whose output was checked against the reference
  implementation (see [Verification](#verification)). Per-category thresholds
  come from the export's `thresholds.csv`, since characters want a much higher
  bar than general tags; the *Use the model's thresholds* setting turns that
  off in favour of a single value.
- ✅ **JoyCaption tag-grounded mode** — the image's existing tags are wired
  into the JoyCaption prompt behind a toggle.

### 2.2 Deprecate / demote — done
LLaVA-1.5, BakLLaVA, InstructBLIP, BLIP-2, Kosmos-2, Moondream 1 and the WD v2
taggers sit in a `LEGACY_MODELS` group below an unselectable heading in the
model list. Still fully runnable, just not promoted.

### 2.3 Keep front and center
JoyCaption Beta One (watch for v1.0), Florence-2 / PromptGen (fast low-VRAM
option), wd-eva02-large-tagger-v3 (still the booru-tagging accuracy benchmark),
wd-vit-large-tagger-v3 (recall-leaning alternative).

## 3. Optimizations

### 3.1 WD tagger runs CPU-only (high impact, small change)
`auto_captioning/models/wd_tagger.py:40` creates
`InferenceSession(model_path)` with no providers, and `requirements.txt` pins
CPU `onnxruntime`. Batch-tagging thousands of images runs entirely on CPU
even on CUDA machines.
- Switch to `onnxruntime-gpu` (or `-directml` on Windows without CUDA) and
  pass `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`,
  respecting the existing device setting.
- Batch inputs (the WD models accept batched tensors) instead of per-image
  session runs.

### 3.2 Directory loading is sequential (high impact for large datasets)
`models/image_list_model.py:load_directory` walks every image on one thread,
calling `imagesize.get()` plus an `exifread` file-open per image. On a
50k-image dataset over spinning disk/NAS this takes minutes.
- Read dimensions + EXIF orientation in a thread pool
  (`concurrent.futures`), then populate the model in one batch.
- Skip the exifread pass for formats that can't carry EXIF (PNG w/o eXIf,
  WebP variants) — cheap magic-byte check first.
- Optional: cache `(mtime, size, dimensions)` per directory to make reopening
  instant.

### 3.3 Captioning throughput
- ~~Keep models loaded between batch runs when settings are unchanged~~ —
  already handled by `AutoCaptioningModel.load_processor_and_model`, which
  reuses the processor and model when the id, device and 4-bit flag match.
  This was listed as outstanding but had never been missing.
- ✅ Tagger batch size is a setting, clamped to the ONNX model's batch
  dimension. True batching for the VLM captioners is still per-image plus
  prefetch.
- ✅ bfloat16 and the 4-bit bitsandbytes path live on the base captioning
  model, so Qwen3-VL and Gemma 4 inherit them; both declare `bfloat16` and
  effectively need 4-bit on consumer GPUs (30B-A3B ≈ 20 GB, Gemma 4 31B
  ≈ 18–20 GB at 4-bit). Verify-on-hardware.

### 3.4 UI responsiveness on large datasets

**Landed in the first two passes:** incremental tag counter; sparse undo
diffs; scaled thumbnail decode on a background pool; cached `tokens:` counts;
async `.txt` writes; debounced filter and Find & Replace; image-viewer decode
cache; captioning prefetch; lazy tokenizer load.

**Landed in the third pass:**
- **Thumbnail memory was unbounded** — a decoded thumbnail was kept on every
  `Image` for the lifetime of the directory, so memory tracked the dataset
  rather than the viewport (roughly gigabytes across a 50k-image set at the
  default width). Now a bounded LRU keyed by path.
- **`dataChanged` spanned first-to-last changed row** — an edit touching rows
  0 and 9999 marked ten thousand rows as changed, and the tag counter diffed
  all of them. Now emitted as contiguous runs.
- **Tag counting re-ranked every tag per edit** — plus a full `Counter` copy
  via unary plus, plus a view reset that discarded the All Tags selection and
  scroll position. Batch captioning reports one row at a time, so that was one
  full ranking pass per image. Zero-count tags are dropped directly and
  ranking is coalesced onto a zero-delay timer.
- **The filter rebuilt each caption once per term** — the claim that this was
  fixed in pass two was not true of the code. Now cached on the `Image` and
  invalidated with the token count.
- **The dimension cache grew forever** — entries are keyed by path, mtime and
  size, so every edited image left its old entry behind. Pruned on save above
  a threshold.
- **torch loaded at startup to draw a combo box** — the roster imported every
  model class at module import. Now imported on demand.

**Still open:**
- True batching for the VLM captioners (currently per-image plus prefetch).
- The `original_images/` scan in `get_file_paths` walks the whole tree before
  filtering; fine for local disks, worth revisiting for network shares.

### 3.5 Dependency refresh — done, with one open decision

Pinned at **`transformers==4.57.6`**, the last 4.x release and the first to
carry `qwen3_vl` / `qwen3_vl_moe`. The 4.x line ended there in January 2026;
the current line is 5.x.

The app no longer depends on a single API shape — `transformers_compat.py`
resolves the auto class and the dtype argument name, and gates models on a
declared minimum version — so moving the pin is a one-line change plus a
hardware pass, not a code migration.

Open decision: **whether to pin 5.x.** For it — Gemma 4 needs `>=5.5`, and
Florence-2 becomes natively supported. Against — Phi-3-Vision and Moondream
still load remote code written against the 4.x API, and neither has been run
here. Verified against the real libraries: `AutoModelForVision2Seq` is removed
in 5.x, while `llava`, `llava_next`, `instructblip`, `blip-2`, `kosmos-2` and
`florence2` are all natively registered.

`flash-attn` wheels stay pinned to the current torch 2.8 build; revisit
alongside any torch bump.

## 4. Quality-of-life aligned with current workflows

- **Export presets**: kohya/OneTrainer both consume `.txt` sidecars (already
  supported); add optional JSONL export (`{"file_name": ..., "text": ...}`)
  for HF `datasets`/diffusers Dreambooth scripts used by FLUX.2 Klein
  training examples.
- **Aspect-ratio bucket calculator** (new): replicate the kohya/OneTrainer
  bucketing algorithm so users can see how their dataset will bucket *before*
  training:
  - Inputs: target resolution area (default 1024², plus 512²/1536² presets),
    bucket step (64 px, kohya `--bucket_reso_steps`), min/max resolution
    (256–2048 defaults, kohya `--min_bucket_reso`/`--max_bucket_reso`), and
    an upscaling toggle (`--bucket_no_upscale` equivalent).
  - Per image: compute the assigned bucket (nearest aspect ratio at the
    target area, dimensions snapped to the step) and the resulting
    resize/crop, exactly as kohya's `make_bucket_resolutions` does.
  - Dataset view: bucket distribution table (bucket → image count) so users
    can spot lonely buckets (batch-of-1 buckets hurt training) and
    over-cropped images.
  - Filters/warnings: `bucket:WxH` filter term; flag images that would be
    upscaled (source below bucket size), cropped more than a threshold %, or
    below the target area entirely (Klein wants ≥1024 long edge).
  - Sidecar-free: purely a calculator/report — no image modification —
    matching what kohya and OneTrainer will do at train time.
- **Caption stats panel**: distribution of token counts per active encoder,
  % images containing the trigger token, tag frequency (exists) — helps spot
  over/under-captioning before training.

## 5. What is left

Items 1–8 of the original order of work are done; the sections above record
what landed where. What remains:

1. **A hardware pass.** Load each captioner under transformers 4.57.6 on a
   real GPU — Qwen3-VL end to end, and Florence-2 / Phi-3-Vision / Moondream,
   which use `trust_remote_code`. Nothing else on this list should move until
   the current pin is known-good.
2. **Decide on transformers 5.x** (3.5). Gemma 4 needs it; Phi-3-Vision and
   Moondream are the risk. The compat layer means this is a pin change plus
   verification, not a migration.
3. **Tag one image with pixai-tagger.** Everything but the weights is tested;
   what remains is confirming the real `preprocess.json` needs no stage beyond
   the eight implemented, and that the tags come out sensible.
4. **Trigger consistency as a filter term** (1.3) and the optional
   quality/rating tag toggle (1.4).
5. **True VLM batching** (3.3) — the tagger batches; the VLM path prefetches
   but still generates one image at a time.
6. `bucket:WxH` as a filter term (4) — the calculator reports the
   distribution, but buckets are not filterable from the image list.

---

*Research notes: FLUX.2 Klein 9B uses a Qwen3 (8B) text embedder — not
Mistral, which only the 32B FLUX.2 [dev] uses — with a 512-token prompt
window; BFL's Klein LoRA docs recommend natural-sentence captions, content-only
captions for style LoRAs, and rare made-up trigger tokens. Krea has published
no training guidance for its closed "Krea" hosted models; FLUX.1 Krea [dev]
follows standard FLUX.1 LoRA practice. Illustrious v2.0 is Onoma's recommended
fine-tuning base and accepts both tags and natural language.*
