# TagGUI Modernization Plan

Goal: align TagGUI's captioning workflow with current (2025–2026) dataset-prep
practices for the models most in use today — **SDXL**, **Illustrious XL**,
**FLUX.2 Klein 9B**, and **FLUX.1 Krea [dev] / Krea** — and optimize the
hot paths that slow down large-dataset work.

Current state: v1.34.0, PySide6 app, `.txt` sidecar captions, CLIP-only
75-token counter, captioner roster centered on JoyCaption Beta One, Florence-2,
WD Tagger v2/v3, LLaVA-era VLMs, BLIP-2/InstructBLIP/Kosmos-2.

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

### 1.2 Token counter per encoder
`widgets/image_tags_editor.py` hardcodes `MAX_TOKEN_COUNT = 75` with the CLIP
ViT-B/32 tokenizer. Make the limit and tokenizer follow the caption profile:
- CLIP (75) for SDXL/Illustrious.
- T5 (512) for FLUX.1/Krea.
- Qwen3 (512) for FLUX.2 Klein.
Show `n / limit` and color-code accordingly; keep CLIP as the default.

### 1.3 Trigger-token tooling
- "Insert trigger token" batch action with two placement modes: **first tag**
  (SDXL/Illustrious, pairs with kohya `keep_tokens`) and **embedded in
  sentence** (FLUX-family).
- Validate trigger consistency across the dataset (filterable: images missing
  the trigger).

### 1.4 Illustrious tag-order support
- Batch reorder mode implementing the booru convention: count tag → character
  → series → general tags, using the WD tagger's category metadata (it already
  distinguishes rating/character/general).
- Optional prepend/strip of quality (`masterpiece, best quality`) and rating
  (`safe`/`sensitive`/`nsfw`/`explicit`) tags — guides differ on whether to
  include these in training captions, so make it a toggle, off by default.

## 2. Auto-captioning model roster

### 2.1 Add
- **Qwen3-VL Instruct (2B/4B/8B, and 30B-A3B)** — the current community
  favorite for NL captions. The small dense variants cover low-VRAM setups;
  **Qwen3-VL-30B-A3B** (MoE, 30B total / 3B active per token) is the quality
  pick — near-flagship captions at moderate inference cost, and it quantizes
  well (4-bit fits in ~20 GB). Highest-value addition.
- **Gemma 4 31B IT** (Google, Mar 2026, Apache 2.0) — flagship dense
  open-weights VLM built from Gemini 3 research; strong detailed captioning
  with variable aspect-ratio image input. Heavier than Qwen3-VL-30B-A3B (all
  31B params are dense), so it wants 4-bit on consumer GPUs; offer it
  alongside Qwen3-VL as the two "high quality" captioners. The smaller
  Gemma 4 E4B is a candidate for the low-VRAM tier.
- **pixai-tagger-v0.9** — newer Danbooru snapshot than WD v3, better recall
  and newer character coverage; complements wd-eva02-large-tagger-v3.
- **JoyCaption tag-grounded mode** — Beta One accepts WD tags as input to
  ground its NL caption. TagGUI already has both pieces; wire the image's
  existing tags into the JoyCaption prompt as an option. This is the current
  best hybrid tags+NL mechanism.

### 2.2 Deprecate / demote
Move to a "legacy" section (still runnable, not promoted): LLaVA-1.5,
BakLLaVA, InstructBLIP, BLIP-2, Kosmos-2, Moondream 1, WD v2 taggers.
These are 2023–2024-era quality and no longer competitive.

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
- Keep models loaded between batch runs when settings are unchanged (avoid
  reload per invocation).
- Expose batch size for VLM captioners where the backend supports it.
- Prefer `dtype=bfloat16` + FlashAttention where already installed; 4-bit
  (bitsandbytes) already exists for JoyCaption — extend it to Qwen3-VL and
  Gemma 4, where it's effectively required on consumer GPUs (30B-A3B ≈ 20 GB,
  Gemma 4 31B ≈ 18–20 GB at 4-bit).

### 3.4 Dependency refresh
- `transformers==4.48.3` (early 2025) is too old for Qwen3-VL — bump to a
  current 4.5x release and re-verify each existing captioner (Florence-2 is
  the usual breakage point; pin `trust_remote_code` versions).
- Revisit `flash-attn` wheels after the torch bump this implies.

## 4. Quality-of-life aligned with current workflows

- **Export presets**: kohya/OneTrainer both consume `.txt` sidecars (already
  supported); add optional JSONL export (`{"file_name": ..., "text": ...}`)
  for HF `datasets`/diffusers Dreambooth scripts used by FLUX.2 Klein
  training examples.
- **Resolution audit**: filterable warning for images under the target
  bucket area (1024² for SDXL/Illustrious/FLUX; Klein wants ≥1024 long edge).
- **Caption stats panel**: distribution of token counts per active encoder,
  % images containing the trigger token, tag frequency (exists) — helps spot
  over/under-captioning before training.

## 5. Suggested order of work

1. WD tagger GPU + batching (3.1) — small diff, immediate payoff.
2. Parallel directory loading (3.2).
3. Caption profiles + per-encoder token counter (1.1, 1.2).
4. Qwen3-VL (incl. 30B-A3B) + Gemma 4 + transformers bump (2.1, 3.4) — do
   together; both need a current transformers release.
5. JoyCaption tag-grounding, pixai-tagger, trigger tooling, Illustrious
   reorder (2.1, 1.3, 1.4).
6. Legacy demotion, export presets, stats panel (2.2, 4).

---

*Research notes: FLUX.2 Klein 9B uses a Qwen3 (8B) text embedder — not
Mistral, which only the 32B FLUX.2 [dev] uses — with a 512-token prompt
window; BFL's Klein LoRA docs recommend natural-sentence captions, content-only
captions for style LoRAs, and rare made-up trigger tokens. Krea has published
no training guidance for its closed "Krea" hosted models; FLUX.1 Krea [dev]
follows standard FLUX.1 LoRA practice. Illustrious v2.0 is Onoma's recommended
fine-tuning base and accepts both tags and natural language.*
