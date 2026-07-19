# FORK_NOTES — socrasteeze/taggui

This is a personal fork of [jhc13/taggui](https://github.com/jhc13/taggui).
This file is the always-current merge map: every place the fork diverges from
upstream, so `git merge upstream/main` stays a short job. Update it in the same
commit as any change that adds a new fork-only file or upstream touchpoint
(same convention as the sibling ai-toolkit and lora-dataset-studio forks).

`Plan.md` is the *roadmap* (captioning alignment to SDXL / Illustrious XL /
FLUX.2 Klein 9B / FLUX.1 Krea, captioner roster, token counters, bucketing —
with Done vs Deferred tracking). This file is the *divergence ledger* — what is
actually landed and where.

## Fork-only files (no upstream counterpart — merges never touch these)

- `Plan.md` — the modernization roadmap.
- `run.bat` — Windows bootstrap: creates/updates a venv, installs
  `requirements.txt`, pins `HF_HOME` to the local SSD cache, launches the app
  (`run.bat update` / `-u` reinstalls deps).
- `.gitattributes` — forces CRLF for `.bat` files.
- `taggui/utils/bucketing.py` — aspect-ratio bucketing math, compatible with
  kohya `make_bucket_resolutions` (step 64, target area, min/max resolution,
  `--bucket_no_upscale` equivalent).
- `taggui/dialogs/bucket_calculator_dialog.py` — the Tools ▸ Aspect Ratio
  Bucket Calculator dialog: distribution table with upscale / heavy-crop /
  sparse-bucket warnings, plus the optional **Process Images into Buckets**
  action (moves originals to `original_images/`, writes resized+cropped PNGs
  in place, carries `.txt` captions along).
- `FORK_NOTES.md` — this file.

## Upstream files with fork edits (merge conflicts concentrate here)

- `taggui/widgets/main_window.py` — two small insertions: the
  `show_bucket_calculator_dialog` slot (+ dialog import) and a new
  `Tools` menu with the calculator action. Everything else untouched.
- `taggui/models/image_list_model.py` — directory loading parallelised with a
  `ThreadPoolExecutor` (`_load_image` extracted); thumbnails downscaled during
  decode via `QImageReader.setScaledSize`; `BACKUP_DIRECTORY_NAME =
  'original_images'` excluded from loading (the bucket processor's backup
  folder must not be re-imported as images).
- `taggui/models/tag_counter_model.py` — incremental tag counting instead of a
  full recount on every change.
- `taggui/auto_captioning/models/wd_tagger.py` — ONNX execution providers
  chosen by device (`get_onnx_providers`: CUDA / DirectML with CPU fallback)
  instead of CPU-only.

## Behavioural notes for merges

- Upstream has no `Tools` menu; if upstream ever adds one, fold the calculator
  action into theirs rather than keeping two.
- The `original_images/` exclusion in `image_list_model.py` and the bucket
  processor in the dialog are a **pair** — dropping one side orphans the other.
- The bucket processor **rewrites images in place** (PNG) and is meant for
  pre-training prep only; trainers that bucket at load time (ai-toolkit) don't
  need it — treat the dialog as a calculator first (see the stack-wide
  integration plan in lora-dataset-studio's `PLAN.md`, Phase 3).

## Merge routine

```
git remote add upstream https://github.com/jhc13/taggui   # once
git fetch upstream && git merge upstream/main
# expected conflict surface: the four edited files above (fork side is small,
# well-delimited insertions). Re-run the app afterwards:
#   run.bat            (Windows)
#   python taggui/run_gui.py
# and sanity-check: directory load, tag editing, WD tagger on GPU,
# Tools ▸ Aspect Ratio Bucket Calculator.
```
