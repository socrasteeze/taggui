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
- `FORK_CHANGELOG.md` — user-facing summary of landed fork enhancements.
- `run.bat` — Windows bootstrap: creates/updates a venv, installs
  `requirements.txt`, pins `HF_HOME` to the local SSD cache, launches the app
  (`run.bat update` / `-u` forces a reinstall). Self-repairing: it pins the
  venv to CPython 3.11/3.12 (the only versions `requirements.txt` has Windows
  torch wheels for), records a `venv\installed-requirements.txt` stamp only
  after a *verified* install so an interrupted one is retried rather than
  silently launching a dependency-less venv, and pauses on every failure path
  so errors stay readable when launched by double-click. The stamp is not
  deleted until a replacement is written, and install is skipped when the
  stamp still matches `requirements.txt` (avoids re-checking multi-GB torch
  URL wheels on every launch).
- `create_shortcut.bat` — one-click Desktop `.lnk` creator (points at
  `run.bat`, uses `images/icon.ico`).
- `.gitattributes` — forces CRLF for `.bat` files.
- `taggui/utils/bucketing.py` — aspect-ratio bucketing math, compatible with
  kohya `make_bucket_resolutions` (step 64, target area, min/max resolution,
  `--bucket_no_upscale` equivalent).
- `taggui/dialogs/bucket_calculator_dialog.py` — the Tools ▸ Aspect Ratio
  Bucket Calculator dialog: distribution table with upscale / heavy-crop /
  sparse-bucket warnings, plus the optional **Process Images into Buckets**
  action (moves originals to `original_images/`, writes resized+cropped PNGs
  in place, carries `.txt` captions along).
- `taggui/utils/dimension_cache.py` — disk cache of image dimensions, pruned
  on save so it does not grow for the life of the install.
- `taggui/utils/tag_writer.py` — background queue for `.txt` sidecar writes.
  A `QObject`; reports failures through `errors_occurred` once the queue
  drains, so a failing batch is one dialog naming the files that failed.
- `taggui/utils/thumbnail_cache.py` — bounded LRU of decoded thumbnails, so
  memory tracks the viewport rather than the dataset.
- `taggui/utils/token_counting.py` — caption token counting; derives each
  tokenizer's special-token overhead instead of assuming CLIP's two.
- `taggui/utils/tokenizers.py` — per-profile tokenizers (CLIP bundled, T5 and
  Qwen3 downloaded into the app data directory), and the honest-fallback
  result type behind the `~n / limit` display.
- `taggui/auto_captioning/transformers_compat.py` — adapts to the installed
  transformers release: image-text auto class, dtype argument name, and the
  per-model minimum-version gate. **Read this first when bumping
  `transformers`.**
- `taggui/auto_captioning/models/gemma_4.py` — Gemma 4 captioner
  (needs `transformers>=5.5`, above the current pin).
- `taggui/auto_captioning/models/pixai_tagger.py` — the pixai-tagger
  captioning-model wrapper.
- `taggui/auto_captioning/models/pixai_tagger_model.py` — its ONNX inference,
  split out because none of it needs torch, which keeps it testable in CI.
- `taggui/auto_captioning/tag_utils.py` — `KAOMOJIS`, `get_onnx_providers`
  and `get_tags_to_exclude`, shared by both taggers. They used to live in
  `wd_tagger.py`, which cannot be imported without torch.
- `taggui/utils/onnx_preprocess.py` — interpreter for the `preprocess.json`
  that deepghs ONNX exports ship. **A mistake here yields wrong tags rather
  than an error**, so it is pinned against the reference implementation by
  `tools/differential_preprocess_check.py`.
- `tools/differential_preprocess_check.py` — that check. Not part of the
  pytest suite: it needs `dghs-imgutils`, whose dependency tree is larger than
  taggui's own. Run it in a throwaway environment after touching
  `onnx_preprocess.py`.
- `tests/`, `pytest.ini`, `requirements-dev.txt`,
  `.github/workflows/tests.yml` — headless test suite and CI.
- `taggui/utils/caption_profiles.py` — SDXL / Illustrious / FLUX caption profiles.
- `taggui/utils/tag_vocab.py` — a1111-format CSV vocab loader + merged completer.
- `taggui/utils/create_shortcut.py` — Windows `.lnk` helper (WScript.Shell).
- `taggui/dialogs/caption_stats_dialog.py` — caption / token / trigger stats.
- `taggui/dialogs/trigger_token_dialog.py` — insert trigger token tooling.
- `taggui/dialogs/create_shortcut_dialog.py` — Tools ▸ Create Desktop Shortcut.
- `taggui/auto_captioning/models/qwen3_vl.py` — Qwen3-VL captioner.
- `FORK_NOTES.md` — this file.

## Upstream files with fork edits (merge conflicts concentrate here)

- `taggui/widgets/main_window.py` — Tools menu (bucket calculator, caption
  stats, update tag lists, create desktop shortcut, caption profile submenu);
  File export actions; Edit trigger / Illustrious reorder; View list/grid;
  async directory load progress; debounced filter; lazy tokenizer load;
  vocab wiring.
- `taggui/models/image_list_model.py` — async directory load worker, dimension
  cache, background thumbnails via `ThumbnailCache`, sparse undo diffs, async
  tag writes with `write_errors_occurred`, trigger insert, Illustrious
  reorder, JSONL / Kohya metadata export, `emit_data_changed_for_rows`;
  `original_images/` exclusion.
- `taggui/models/proxy_image_list_model.py` — cached `tokens:` filter counts
  and cached joined captions; re-filters when the tokenizer arrives;
  `transformers` import is type-only.
- `taggui/models/tag_counter_model.py` — incremental tag counting; `tags`
  property for vocab completer; correct `beginResetModel` ordering; ranking
  coalesced onto a zero-delay timer (`publish_pending_counts`).
- `README.md` — fork banner, Windows `run.bat` install notes, links to
  `FORK_CHANGELOG.md`.
- `taggui/widgets/image_tags_editor.py` — profile token limits and the
  approximate-count marker; merged dataset+CSV autocomplete; `transformers`
  import is type-only.
- `taggui/widgets/image_viewer.py` — decoded-image cache on resize.
- `taggui/widgets/image_list.py` — list/grid view mode.
- `taggui/widgets/auto_captioner.py` — JoyCaption tag-grounding toggle;
  tagger batch size; unselectable roster headings. Routes by model id rather
  than importing model classes, keeping torch out of startup.
- `taggui/dialogs/settings_dialog.py` — caption profile + autocomplete mode.
- `taggui/dialogs/find_and_replace_dialog.py` — debounced match counts.
- `taggui/dialogs/batch_reorder_tags_dialog.py` — move tags to back;
  Illustrious order button, which emits `reorder_illustrious_requested` so the
  window can supply the vocabulary categories.
- `taggui/auto_captioning/captioning_thread.py` — prefetch, tagger batching
  with a configurable size, and the group-heading guard.
- `taggui/auto_captioning/auto_captioning_model.py` — lazily resolved auto
  class, compat dtype argument, `minimum_transformers_version` gate.
- `taggui/auto_captioning/models/wd_tagger.py` — batch generate; the ONNX
  provider and tag helpers moved to `tag_utils.py`.
- `taggui/auto_captioning/models/joycaption.py` — tag-grounded prompts.
- `taggui/auto_captioning/models/qwen3_vl.py` — declares its minimum
  transformers version.
- `taggui/auto_captioning/models_list.py` — `RECOMMENDED_MODELS` /
  `LEGACY_MODELS` groups; routing is data (`get_model_class_location`) with
  classes imported on demand.
- `taggui/utils/settings.py` — new defaults (profile, autocomplete mode, etc.).
- `taggui/utils/image.py` — `token_count` / `caption` caches and
  `invalidate_caches()`; the thumbnail moved to `ThumbnailCache`.
- `taggui/utils/tag_vocab.py` — `get_names_in_categories()` for the
  Illustrious reorder; one-character prefix suggestions.
- `requirements.txt` — `transformers==4.57.6`.
- `README.md` — GPU tagging, captioner version requirements, running tests.

## Behavioural notes for merges

- Upstream has no `Tools` menu; if upstream ever adds one, fold the calculator
  action into theirs rather than keeping two.
- The `original_images/` exclusion in `image_list_model.py` and the bucket
  processor in the dialog are a **pair** — dropping one side orphans the other.
- The bucket processor **rewrites images in place** (PNG) and is meant for
  pre-training prep only; trainers that bucket at load time (ai-toolkit) don't
  need it — treat the dialog as a calculator first (see the stack-wide
  integration plan in lora-dataset-studio's `PLAN.md`, Phase 3).
- Directory load is async: callers must wait for `load_finished` before
  selecting rows (main_window handles this).
- Undo history items are sparse `{index: previous_tags}` dicts, not full
  dataset snapshots.
- Anything that mutates `image.tags` must call `image.invalidate_caches()`;
  the token count and joined caption both derive from them. The four existing
  mutation paths all funnel through `_commit_changes`, `restore_history_tags`,
  `update_image_tags` or `add_tags`.
- `TagCounterModel` publishes its ranking on a zero-delay timer. The counts
  are correct immediately; `most_common_tags` catches up when the event loop
  runs, or on `publish_pending_counts()`. Tests call it explicitly.
- Model classes are imported on demand. Import `models_list` freely - it does
  not pull in torch - but `get_model_class()` does.
- The batched tagger route is selected by `is_tagger_model_id`, not by
  `isinstance(model, WdTagger)`. A new tagger must be added to
  `TAGGER_CLASS_NAMES` or it silently drops to one image at a time.
- pixai-tagger and the WD taggers share filenames but not preprocessing: RGB
  channel-first with ImageNet normalisation against BGR channel-last at raw
  0-255. Do not merge their image paths.
- Bumping `transformers` is a `requirements.txt` change plus a hardware pass;
  the code adapts through `transformers_compat.py`. Note
  `AutoModelForVision2Seq` does not exist in 5.x, and Gemma 4 does not exist
  below 5.5.

## Merge routine

```
git remote add upstream https://github.com/jhc13/taggui   # once
git fetch upstream && git merge upstream/main
# expected conflict surface: the edited upstream files above.
# Then, in order:
#   pytest             (headless; catches most merge damage in seconds)
#   run.bat            (Windows)
#   python taggui/run_gui.py
# and sanity-check: directory load (progress dialog), tag editing, WD tagger
# on GPU, Tools ▸ Aspect Ratio Bucket Calculator, caption profile, CSV
# type-ahead (Tools ▸ Update Tag Lists), export JSONL.
```
