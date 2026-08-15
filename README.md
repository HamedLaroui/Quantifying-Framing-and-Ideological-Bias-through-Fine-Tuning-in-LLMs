# Narrative Framing in LLMs Fine-tuned on Israeli/Palestinian News Outlets

This repo covers the full pipeline for a study comparing how Gemma 3-12B's framing of the
Israel-Palestine conflict shifts after fine-tuning on four news outlets with different editorial
perspectives: **WAFA**, **PNN**, **The Jerusalem Post**, and **Israel National News**. It has three
stages: scrape, fine-tune, analyze; each covered by notebooks in this repo.

## Pipeline overview

```
news_scraper.ipynb
        │
        ▼
  four outlet CSVs (wafa_raw_articles.csv, pnn_raw_articles.csv,
                     jerusalem_post_articles.csv, israelnationalnews_articles.csv)
        │
        ▼
  gemma3_WAFA_training.ipynb  ─┐
  gemma3_PNN_training.ipynb   ─┤  each produces one LoRA adapter,
  gemma3_12B_JPost_training.ipynb ─┤  saved to its own Google Drive folder
  gemma3_INN_training.ipynb ─┘
        │
        ▼
  gemma3_multimodel_framing_analysis.ipynb
        │
        ▼
  comparison charts: base Gemma 3 vs. all four fine-tuned adapters
```

## Notebooks

| Notebook | Stage | What it does |
|---|---|---|
| `news_scraper.ipynb` | Scrape | Scrapes all four outlets, filtered to Gaza/Israel-Palestine coverage on/after 7 Oct 2023 |
| `gemma3_WAFA_training.ipynb` | Fine-tune | QLoRA fine-tunes Gemma 3-12B on WAFA articles |
| `gemma3_PNN_training.ipynb` | Fine-tune | QLoRA fine-tunes Gemma 3-12B on PNN articles |
| `gemma3_JPost_training.ipynb` | Fine-tune | QLoRA fine-tunes Gemma 3-12B on The Jerusalem Post articles |
| `gemma3_INN_training.ipynb` | Fine-tune | QLoRA fine-tunes Gemma 3-12B on Israel National News articles |
| `gemma3_multimodel_framing_analysis.ipynb` | Analyze | Loads base Gemma 3 + all four adapters, runs a framing-probe battery, produces comparison charts |

The four training notebooks are intentionally near-identical with same base model, same LoRA config
(`r=16`, `alpha=32`, `dropout=0.1`), same training hyperparameters, same `TARGET_N=980` downsample
target, so that any differences the analysis notebook finds reflect the training corpus, not the
training setup. The only real differences between them are the input CSV and the Drive folder each
adapter saves to.

---

## Stage 1: Scraping (`news_scraper.ipynb`)

Scrapes English-language articles from all four outlets. WAFA and PNN use sequential-ID
iteration, Jerusalem Post uses sitemap discovery + JSON-LD parsing, and Israel National News uses
Selenium (keyword search + article scraping) since its content is client-side rendered. See the
notebook's own intro cells for details on each site's approach.

**Output:** `wafa_raw_articles.csv`, `pnn_raw_articles.csv`, `jerusalem_post_articles.csv`,
`israelnationalnews_articles.csv`, each with a `content` (WAFA/PNN/JPost) or `article`
(Israel National News, pre-standardization) text column, used directly by the corresponding
training notebook. Upload these to Google Drive before running the training notebooks. Each one
expects its CSV at `/content/drive/MyDrive/<filename>`.

---

## Stage 2: Fine-tuning (`gemma3_12B_<outlet>_training.ipynb`)

Each notebook:

1. Mounts Google Drive and creates an outlet-specific folder (`gemma3_12b_wafa`,
   `gemma3_12b_pnn`, `gemma3_12b_jpost`, `gemma3_12b_israelnews`) so adapters never overwrite
   each other.
2. Loads the outlet's CSV, drops rows with missing title/body, downsamples to `TARGET_N=980`
   articles (same seed, same split) for comparability across outlets.
3. Loads Gemma 3-12B in 4-bit (QLoRA) via `bitsandbytes`.
4. Fine-tunes a LoRA adapter (`r=16`, `alpha=32`, 2 epochs, effective batch size 8) on a
   plain `Title: ...\n\nArticle: ...` next-token-prediction format.
5. Saves the final adapter to `<DRIVE_DIR>/gemma3_12b_<outlet>_final`. This exact path is what
   the multi-model analysis notebook expects.

**Requirements:** Colab A100 (40GB), a Hugging Face account with `google/gemma-3-12b-it` access
and an `HF_TOKEN` Colab secret.

---

## Stage 3: Multi-model comparison (`gemma3_multimodel_framing_analysis.ipynb`)

Loads the base Gemma 3-12B model plus all four LoRA adapters side by side, runs a framing-probe
battery (Yes/No and Likert-style prompts across a fixed item bank) against each, and produces
comparison charts showing how each outlet's fine-tuning shifts the model's framing relative to
base and to each other.

**Requirements:** same as training (Colab A100, HF token), needs all four adapters already
trained and saved to their expected Drive paths from Stage 2.

---

## Setup

```bash
pip install -r requirements.txt
```

The scraping notebook runs anywhere with internet access. The training and analysis notebooks are
built for Google Colab specifically (they mount Drive, request an A100 runtime, and read the
`HF_TOKEN` Colab secret). Running them elsewhere would need those parts adapted.

## Notes on scope and use

- **No scraped article text or trained model weights are included in this repo**. Republishing full news article text or redistributing fine-tuned model weights
  raises separate copyright/licensing questions from sharing the code itself (check each
  outlet's terms of use before redistributing scraped content, and note that `google/gemma-3-12b-it`
  is a gated model under Google's own license).
- The keyword list, date filter (7 Oct 2023 onward), and `TARGET_N=980` sampling target reflect
  this project's specific research design and would need adjusting for other purposes.
