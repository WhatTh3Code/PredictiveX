# Raw data — provenance

`ai4i2020.csv` is the AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, dataset 601,
CC BY 4.0; Matzka, 2020). **This file is never modified**; all processing writes to `data/processed/`.

## Acquisition

The official source is <https://archive.ics.uci.edu/dataset/601>. During development the UCI site was blocked
by the build environment's network policy, so the file was taken from a public GitHub mirror
(`github.com/lingchenlijing/AI4I-2020-Predictive-Maintenance-Dataset`, `data/ai4i2020.csv`) and verified
against the dataset's published characteristics before use:

| Check | Expected (UCI / paper) | This file |
|---|---|---|
| Rows × columns | 10,000 × 14 | 10,000 × 14 |
| Machine failures | 339 | 339 |
| TWF / HDF / PWF / OSF / RNF | 46 / 115 / 95 / 98 / 19 | 46 / 115 / 95 / 98 / 19 |
| Product type L / M / H | 60% / 30% / 10% | 6,000 / 2,997 / 1,003 |
| First record | UDI 1, M14860 | UDI 1, M14860 |

SHA-256: `59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183`
(recorded in `ml/config.py`; `scripts/prepare_data.py` and `tests/data` check it).

## Re-downloading from the official source

1. Download the ZIP from <https://archive.ics.uci.edu/dataset/601> and extract `ai4i2020.csv` here.
2. Run `python scripts/prepare_data.py`. It prints whether the checksum matches the file used for the
   reported results. (UCI has re-packaged datasets in the past; if the checksum differs but the checks above
   still hold, re-run `scripts/run_pipeline.sh` to regenerate every result.)
