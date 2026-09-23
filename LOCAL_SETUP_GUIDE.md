# Local setup (research prototype)

This guide runs from the repository root. The small protocol tests do not need the ECG dataset or GPU. Training is expensive and has **not** been run with the corrected protocol in this repository.

## Python environment

Python 3.10+ を推奨します。歴史的な実験環境の依存バージョンは保存されていないため、以下は再現のための**導入例**であり、当時と同一の出力を保証しません。

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch snntorch wfdb pandas numpy matplotlib seaborn pyyaml scikit-learn scipy tqdm
python -m unittest discover -s tests
python -m compileall -q src tests
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch snntorch wfdb pandas numpy matplotlib seaborn pyyaml scikit-learn scipy tqdm
python -m unittest discover -s tests
python -m compileall -q src tests
```

各パッケージの固定バージョンは未確定です。GPU版PyTorchが必要な場合は利用環境に合わせて公式の導入手順を確認してください。

## Data

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)の使用条件を確認し、必要な`.hea`、`.dat`、`.atr`を`data/mitdb/`へ配置してください。データはこのrepoには含めません。`wfdb.dl_database('mitdb', 'data/mitdb')`でも取得できます。ダウンロードはネットワークと容量を要します。

## Corrected protocol

```bash
python src/train_corrected_protocol.py
```

このスクリプトは201/202が同一被験者である事実を考慮して202をtestから除外し、train・validation・独立testを分離します。SNNとCNNを学習するため時間がかかります。旧`src/train_final_optimized.py`はFold 0の歴史的スクリプトであり、5-fold完遂や独立testの結果は生成しません。結果の読み方は[README.md](README.md)と[results/README.md](results/README.md)を先に確認してください。
