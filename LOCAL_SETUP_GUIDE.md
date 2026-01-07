# ローカル環境構築・実行ガイド

本ガイドは、Manus AIで作成したECG異常検知プロジェクトをローカル環境（VSCode）で実行し、最終目標を達成するための手順をまとめたものです。

## 1. 環境構築

### 1.1. 仮想環境の作成
ターミナルで以下のコマンドを実行してください。

```bash
# プロジェクトディレクトリへ移動
cd ecg_snn_project

# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 依存ライブラリのインストール
pip install torch torchvision torchaudio snntorch wfdb pandas numpy matplotlib seaborn pyyaml scikit-learn tqdm
```

### 1.2. データセットの取得
`src/data_loader.py` を実行するか、以下のPythonコードでMIT-BIHデータをダウンロードしてください。

```python
import wfdb
import os

data_dir = 'data/mitdb'
os.makedirs(data_dir, exist_ok=True)
wfdb.dl_database('mitdb', data_dir)
```

## 2. 実行ステップ（目標達成に向けて）

### ステップ1: 評価の確定（5-fold完遂）
`src/train_final_optimized.py` を実行して、5-foldすべての学習と評価を行います。
- **目標**: Sens≥0.80, Spec≥0.90, Macro F1≥0.75
- **ヒント**: `find_optimal_threshold_aggressive` の `target_sensitivity` を調整してください。

### ステップ2: CNNベースラインの正常化
`src/train_cnn_improved.py` を実行します。
- **目標**: Accuracy ≥ 0.90
- **ヒント**: 学習率（LR）を `1e-4` から `1e-3` の間で調整し、BatchNormが効いているか確認してください。

### ステップ3: SOPs/FLOPs ≤ 1 の達成
`src/sops_detailed_analysis.py` を実行します。
- **目標**: SOPs / FLOPs ≤ 1
- **ヒント**: Layer1の発火率が高い場合は、`config/config.yaml` でLIFの `threshold` を上げるか、`GradedDeltaEncoder` の `threshold_factor` を上げてください。

### ステップ4: SNNの価値の数値化（STDPまたはノイズ）
- **STDP**: `src/stdp_adaptation_v2.py` を実行し、少数の正常拍（few-shot）での適応曲線を確認してください。
- **ノイズ**: `src/noise_robustness.py` を実行し、ベースライン変動下での性能劣化をCNNと比較してください。

## 3. 主要ファイルの役割

| ファイル | 役割 |
| :--- | :--- |
| `src/snn_model_v2.py` | 時間軸SNNのモデル定義。SOPs計測用に修正済み。 |
| `src/data_loader.py` | 患者分割・層化k-fold・Graded Deltaエンコーディング。 |
| `src/train_final_optimized.py` | SNNのメイン学習スクリプト。閾値最適化ロジック搭載。 |
| `src/evaluate_trained_models.py` | 学習済みモデルの厳密な評価と平均±stdの算出。 |
| `src/sops_detailed_analysis.py` | 層別SOPs実測とFLOPs比較。 |
| `config/config.yaml` | 全体のハイパーパラメータ管理。 |

## 4. 成功のためのアドバイス

- **不均衡対策**: MIT-BIHは非常に偏っています。`WeightedRandomSampler` の重みを `[1.0, 10.0]` 程度まで上げるとSensitivityが改善しやすくなります。
- **SOPs削減**: SNNの消費電力を下げるには、スパイクを「疎（Sparse）」にすることが鍵です。LIFの閾値を少しずつ上げて、精度を維持できる限界を探ってください。

研究の成功を心より応援しております！
