# Result provenance and reading order

1. `core_evaluation_summary.csv`、`threshold_optimization.csv`、`snn_vs_cnn_comparison.csv`、`final_results_summary.md`は数値が相互に一致する**歴史的なFold 0 validation snapshot**です。実行ログとcheckpointの完全な対応は未確認。閾値を同じvalidationから選んでいるため、独立testの結果ではありません。
2. `final_report_revised.md`は同じFold 0のSNN値が上記と食い違います。**HISTORICAL / INCONSISTENT**。出典実行を特定できるまで、性能主張には使用しません。
3. `final_report.md`は初期の広い研究構想・実験記述を含みます。**HISTORICAL / UNVERIFIED**。Fold数、SOPs、STDP、ノイズ等の実施状況をここから確定しません。
4. PNGは複数の試行から作られた可能性があります。原本の実行ログと対応できないものは図示資料であり、性能・省電力性の証拠ではありません。
5. `models/`の重みは歴史的checkpointです。どのCSV・図の生成に使われたか、現在のコードで再ロードできるかは未検証です。

修正後の独立testプロトコルによる学習・評価結果は**まだありません**。旧ファイルを削除せず、矛盾と来歴を明示して保存します。
