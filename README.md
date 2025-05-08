# Mahjong Tile Classifier

このプロジェクトは、麻雀牌の画像をCNNで分類するモデルです。
学習済みモデルを用いて、Streamlitアプリで画像から牌を判定できます。

## フォルダ構成

mahjong_ml_project/

├── class_indices.json # ラベルとクラス番号の対応

├── mahjong_cnn_local.py # モデルの学習用スクリプト

├── mahjong_cnn_model.h5 # 学習済みモデル

├── predict_tile.py # 単一画像の予測スクリプト

├── streamlit_app.py # StreamlitによるWebアプリ

└── test_tile.png # テスト画像



## 必要ライブラリ

- Python 3.9
- TensorFlow 2.10
- Streamlit
- NumPy 1.26
- OpenCV（必要であれば）

※仮想環境 `tf_env39` 内にインストールしているパッケージは `requirements.txt` にまとめる予定です。

## 実行方法

```bash
# 仮想環境を有効化（Anacondaの例）
conda activate tf_env39

# アプリを起動
streamlit run streamlit_app.py

アプリ画面で画像をアップロードすると、麻雀牌の種類を推論します。