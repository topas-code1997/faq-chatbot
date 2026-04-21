# 社内FAQチャットボット（RAG対応）

複数のPDFを読み込み、ベクトル検索で関連情報を抽出してAIが回答する社内FAQ検索システムです。

## デモ

※デモをご希望の方はお気軽にご連絡ください。

## 概要

社内マニュアルやFAQ文書（PDF）をアップロードすると、自然言語で質問できるチャットボットです。RAG（検索拡張生成）技術を使って、関連部分だけを抽出してAIに渡すため、精度の高い回答が得られます。

## 機能

- 複数PDFの読み込み・ベクトル化
- 自然言語での質問応答
- 関連ソースの表示
- Webブラウザから操作できるシンプルなUI

## 使用技術

| カテゴリ | 技術 |
|---|---|
| バックエンド | Python / Flask |
| AI | OpenAI API（GPT-4） |
| ベクトルDB | ChromaDB |
| デプロイ | Render |

## セットアップ

git clone https://github.com/topas-code1997/faq-chatbot.git
cd faq-chatbot
pip install -r requirements.txt

.envファイルを作成して以下を設定：
OPENAI_API_KEY=your_api_key_here

python app.py

ブラウザで http://localhost:5000 を開く。

## 作者

AIエンジニアとして活動中。
お仕事のご相談はクラウドワークスまたはXからどうぞ。
