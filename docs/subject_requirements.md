# call-me-maybe 課題要件

この文書は `en.subject.pdf` Version 1.5 のうち、実装、評価、提出に直接関係する要件を整理したものです。ページ番号はPDF上のページ番号です。一般的な背景説明とAI利用に関する一般論は省略しています。

## 1. 課題の目的

- 自然言語のプロンプトを、構造化された関数呼び出しへ変換するツールを作る。（PDF p.10）
- 質問への答えそのものを計算・返却するのではなく、呼び出す関数名と、その関数に渡す引数を生成する。（PDF p.10）
- 小規模LLMが自発的に正しいJSONを生成することに依存せず、制約付きデコードによって出力が必ず有効なJSONかつ指定スキーマ準拠になるようにする。（PDF pp.10, 13）

例として `What is the sum of 40 and 2?` に対し、答えの `42` ではなく、概ね次の情報を出力する。

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

## 2. 使用言語・コード品質

- Python 3.10以上を使用する。（PDF p.7）
- flake8のコーディング規約に従う。（PDF p.7）
- 関数の引数、戻り値、必要な変数に型ヒントを付ける。（PDF p.7）
- すべての関数が、指定されたmypyチェックをエラーなしで通過しなければならない。（PDF pp.7-8）
- 関数とクラスには、目的、引数、戻り値を説明するPEP 257準拠のdocstringを付ける。Google形式またはNumPy形式などを使用できる。（PDF p.7）
- すべてのクラスで、検証にPydanticを使用する。（PDF p.8）

## 3. 依存関係と禁止事項

- `numpy` と `json` を使用できる。（PDF p.8）
- `numpy` と `pydantic` を、`uv`を使った仮想環境にインストールする。（PDF p.8）
- `dspy` および同種のパッケージを使用してはならない。明記されている禁止例は、PyTorch、Hugging Faceパッケージ、Transformers、Outlinesなど。（PDF p.8）
- 提供された `llm_sdk` パッケージのprivateなメソッドや属性を使用してはならない。（PDF p.8）
- `llm_sdk` は、`src` と同じ親ディレクトリにコピーして使用できる。（PDF p.8）
- 評価者とMoulinetteは、依存関係の準備として `uv sync` のみを実行する。（PDF p.8）

## 4. 使用モデル

- デフォルトモデルとして `Qwen/Qwen3-0.6B` を使用する。（PDF p.8）
- 他のモデルへの対応は可能だが、提出物は必ず `Qwen/Qwen3-0.6B` で動作しなければならない。（PDF p.8）
- 呼び出す関数はLLMに選択させる。ヒューリスティックなど、LLM以外の方法で関数を選んではならない。（PDF p.8）

## 5. 実行方法とパス

プログラムは次の形式で実行できなければならない。ここで `src` は実装ファイルを格納するディレクトリ。（PDF p.9）

```bash
uv run python -m src \
  [--functions_definition <function_definition_file>] \
  [--input <input_file>] \
  [--output <output_file>]
```

- 引数を省略した場合、入力は `data/input/` から読み、出力は `data/output/` に書き込む。（PDF p.9）
- `--functions_definition` で関数定義ファイルを指定できる。（PDF p.9）
- `--input` でプロンプト入力ファイルを指定できる。（PDF p.9）
- `--output` で出力ファイルを指定できる。（PDF p.9）

実行例：

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## 6. 入力ファイル

プログラムは通常、`data/input/` にある2つのJSONファイルを処理する。（PDF pp.10-11）

### 6.1 プロンプト入力

`function_calling_tests.json` はJSON配列で、各要素は次の形式を持つ。（PDF p.10）

```json
{
  "prompt": "What is the sum of 2 and 3?"
}
```

- `prompt` は処理対象の自然言語リクエスト。
- ファイルには複数のプロンプトを格納できる。

### 6.2 関数定義入力

`functions_definition.json` は、利用可能な関数を格納したJSON配列。各関数定義には次の情報が含まれる。（PDF p.11）

- 関数名
- 説明
- 引数名と各引数の型
- 戻り値の型

基本構造：

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {
      "type": "number"
    },
    "b": {
      "type": "number"
    }
  },
  "returns": {
    "type": "number"
  }
}
```

### 6.3 入力に関する可変条件とエラー処理

- 評価時には、提示されている例とは異なるプロンプトと関数セットが使われる。（PDF p.11）
- 提示された例を前提にした解答をハードコードしてはならない。（PDF p.14）
- 入力ファイルが不正なJSONである場合を処理する。（PDF p.11）
- 入力ファイルが存在しない場合を処理する。（PDF p.11）
- malformed input、missing file、edge caseを、クラッシュさせず適切に処理する。（PDF pp.8, 15）
- すべてのエラーをgracefulに処理し、利用者に明確なエラーメッセージを示す。未処理例外でプログラムがクラッシュした場合、評価では機能しないものとみなされる。（PDF pp.7-8）

## 7. LLM SDK

提供される `Small_LLM_Model` ラッパークラスを通じてLLMを操作する。（PDF p.12）

使用可能として説明されているpublicメソッド：

```python
get_logits_from_input_ids(input_ids: list[int]) -> list[float]
```

- トークンIDのリストを受け取り、LLMが生成したlogitsを返す。

```python
get_path_to_vocab_file() -> str
```

- トークンIDとトークンの対応を格納した語彙ファイルのパスを返す。

```python
encode(text: str) -> Tensor
```

- 文字列をモデルのトークナイザーでトークンIDのTensorに変換する。

```python
decode(token_ids: list[int]) -> str
```

- トークンIDのリストを文字列へ戻す。このメソッドの使用は任意。

## 8. 生成パイプライン

生成処理は、概ね次の流れを取る。（PDF pp.12-13）

1. 自然言語プロンプトを受け取る。
2. プロンプトをサブワード単位のトークンへ分割する。空白、句読点、単語の部分分割など、トークナイザー固有の表現を考慮する。
3. トークンをモデルが扱う数値IDへ変換する。
4. LLMに入力IDを処理させる。
5. 次のトークン候補すべてに対するlogitsを取得する。
6. 制約適用後の候補から次のトークンを選ぶ。
7. 選んだトークンを入力へ追加し、完全な応答が生成されるまで2〜6をトークン単位で繰り返す。

## 9. 制約付きデコード

制約付きデコードは、LLMが次トークンを選ぶ前にlogitsへ介入して実装する。（PDF p.13）

各生成ステップで、次の処理を行う。

1. LLMから、次に生成可能な全トークンのlogitsを取得する。
2. 追加後も有効なJSON構造と期待されるスキーマの両方を維持できるトークンを特定する。
3. JSONまたはスキーマを壊すトークンのlogitを負の無限大に設定する。
4. 残った有効なトークンだけから次のトークンを選ぶ。

制約は次の両方を保証しなければならない。

- 構文的に正しいJSONであること。
- 選択された関数の定義スキーマに適合すること。

具体的には次の点を扱う。

- `functions_definition.json` で `number` と指定された値は、JSONとして有効な整数または浮動小数点数に制限する。（PDF p.13）
- 各生成トークンを追加しても、JSON構造と期待スキーマの制約を維持できるようにする。（PDF p.13）
- 語彙JSONにあるトークンIDと文字列表現の対応を利用して、各ステップで許可可能なトークンを判断する。（PDF p.13）
- 関数定義をプロンプトで提示して正しいJSONが偶然生成されることを期待するだけの実装は禁止される。（PDF p.13）

## 10. 出力ファイル

- 通常の出力先は `data/output/function_calling_results.json`。（PDF p.14）
- すべてのプロンプトに対する結果を、単一のJSON配列として1ファイルに保存する。（PDF p.14）
- 入力プロンプト1件につき、出力配列へJSON objectを1件追加する。（PDF p.14）
- 各objectは、次の3キーを**正確に**含む。（PDF p.14）
  - `prompt` (`string`): 元の自然言語リクエスト。
  - `name` (`string`): 呼び出す関数の名前。
  - `parameters` (`object`): 必須引数すべてと、関数定義に一致する型の値。

出力例：

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```

### 10.1 出力検証規則

- 出力ファイルは有効なJSONでなければならない。末尾カンマとコメントは禁止。（PDF p.14）
- `name` は、入力された関数定義に存在する関数名と一致させる。（PDF p.15）
- キーと型は、選択された関数の `functions_definition.json` 内のスキーマと正確に一致しなければならない。（PDF p.14）
- 出力内の余分なキーと説明文は禁止。（PDF p.14）
- 必須引数をすべて含める。（PDF p.14）
- 引数の型は関数定義の指定（`number`、`string`、`boolean` など）と一致させる。（PDF p.14）

## 11. 性能・信頼性の評価目標

- 関数選択と引数抽出で90%以上の正解率を達成する。（PDF p.15）
- 生成結果の100%を、parse可能かつスキーマ準拠の有効なJSONにする。（PDF p.15）
- 標準的なハードウェアで、すべてのテストプロンプトを5分未満で処理する。（PDF p.15）
- 不正入力、ファイル欠落、エッジケースをgracefulに処理する。（PDF p.15）

## 12. テスト

### 12.1 機能確認手順

1. 入力ファイルを `data/input/` に配置する。
2. 指定された `uv run python -m src ...` コマンドで実行する。
3. 出力JSONが作成されたことを確認する。
4. JSONの構造と内容を検証する。
5. 関数名と引数型が関数定義に一致することを確認する。（PDF p.15）

### 12.2 確認すべきエッジケース

- 空文字列
- 大きな数値
- 特殊文字
- 誤った型
- 曖昧なプロンプト
- 複数引数を持つ関数（PDF p.15）

### 12.3 テストコードに関する指針

- 機能とエッジケースを検証するテストプログラムを作る。（PDF p.8）
- `pytest` または `unittest` などを使用できる。（PDF p.8）
- このテストプログラム自体は提出・採点対象ではない。（PDF p.8）

## 13. 例外・リソース管理

- 潜在的なエラーを `try-except` で管理し、関数が例外をgracefulに扱うようにする。（PDF p.7）
- ファイルや接続などのリソースには可能な限りcontext managerを使用し、自動的に解放する。（PDF p.7）
- ファイルハンドル、ネットワーク接続などのリソースリークを防ぐ。（PDF p.7）
- 予期しないクラッシュを起こさず、エラー時には明確なメッセージを示す。（PDF p.8）

## 14. Makefile

リポジトリにMakefileを置き、次のruleを実装する。（PDF pp.7-8）

- `install`: pip、uv、pipxなど任意のパッケージマネージャーで依存関係をインストールする。
- `run`: メインスクリプトを実行する。
- `debug`: Python標準デバッガー（例: pdb）でメインスクリプトを実行する。
- `clean`: `__pycache__`、`.mypy_cache` などの一時ファイルやキャッシュを削除する。
- `lint`: 次のコマンドを実行する。

```bash
flake8 .
mypy . \
  --warn-return-any \
  --warn-unused-ignores \
  --ignore-missing-imports \
  --disallow-untyped-defs \
  --check-untyped-defs
```

- `lint-strict` は任意。実装する場合は `flake8 .` と `mypy . --strict` を実行する。
- `--strict` での型チェックを試すことが強く推奨されている。

## 15. その他の開発指針

- Python生成物を除外する `.gitignore` を用意する。（PDF p.8）
- 開発時の依存関係分離には仮想環境の使用が推奨される。（PDF p.8）

## 16. README.md

リポジトリrootに、英語で書かれた `README.md` を置く。（PDF pp.16-17）

READMEの最低要件：

- 先頭行をイタリックにし、次の文面にする。ログイン名を実際の値に置き換える。（PDF p.16）

```markdown
*This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].*
```

- `Description`: 課題の目的と概要を明確に説明する。
- `Instructions`: インストール、準備、実行に必要な情報を記載する。
- `Resources`: 関連する公式文書、記事、チュートリアルなどの一般的な参考資料を列挙する。
- `Resources` 内で、AIをどの作業およびプロジェクトのどの部分に使用したか説明する。
- 制約付きデコードのアルゴリズムを詳しく説明する。
- 実装上の主要な設計判断を説明する。
- 解答の正確性、速度、信頼性を分析する。
- 遭遇した課題と、その解決方法を説明する。
- 実装を検証したテスト戦略を説明する。
- プログラムの明確な実行例を載せる。
- プロジェクトに応じて、使用例、機能一覧、技術選択などの追加セクションが必要になる場合がある。

## 17. 提出物とリポジトリ

通常どおりGitリポジトリへ提出し、defenseではリポジトリ内のファイルだけが評価される。（PDF p.19）

リポジトリに必要なもの：

- 実装を格納した `src/` ディレクトリ
- 依存関係管理用の `pyproject.toml` と `uv.lock`
- 提供パッケージからコピーした `llm_sdk/` ディレクトリ
- デモ用テストファイルを格納した `data/input/` ディレクトリ
- 必要事項を網羅した `README.md`
- 解答の実行に必要なその他のファイル

提出時の注意：

- `output/` ディレクトリをリポジトリに含めてはならない。既定構成では生成先の `data/output/` がこれに該当し、peer review時に生成される。（PDF p.19）
- ファイル名が正しいことを提出前に再確認する。（PDF p.19）
- 評価中に、挙動、関数、スクリプト、表示、データ構造などへの小規模な変更を求められる場合がある。評価ガイドラインで指定された変更を数分程度で実施できるよう、実装を理解しておく。（PDF pp.19-20）

## 18. Bonus（任意）

Bonusは合格に必須ではない。READMEへ記載するだけでは認められず、実装済みかつ動作する必要があり、評価中にデモを求められる場合がある。（PDF p.18）

- `Qwen/Qwen3-0.6B` 以外の複数LLMモデルに対応する。
- メインコードでSDKの `encode` と `decode` を直接使わず、`get_logits_from_input_ids` と `get_path_to_vocab_file` を使ってトークナイザーを再実装する。
- 高度なエラー回復機構を実装する。
- キャッシュやバッチ処理による性能最適化を行う。
- 包括的なテストスイートを作る。
- 生成過程を可視化する。
- 複雑にネストした関数引数に対応する。
- tokenizerの `encode` と、任意で `decode` の公開実装を作る。
- encoding、decoding、制約付きデコードの統合をデモする。

## 19. PDFに明記されていない入力検証方針

次の条件は合理的な防御的検証ではあるが、課題の必須条件としては明記されていない。

- 関数名、説明、プロンプトの空文字・空白文字を禁止すること。
- 関数名の重複を禁止すること。
- プロンプトの重複を禁止すること。
- 入力objectの余分なキーを禁止すること。
- 入力配列を1件以上に制限すること。

ただし、PDFは空文字列をテストすべきエッジケースとして挙げている。空文字を拒否するのか、処理可能な入力として扱うのかを実装側で決め、クラッシュせず一貫した挙動にする必要がある。（PDF p.15）
