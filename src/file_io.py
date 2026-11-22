"""JSONの読み込みと、一時ファイルを使った安全な書き込み処理。"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from .models import FunctionCallingResult


def _reject_nonstandard_constant(value: str) -> NoReturn:
    """PythonのJSONパーサーが受理するNaNと無限大の表記を拒否する。"""
    raise ValueError(f"invalid JSON numeric constant: {value}")


def read_json(input_path: str | Path) -> Any:
    """一つのUTF-8 JSONファイルを読み込んで解析する。"""
    with Path(input_path).open("r", encoding="utf-8") as file:
        #json文字列からpython辞書へ
        return json.load(file, parse_constant=_reject_nonstandard_constant)


def save_results(
    results: list[FunctionCallingResult],
    output_path: str | Path,
) -> None:
    """結果を一時ファイルへ書き、完成後にUTF-8 JSONとして置き換える。"""
    path = Path(output_path)
    #出力ファイルの保存先ディレクトリを作る
    #（まだにないなら作る・あってもエラーにしない）
    path.parent.mkdir(parents=True, exist_ok=True)
    #一時ファイル保存用のパス
    temporary_path: Path | None = None

    try:
        #重複しない一時ファイル名を自動で作り、そのファイルを開いて返す
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            #resultをpydanticモデル→python辞書→json文字列に変換
            json.dump(
                [result.model_dump(mode="json") for result in results],
                temporary_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            temporary_file.write("\n")
            #python書き込みバッファからOSにデータを渡し、仮ファイルを更新
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        #失敗時に仮ファイルを既に作成していた場合、仮ファイルを削除
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
