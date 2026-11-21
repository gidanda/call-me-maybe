"""JSON入力とアトミックな出力ファイルの処理。"""

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
        return json.load(file, parse_constant=_reject_nonstandard_constant)


def save_results(
    results: list[FunctionCallingResult],
    output_path: str | Path,
) -> None:
    """完全な結果リストをUTF-8 JSONとしてアトミックに書き込む。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                [result.model_dump(mode="json") for result in results],
                temporary_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
