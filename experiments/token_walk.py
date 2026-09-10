"""Observe tokenization and a short greedy continuation via the public SDK."""

import argparse
import sys

from llm_sdk import Small_LLM_Model


def experiment(prompt: str, steps: int, top_k: int) -> None:
    """Print input tokens and repeatedly select the highest-scoring token."""
    print("モデルを読み込みます…", flush=True)
    model = Small_LLM_Model()

    # encode returns a batch of one sequence; extract that sequence as a list.
    encoded = model.encode(prompt)
    ids: list[int] = encoded.tolist()[0]
    if not ids:
        raise ValueError("入力には少なくとも1つのトークンが必要です。")

    print(f"\n入力: {prompt!r}")
    print(f"encode の形: {tuple(encoded.shape)}")
    print(f"トークンID: {ids}")
    print(f"decode で復元: {model.decode(ids)!r}")
    print("\n各トークン（空白や改行も表示）:")
    for token_id in ids:
        print(f"  {token_id:6d}  {model.decode([token_id])!r}")

    print("\nlogit は確率ではなく、次の候補のスコアです。")
    print("単独トークンは文字の一部の場合があります。")
    print("特殊トークンは decode で非表示になる場合があります。")
    print("この実験は終了トークンを判定せず、指定回数だけ進めます。")
    generated: list[int] = []
    for step in range(1, steps + 1):
        # Score every possible next token using the entire current sequence.
        logits = model.get_logits_from_input_ids(ids)
        if not logits:
            raise ValueError("モデルからスコアが返りませんでした。")
        ranked = sorted(
            range(len(logits)), key=lambda i: logits[i], reverse=True
        )[:top_k]

        print(f"\n--- Step {step}: 上位候補 / 全{len(logits)}候補 ---")
        for token_id in ranked:
            print(
                f"  ID={token_id:6d}  logit={logits[token_id]:9.4f}"
                f"  text={model.decode([token_id])!r}"
            )

        # Greedy selection: no sampling and no JSON constraints yet.
        next_id = ranked[0]
        ids.append(next_id)
        generated.append(next_id)
        print(f"選択: ID={next_id} / {model.decode([next_id])!r}")
        print(f"ここまでの続き: {model.decode(generated)!r}", flush=True)

    print(f"\n入力＋続き: {model.decode(ids)!r}")


def main() -> int:
    """Parse experiment options and report errors without a traceback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.steps <= 30:
        parser.error("--steps は 0〜30 にしてください。")
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k は 1〜20 にしてください。")
    if not args.prompt:
        parser.error("--prompt に空でない文章を指定してください。")
    try:
        experiment(args.prompt, args.steps, args.top_k)
    except KeyboardInterrupt:
        print("\n実験を中断しました。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"実験に失敗しました: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
