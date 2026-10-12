"""Check token-exact greedy output from a separately owned native engine.

This request gate does not launch services, qualify external recovery or compare
performance. Its caller freezes the installed engine, model and GPU ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import requests

from .artifacts import external_path


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite number in native response")
    return number


def reject_constant(value: str):
    raise ValueError(f"Nonstandard numeric constant in native response: {value}")


def output_evidence(engine: str, response: dict, prompt_tokens: int, output_tokens: int) -> dict:
    if engine == "vllm":
        choices = response["choices"]
        if len(choices) != 1:
            raise ValueError("Expected one native completion")
        choice = choices[0]
        text, tokens = choice["text"], choice["token_ids"]
        probabilities = choice["logprobs"]["token_logprobs"]
        usage = response["usage"]
        reason = choice["finish_reason"]
    elif engine == "sglang":
        text = response["text"]
        usage = response["meta_info"]
        pairs = usage["output_token_logprobs"]
        probabilities = [pair[0] for pair in pairs]
        tokens = [pair[1] for pair in pairs]
        reason = usage["finish_reason"]["type"]
    else:
        raise ValueError(f"Unknown native engine: {engine}")
    if not isinstance(text, str) or not text:
        raise ValueError("Native completion has no text")
    if reason != "length":
        raise ValueError(f"Native completion did not reach its fixed length: {reason}")
    for name, expected in (("prompt_tokens", prompt_tokens), ("completion_tokens", output_tokens)):
        if type(usage.get(name)) is not int or usage[name] != expected:
            raise ValueError(f"Native {name} differs: expected {expected}, got {usage.get(name)}")
    if not isinstance(tokens, list) or len(tokens) != output_tokens:
        raise ValueError("Native generated-token coverage differs")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Invalid generated token ID")
    if len(probabilities) != output_tokens or any(
        type(probability) not in (int, float) or not math.isfinite(probability)
        for probability in probabilities
    ):
        raise ValueError("Missing or nonfinite selected-token log probabilities")
    return {"text": text, "token_ids": tokens, "token_logprobs": probabilities, "usage": usage}


def payload(engine: str, model: str, tokens: list[int], output_tokens: int) -> tuple[str, dict]:
    sampling = {"temperature": 0, "ignore_eos": True}
    if engine == "vllm":
        return "/v1/completions", {
            "model": model,
            "prompt": tokens,
            "max_tokens": output_tokens,
            "return_token_ids": True,
            "logprobs": 1,
            "stream": False,
            **sampling,
        }
    if engine == "sglang":
        return "/generate", {
            "input_ids": tokens,
            "sampling_params": {"max_new_tokens": output_tokens, **sampling},
            "return_logprob": True,
            "logprob_start_len": -1,
            "stream": False,
        }
    raise ValueError(f"Unknown native engine: {engine}")


def qualify(args) -> dict:
    prompts_bytes = args.prompts.read_bytes()
    prompts = json.loads(prompts_bytes)
    if (
        not isinstance(prompts, list)
        or not 1 <= len(prompts) <= 32
        or any(
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 16384
            or any(type(token) is not int or token < 0 for token in tokens)
            for tokens in prompts
        )
    ):
        raise ValueError("Use 1–32 bounded, nonempty arrays of nonnegative integer token IDs")
    if not 1 <= args.output_tokens <= 64 or not 2 <= args.repeats <= 10:
        raise ValueError("Use 1–64 output tokens and 2–10 repeated greedy requests")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Native readiness output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "state": "RUNNING",
        "engine": args.engine,
        "model": args.model,
        "prompts_sha256": hashlib.sha256(prompts_bytes).hexdigest(),
        "prompt_lengths": [len(tokens) for tokens in prompts],
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "scope": "Native output readiness only; no external-state recovery or performance qualification",
        "requests": [],
    }
    with requests.Session() as session:
        session.trust_env = False
        try:
            for index, tokens in enumerate(prompts):
                previous = None
                for repeat in range(args.repeats):
                    prefix = args.output / f"prompt-{index:02d}-repeat-{repeat:02d}"
                    endpoint, body = payload(args.engine, args.model, tokens, args.output_tokens)
                    prefix.with_suffix(".request.json").write_text(
                        json.dumps(body, indent=2, allow_nan=False) + "\n"
                    )
                    with session.post(
                        args.url.rstrip("/") + endpoint,
                        json=body,
                        stream=True,
                        timeout=(10, 300),
                    ) as response:
                        with prefix.with_suffix(".response.txt").open("xb") as output:
                            size = 0
                            for chunk in response.iter_content(65536):
                                output.write(chunk)
                                size += len(chunk)
                                if size > 2 * 1024**2:
                                    raise ValueError("Native response exceeds the 2 MiB bound")
                        response.raise_for_status()
                    decoded = json.loads(
                        prefix.with_suffix(".response.txt").read_bytes(),
                        parse_float=finite_float,
                        parse_constant=reject_constant,
                    )
                    evidence = output_evidence(
                        args.engine, decoded, len(tokens), args.output_tokens
                    )
                    result["requests"].append({"prompt": index, "repeat": repeat, **evidence})
                    if previous is not None and (
                        evidence["text"] != previous["text"]
                        or evidence["token_ids"] != previous["token_ids"]
                    ):
                        raise ValueError("Repeated greedy native output differs")
                    previous = evidence
            result["state"] = "NATIVE_OUTPUT_READY_NOT_RECOVERY_QUALIFIED"
        except BaseException as error:
            result.update(
                state="INVALID", error={"type": type(error).__name__, "message": str(error)}
            )
            raise
        finally:
            (args.output / "RESULT.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    print(json.dumps(qualify(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
