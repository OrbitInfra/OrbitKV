"""A finite HTTP response alone does not establish token-exact native readiness."""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from requests.exceptions import HTTPError

from benches.serving.native_model_readiness import qualify


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "fault",
    [None, "nonfinite", "overflow", "counts", "missing_tokens", "finish", "changed_output", "http"],
)
def test_real_http_native_gate_retains_invalid_responses_and_requires_complete_greedy_output(
    tmp_path, engine, fault
):
    requests = []
    vllm = {
        "choices": [
            {
                "text": "native output",
                "token_ids": [19, 21],
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    sglang = {
        "text": "native output",
        "meta_info": {
            "output_token_logprobs": [[-0.1, 19, "native"], [-0.2, 21, " output"]],
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "finish_reason": {"type": "length"},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert self.path == ("/v1/completions" if engine == "vllm" else "/generate")
            assert body["prompt" if engine == "vllm" else "input_ids"] == [3, 5, 7]
            if engine == "vllm":
                assert body["return_token_ids"] is True and body["logprobs"] == 1
                assert body["ignore_eos"] is True and body["max_tokens"] == 2
            else:
                assert body["return_logprob"] is True and body["logprob_start_len"] == -1
                assert body["sampling_params"]["ignore_eos"] is True
                assert body["sampling_params"]["max_new_tokens"] == 2
            response = copy.deepcopy(vllm if engine == "vllm" else sglang)
            usage = response["usage" if engine == "vllm" else "meta_info"]
            if fault == "counts":
                usage["prompt_tokens"] = 4
            elif fault == "missing_tokens":
                if engine == "vllm":
                    del response["choices"][0]["token_ids"]
                else:
                    usage["output_token_logprobs"] = []
            elif fault == "finish":
                if engine == "vllm":
                    response["choices"][0]["finish_reason"] = "stop"
                else:
                    usage["finish_reason"]["type"] = "stop"
            elif fault == "changed_output" and len(requests) == 2:
                if engine == "vllm":
                    response["choices"][0]["token_ids"][0] = 20
                else:
                    usage["output_token_logprobs"][0][1] = 20
            wire = json.dumps(response).encode()
            if fault in {"nonfinite", "overflow"}:
                wire = wire.replace(b"-0.1", b"NaN" if fault == "nonfinite" else b"1e999")
            self.send_response(400 if fault == "http" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    prompts = tmp_path / "prompts.json"
    prompts.write_text("[[3,5,7]]")
    output = tmp_path / "result"
    args = SimpleNamespace(
        engine=engine,
        model="frozen-model",
        url=f"http://127.0.0.1:{server.server_port}",
        prompts=prompts,
        output=output,
        output_tokens=2,
        repeats=2,
    )
    try:
        if fault:
            with pytest.raises((ValueError, KeyError, HTTPError)):
                qualify(args)
        else:
            result = qualify(args)
            assert result["state"] == "NATIVE_OUTPUT_READY_NOT_RECOVERY_QUALIFIED"
            assert [row["token_ids"] for row in result["requests"]] == [[19, 21], [19, 21]]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not thread.is_alive()
    result = json.loads((output / "RESULT.json").read_text())
    assert result["state"] == ("INVALID" if fault else "NATIVE_OUTPUT_READY_NOT_RECOVERY_QUALIFIED")
    assert len(requests) == (2 if fault in {None, "changed_output"} else 1)
    for repeat in range(len(requests)):
        assert (output / f"prompt-00-repeat-{repeat:02d}.request.json").is_file()
        assert (output / f"prompt-00-repeat-{repeat:02d}.response.txt").is_file()
