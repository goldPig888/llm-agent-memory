import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location('run_local', Path(__file__).parents[1] / 'scripts/run_local.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def test_local_endpoint_only():
    assert local.local_url('http://127.0.0.1:8080/v1/') == 'http://127.0.0.1:8080/v1'
    for url in ['https://api.openai.com/v1', 'http://example.com', 'file:///tmp/model']:
        with pytest.raises(ValueError):
            local.local_url(url)


def test_local_capture_maps_effort_and_preserves_images_and_tools(tmp_path):
    def respond(request):
        body = json.loads(request.read())
        assert body['chat_template_kwargs'] == {'reasoning_strength': 'medium'}
        assert 'reasoning_effort' not in body
        assert body['messages'][0]['content'][0]['type'] == 'image_url'
        assert body['tools'][0]['function']['name'] == 'move_to'
        return httpx.Response(200, json={'choices': [], 'usage': {'prompt_tokens': 20, 'completion_tokens': 5}})
    capture = local.Capture(tmp_path, 'medium', httpx.MockTransport(respond))
    request = httpx.Request('POST', 'http://127.0.0.1:8080/v1/chat/completions', json={
        'reasoning_effort': 'high', 'messages': [{'role': 'user', 'content': [{'type': 'image_url'}]}],
        'tools': [{'type': 'function', 'function': {'name': 'move_to'}}]})
    assert capture.handle_request(request).status_code == 200
    events = [json.loads(line) for line in (tmp_path / 'api_usage.jsonl').read_text().splitlines()]
    assert events[-1]['usage']['input_tokens'] == 20
    assert capture.calls == 1
