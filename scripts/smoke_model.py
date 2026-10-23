"""Verify local model health, two-camera input, and structured robot tool output."""
import base64
import io
import json
import httpx
from PIL import Image, ImageDraw

image = Image.new('RGB', (256, 256), 'white')
ImageDraw.Draw(image).rectangle((80, 80, 170, 170), fill='red')
buffer = io.BytesIO(); image.save(buffer, format='PNG')
url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
with httpx.Client(trust_env=False, timeout=1800) as client:
    response = client.post('http://127.0.0.1:8081/v1/chat/completions', json={
        'model': 'muse-glimmer', 'max_tokens': 4096,
        'chat_template_kwargs': {'reasoning_strength': 'low'},
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'These are two test camera views. Call inspect_scene with the color and shape you see. This is a wiring test; do not move a robot.'},
            {'type': 'image_url', 'image_url': {'url': url}},
            {'type': 'image_url', 'image_url': {'url': url}}]}],
        'tools': [{'type': 'function', 'function': {'name': 'inspect_scene',
            'description': 'Report the visible shape and its color.',
            'parameters': {'type': 'object', 'properties': {'color': {'type': 'string'},
                'shape': {'type': 'string'}}, 'required': ['color', 'shape']}}}]})
    response.raise_for_status()
    payload = response.json()
    message = payload['choices'][0]['message']
    calls = message.get('tool_calls', [])
    assert calls, f'No structured tool call: {message.get("content")}'
    arguments = json.loads(calls[0]['function']['arguments'])
    assert 'red' in arguments.get('color', '').lower(), arguments
    assert calls[0]['function']['name'] == 'inspect_scene'
    print(json.dumps({'passed': True, 'tool': calls[0]['function'], 'usage': payload.get('usage')}))
