import json
import httpx
import pytest

from llm_agent_memory.harness.haimaker import BatchBudget,HaimakerCapture,ProviderStop,ENDPOINT,MODEL


def test_api_transport_sends_max_and_images_without_logging_credentials(tmp_path):
    key='test-secret-not-a-real-key';seen=[]
    def respond(request):
        body=json.loads(request.read());seen.append(body)
        assert request.headers['Authorization']=='Bearer '+key
        return httpx.Response(200,json={'choices':[{'message':{'content':'robot.done()'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':50}})
    budget=BatchBudget(5,tmp_path/'budget.json')
    transport=HaimakerCapture(tmp_path/'api',key,budget,inner=httpx.MockTransport(respond))
    with httpx.Client(transport=transport) as client:
        client.post(ENDPOINT,json={'model':MODEL,'messages':[{'role':'user','content':[
            {'type':'image_url','image_url':{'url':'data:image/png;base64,abc'}}]}]})
    assert seen[0]['reasoning_effort']=='max' and seen[0]['thinking']['type']=='enabled'
    assert seen[0]['max_tokens']==16384 and 'chat_template_kwargs' not in seen[0]
    assert seen[0]['messages'][0]['content'][0]['type']=='image_url'
    assert budget.used==pytest.approx((100*.3+50*1.2)/1e6)
    assert all(key not in p.read_text() for p in tmp_path.rglob('*.json*'))


def test_budget_shared_across_episodes_blocks_before_request(tmp_path):
    budget=BatchBudget(.4,tmp_path/'budget.json')
    reserved=budget.reserve(16384)
    with pytest.raises(ProviderStop,match='budget_reservation'):budget.reserve(16384)
    assert budget.used==reserved


def test_unknown_usage_retains_reservation_and_stops(tmp_path):
    budget=BatchBudget(5,tmp_path/'budget.json')
    transport=HaimakerCapture(tmp_path/'api','test',budget,inner=httpx.MockTransport(
        lambda request:httpx.Response(200,json={'choices':[]})))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ProviderStop,match='usage_unknown'):
            client.post(ENDPOINT,json={'model':MODEL,'messages':[]})
    assert budget.used>.3


def test_endpoint_allowlist_and_provider_errors_have_no_retry(tmp_path):
    calls=[]
    budget=BatchBudget(5,tmp_path/'budget.json')
    def respond(request):
        calls.append(1);return httpx.Response(429,json={'error':{'message':'quota'}})
    transport=HaimakerCapture(tmp_path/'api','test',budget,inner=httpx.MockTransport(respond))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError,match='endpoint'):client.post('https://example.com',json={})
        with pytest.raises(ProviderStop,match='provider_http_429'):
            client.post(ENDPOINT,json={'model':MODEL,'messages':[]})
    assert len(calls)==1 and budget.used>.3
