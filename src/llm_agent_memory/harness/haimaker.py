"""Explicit Haimaker Chat transport; fail-closed batch accounting, no retries."""
import json
import math
from pathlib import Path
import time

import httpx

MODEL='deepseek/deepseek-v4.1-flash'
ENDPOINT='https://api.haimaker.ai/v1/chat/completions'
INPUT_RATE=.30/1e6
OUTPUT_RATE=1.20/1e6
CONTEXT_BOUND=1048576


class ProviderStop(RuntimeError):
    pass


class BatchBudget:
    def __init__(self,cap,path):
        if not math.isfinite(cap) or not 0<cap<=5:raise ValueError('Batch cap must be 0..5 USD')
        self.cap=cap;self.used=0.;self.path=Path(path)
        self.save()

    def save(self):
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(cap_usd=self.cap,charged_upper_bound_usd=self.used,
            pricing='Peak list-price upper bound; not a promotional invoice.'),indent=2))
        temp.replace(self.path)

    def reserve(self,max_tokens):
        # Bound all possible input, including image tokens, by model context.
        amount=CONTEXT_BOUND*INPUT_RATE+max_tokens*OUTPUT_RATE
        if self.used+amount>self.cap:raise ProviderStop('budget_reservation_limit')
        self.used+=amount;self.save();return amount

    def reconcile(self,reserved,usage,max_tokens):
        inp=usage.get('prompt_tokens');out=usage.get('completion_tokens')
        if not all(type(x) is int and x>=0 for x in (inp,out)):
            raise ProviderStop('usage_unknown_reservation_retained')
        if inp>CONTEXT_BOUND or out>max_tokens:
            raise ProviderStop('usage_exceeded_reservation_bound')
        cost=inp*INPUT_RATE+out*OUTPUT_RATE
        self.used-=reserved-cost;self.save();return cost


class HaimakerCapture(httpx.BaseTransport):
    def __init__(self,folder,key,budget,max_tokens=16384,inner=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.key=key;self.budget=budget;self.max_tokens=max_tokens
        self.episode=0;self.calls=0;self.inner=inner or httpx.HTTPTransport(trust_env=False)

    def event(self,**fields):
        with (self.folder/'api_usage.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(episode=self.episode,call=self.calls,**fields))+'\n')

    def handle_request(self,request):
        if str(request.url)!=ENDPOINT:raise ValueError('Unexpected provider endpoint')
        body=json.loads(request.read())
        if body.get('model')!=MODEL:raise ValueError('Unexpected model')
        if 'chat_template_kwargs' in body:raise ValueError('Local template parameters cannot be sent to API')
        reserved=self.budget.reserve(self.max_tokens)
        self.calls+=1
        body.update(reasoning_effort='max',thinking={'type':'enabled'},max_tokens=self.max_tokens)
        body.pop('seed',None)  # No unsupported sampling controls in thinking mode.
        prefix=self.folder/f'ep{self.episode:02d}_call{self.calls:02d}'
        prefix.with_name(prefix.name+'_request.json').write_text(json.dumps(body))
        self.event(event='request_started',reservation_usd=reserved)
        started=time.monotonic()
        try:
            response=self.inner.handle_request(httpx.Request('POST',ENDPOINT,json=body,
                headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key},
                extensions=request.extensions))
            response.read()
        except Exception:
            self.event(event='usage_unknown',budget_charge_usd=reserved)
            raise ProviderStop('network_error_reservation_retained') from None
        prefix.with_name(prefix.name+'_response.json').write_bytes(response.content)
        try:payload=response.json()
        except ValueError:
            self.event(event='usage_unknown',http_status=response.status_code,budget_charge_usd=reserved)
            raise ProviderStop('invalid_response_reservation_retained') from None
        usage=payload.get('usage') or {}
        if response.status_code>=400:
            self.event(event='usage_unknown',http_status=response.status_code,budget_charge_usd=reserved)
            raise ProviderStop('provider_http_'+str(response.status_code))
        try:cost=self.budget.reconcile(reserved,usage,self.max_tokens)
        except ProviderStop:
            self.event(event='usage_unknown',http_status=response.status_code,budget_charge_usd=reserved)
            raise
        self.event(event='response',http_status=response.status_code,latency_s=time.monotonic()-started,
            estimated_usd=cost,budget_charge_usd=cost,
            usage=dict(input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],
                input_tokens_details=usage.get('prompt_tokens_details',{}),
                output_tokens_details=usage.get('completion_tokens_details',{})))
        return response

    def close(self):self.inner.close()
