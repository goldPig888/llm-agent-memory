import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec=importlib.util.spec_from_file_location('watch_baseline',Path(__file__).resolve().parents[1] / 'scripts/watch_baseline.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


def setup_campaign(tmp_path):
    root=tmp_path/'baseline';root.mkdir()
    folder=root/'episodes/task00_state01/rollout';(folder/'api').mkdir(parents=True)
    (folder/'api/ep00_call01_request.json').write_text('{"saved":true}')
    row=dict(directory='episodes/task00_state01/rollout',status='running',pid=12345,
             task_id=0,initial_state_ids=[1])
    complete=dict(directory='episodes/task00_state00/rollout',status='finished',task_id=0,initial_state_ids=[0])
    manifest=dict(status='running',runs=[complete,row])
    (root/'campaign.json').write_text(json.dumps(manifest))
    return root,folder,manifest


def test_completed_outcomes_are_never_requeued(tmp_path):
    root,folder,m=setup_campaign(tmp_path)
    (folder/'summary.json').write_text(json.dumps({'results':[{'status':'failure'}]}))
    watch.reconcile(root,m)
    assert [r['status'] for r in m['runs']]==['finished','finished']
    assert (folder/'summary.json').exists()


def test_incomplete_attempt_archived_before_retry(tmp_path):
    root,folder,m=setup_campaign(tmp_path)
    watch.reconcile(root,m)
    row=m['runs'][1]
    assert row['status']=='pending' and 'pid' not in row
    archive=root/row['attempts'][0]['directory']
    assert (archive/'api/ep00_call01_request.json').read_text()=='{"saved":true}'
    assert not folder.exists()


def test_repeated_interruptions_have_bounded_retries(tmp_path):
    root,folder,m=setup_campaign(tmp_path)
    m['runs'][1]['attempts']=[{}]*3
    watch.reconcile(root,m)
    assert m['runs'][1]['status']=='error'


def test_live_controller_is_not_duplicated(tmp_path,monkeypatch):
    root,folder,m=setup_campaign(tmp_path)
    monkeypatch.setattr(watch,'MODEL_ROOT',tmp_path)
    monkeypatch.setattr(watch,'processes',lambda *args:[111])
    monkeypatch.setattr(watch,'process_matches',lambda *args:True)
    launched=[]
    monkeypatch.setattr(watch,'start_service',lambda *args:launched.append(args))
    monkeypatch.setattr(sys,'argv',['watch','--output',str(root)])
    watch.main()
    assert not launched
    assert json.loads((root/'watchdog_status.json').read_text())['state']=='running'
    assert json.loads((root/'campaign.json').read_text())==m


def test_missing_worker_recovers_only_pending_work(tmp_path,monkeypatch):
    root,folder,m=setup_campaign(tmp_path)
    monkeypatch.setattr(watch,'MODEL_ROOT',tmp_path)
    monkeypatch.setattr(watch,'processes',lambda *args:[])
    monkeypatch.setattr(watch,'process_matches',lambda *args:False)
    launched=[]
    monkeypatch.setattr(watch,'start_service',lambda *args:launched.append(args))
    class Response:
        status_code=200
        def json(self): return {'default_generation_settings':{'n_ctx':131072}}
    class Client:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def get(self,url): return Response()
    monkeypatch.setattr(watch.httpx,'Client',Client)
    monkeypatch.setattr(sys,'argv',['watch','--output',str(root)])
    watch.main()
    after=json.loads((root/'campaign.json').read_text())
    assert after['runs'][0]['status']=='finished'
    assert after['runs'][1]['status']=='pending'
    assert len(launched)==1 and launched[0][0]=='glimmer-baseline-queue-recovery'
