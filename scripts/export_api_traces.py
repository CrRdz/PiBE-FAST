"""Export existing API regression scenarios; synthetic audio, no human replay."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('api_fixture', ROOT/'tests/test_web.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CASES = [
 'test_late_audio_cannot_downgrade_reported_acute_symptom',
 'test_audio_from_reset_session_is_rejected',
 'test_symptom_update_and_correction_are_persisted_as_revisions',
 'test_manual_symptoms_allow_unknown_onset_and_survive_retry',
]
class Recorder:
    def __init__(self, client, rows): self.client, self.rows = client, rows
    def __getattr__(self, method):
        original = getattr(self.client, method)
        if method not in ('get', 'post'): return original
        def request(path, **kwargs):
            response = original(path, **kwargs)
            self.rows.append(dict(method=method.upper(), path=path,
                payload=kwargs.get('json'), status=response.status_code,
                response=response.get_json()))
            return response
        return request

def main():
    records=[]
    for name in CASES:
        case=module.BefastWebApiTest(name)
        case.setUp()
        rows=[]
        try:
            case.client=Recorder(case.client,rows)
            getattr(case,name)()
            records.append(dict(case=name,passed=True,requests=rows))
        finally: case.tearDown()
    paths=['scripts/export_api_traces.py','tests/test_web.py','app/web.py',
           'app/befast/session.py','app/befast/report.py','app/befast/urgency.py',
           'app/befast/config.py','app/history.py','app/speech_audio.py']
    result=dict(scope='Existing Flask test-client regression scenarios with synthetic sine-wave audio and stub ASR/model. Production API, session, report and temporary history store. No real participant input, network transport, browser or camera validation.',
        cases=records, passed=len(records),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    out=ROOT/'experiments/workflow_validation/api-traces.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(records)} API scenarios passed; {sum(len(r["requests"]) for r in records)} requests recorded: {out}')
if __name__=='__main__': main()
