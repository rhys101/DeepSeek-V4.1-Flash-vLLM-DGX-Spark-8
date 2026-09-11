"""Bounded retrieval probes near the configured context limit; not a broad quality benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    p.add_argument('--tag',required=True,help='Identical fresh tag on each engine')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',tag=a.tag,cases=[],
                method='C1; unique prefix per size, identical prompts across engines, thinking off, 256-token output budget; exact retrieval of three records separated by repeated-word filler. Records at beginning, middle and end. This is a capability smoke, not broad long-context quality.')
    def save(): (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for size in (32768,131072,299000):
            rng=random.Random(size+41)
            expected={k:str(rng.randrange(10000000,100000000)) for k in ('alpha','beta','gamma')}
            prompt=(f'[long-context {a.tag} size={size}]\n'
                    'Three exact records are embedded below. Ignore all filler. At the end, return only a JSON object mapping alpha, beta and gamma to their recorded eight-digit string values.\n'
                    f'RECORD alpha={expected["alpha"]}\n'+ ' the'*(size//2)+
                    f'\nRECORD beta={expected["beta"]}\n'+ ' the'*(size-size//2)+
                    f'\nRECORD gamma={expected["gamma"]}\n'
                    'Return the three recorded values as a JSON object with string values. No explanation.')
            body=dict(model='deepseek-v41-flash',temperature=0,max_tokens=256,
                      chat_template_kwargs={'thinking':False},messages=[dict(role='user',content=prompt)])
            start=time.time()
            req=urllib.request.Request(a.base.rstrip('/')+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=1200) as r:response=json.load(r)
            (out/f'{size}-response.json').write_text(json.dumps(response,indent=2)+'\n')
            text=response['choices'][0]['message']['content'].strip()
            if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
            actual=json.loads(text);usage=response.get('usage') or {}
            row=dict(filler_units=size,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                     expected=expected,actual=actual,usage=usage,total_s=time.time()-start,
                     finish_reason=response['choices'][0].get('finish_reason'))
            row['passed']=actual==expected and size-64 <= usage.get('prompt_tokens',0) <= size+768 and usage['prompt_tokens']+256<=300000
            result['cases'].append(row);save();print(json.dumps(row),flush=True)
            assert row['passed'],f'Long-context retrieval/usage check failed at {size}'
        result['status']='PASS';save()
    except Exception as exc:
        result['status']='FAIL';result['error']=repr(exc);save();raise

if __name__=='__main__':main()
