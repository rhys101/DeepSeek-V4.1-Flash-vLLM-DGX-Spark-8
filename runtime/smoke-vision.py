"""Check single-image OCR/shapes and four-image ordering through the serving API."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',default='http://127.0.0.1:8000/v1')
p.add_argument('--model',default='deepseek-v41-flash')
p.add_argument('--out',type=Path,required=True,help='New directory for requests and responses')
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
fixtures=Path(__file__).resolve().parent/'vision-fixtures'
expected=json.loads((fixtures/'expected.json').read_text())

def image(n):
    encoded=base64.b64encode((fixtures/f'{n}.png').read_bytes()).decode()
    return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+encoded))

def check(name,prompt,images,answer):
    body=dict(model=a.model,messages=[dict(role='user',content=[dict(type='text',text=prompt)]+images)],
              temperature=0,max_tokens=160,chat_template_kwargs=dict(thinking=False))
    (a.out/f'{name}-request.json').write_text(json.dumps(body,indent=2)+'\n')
    request=urllib.request.Request(a.base.rstrip('/')+'/chat/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    started=time.monotonic()
    with urllib.request.urlopen(request,timeout=300) as response: data=json.load(response)
    (a.out/f'{name}-response.json').write_text(json.dumps(data,indent=2)+'\n')
    result=data['choices'][0]['message']['content'].strip()
    if result.startswith('```'):result=result.split('\n',1)[1].rsplit('```',1)[0].strip()
    actual=json.loads(result)
    assert actual==answer,dict(actual=actual,expected=answer)
    print(json.dumps(dict(status='PASS',test=name,elapsed_seconds=round(time.monotonic()-started,3),answer=actual)))

check('single-image','Read the printed code, and identify the two shapes from left to right. Return only a JSON object with keys "code", "left_shape", "right_shape". All three values must be strings. Each shape string must contain its color, a space, and shape name, for example "green triangle".',[image(1)],expected['single'])
check('four-images','Read the printed code at the top of each image. Return only a JSON array of four strings, in the same order as the four attached images.',[image(n) for n in range(1,5)],expected['four'])
(a.out/'PASS').write_text('Single-image content and four-image ordering passed.\n')
