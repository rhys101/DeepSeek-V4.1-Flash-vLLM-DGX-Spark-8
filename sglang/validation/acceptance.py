"""Run the same text/vision fixtures, plus structured output and tools, through either engine."""
import argparse
import base64
import concurrent.futures
import json
from pathlib import Path
import urllib.request

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    base=a.base.rstrip('/');fixtures=Path(__file__).with_name('vision-fixtures')
    expected=json.loads((fixtures/'expected.json').read_text())
    def send(name,messages,**extra):
        body=dict(model='deepseek-v41-flash',temperature=0,max_tokens=256,
                  chat_template_kwargs={'thinking':False},messages=messages,**extra)
        req=urllib.request.Request(base+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as r:d=json.load(r)
        (out/(name+'.json')).write_text(json.dumps(dict(request=body,response=d),indent=2)+'\n')
        return d['choices'][0]['message']
    def user(text):return [dict(role='user',content=text)]
    def text(name,prompt):return send(name,user(prompt))['content'].strip()
    def parse(s):
        if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
        return json.loads(s)
    def image(n):return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode((fixtures/f'{n}.png').read_bytes()).decode()))
    assert text('arithmetic','What is 17 times 19? Return only the integer.')=='323'
    # Objective answers at C8 catch failures hidden by a one-request smoke test.
    for trial in (1,2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            answers=list(pool.map(lambda i:text(f'batch-{trial}-{i}',f'What is {17+i} times 19? Return only the integer.'),range(8)))
        assert answers==[str((17+i)*19) for i in range(8)],answers
    prompt='Read the printed code, and identify the two shapes from left to right. Return only a JSON object with keys "code", "left_shape", "right_shape". All three values must be strings. Each shape string must contain its color, a space, and shape name, for example "green triangle".'
    assert parse(send('single-image',user([dict(type='text',text=prompt),image(1)]))['content'].strip())==expected['single']
    prompt='Read the printed code at the top of each image. Return only a JSON array of four strings, in the same order as the four attached images.'
    assert parse(send('four-images',user([dict(type='text',text=prompt)]+[image(i) for i in range(1,5)]))['content'].strip())==expected['four']
    schema=dict(type='json_schema',json_schema=dict(name='answer',strict=True,schema=dict(type='object',properties={'answer':{'type':'integer'}},required=['answer'],additionalProperties=False)))
    assert parse(send('structured',user('Return an object whose answer is the integer 42.'),response_format=schema)['content'].strip())=={'answer':42}
    tool=dict(type='function',function=dict(name='lookup_fixture',description='Retrieve a stored test value.',parameters=dict(type='object',properties={'key':{'type':'string'}},required=['key'],additionalProperties=False)))
    messages=user('Use lookup_fixture to retrieve the value for key alpha. Do not guess.')
    assistant=send('tool-call',messages,tools=[tool]);calls=assistant.get('tool_calls') or []
    assert len(calls)==1 and calls[0]['function']['name']=='lookup_fixture'
    assert json.loads(calls[0]['function']['arguments'])=={'key':'alpha'}
    messages += [assistant,dict(role='tool',tool_call_id=calls[0]['id'],content='{"value":42}')]
    assert '42' in send('tool-result',messages,tools=[tool])['content']
    (out/'result.json').write_text(json.dumps(dict(status='PASS',checks=['arithmetic','C8 arithmetic twice','single image','four images','structured JSON','tool round trip'],limitation='Capability smoke only; not a broad quality evaluation or long-context validation'),indent=2)+'\n')
    print('PASS: text, C8, vision, structured JSON and tool round trip')

if __name__=='__main__': main()
