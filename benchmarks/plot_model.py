"""Validate end-to-end samples and render final model charts, without rerunning."""
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys
raw=Path(sys.argv[1]);data=json.loads(raw.read_text());assert data['passed'] and len(data['cases'])==5
decode_raw=Path(sys.argv[2]) if len(sys.argv)>2 else None
decode_data=json.loads(decode_raw.read_text()) if decode_raw else None
if decode_data:
    assert decode_data['passed'] and decode_data['parent_sha256']==hashlib.sha256(raw.read_bytes()).hexdigest()
    assert decode_data['chrome_version']==data['chrome_version'] and decode_data['environment']==data['environment']
    for name,sha in data['source_sha256'].items():
        if name.startswith('src/'):assert decode_data['source_sha256'][name]==sha
    for c,original in zip(decode_data['cases'],data['cases'],strict=True):
        assert c['prompt_ids']==original['prompt_ids'] and c['context_length']==original['context_length']
        for m in c['measurements']:
            metric=m['decode'];s=metric['samples'];assert len(s)==metric['count']==24
            for k,v in [('mean',statistics.mean(s)),('median',statistics.median(s)),('stddev',statistics.pstdev(s)),('min',min(s)),('max',max(s)),('p95',sorted(s)[22])]:assert abs(metric[k]-v)<1e-7
for c in data['cases']:
    for m in c['measurements']:
        for name in ['prefill','ttft','decode']:
            metric=m[name]
            if metric is None:continue
            s=metric['samples'];assert len(s)==metric['count']
            for k,v in [('mean',statistics.mean(s)),('median',statistics.median(s)),('stddev',statistics.pstdev(s)),('min',min(s)),('max',max(s))]:assert abs(metric[k]-v)<1e-7
        assert sum(a['bytes'] for a in m['allocations'])==m['explicit_resident_bytes']
        for role,total in m['roles'].items():assert sum(a['bytes'] for a in m['allocations'] if a['role']==role)==total
out=Path(__file__).parent/'charts'/'model';out.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256(raw.read_bytes()).hexdigest()
definitions=[('kv-memory','Persistent KV including shared tables','MiB; capacity = prompt tokens + 32',lambda c,m:(m['roles']['k']+m['roles']['v']+m['roles'].get('tables',0))/2**20),
             ('decode-latency','Real-model decode latency','Median ms; identical supplied continuation, 24 steps per path' if decode_data else 'Median ms per generated token; absent EOS-only samples are omitted',lambda c,m:m['decode']['median'] if m['decode'] else None),
             ('prefill-latency','Real-model prefill latency','Median seconds; chunks of 32, includes host/GPU/readback',lambda c,m:m['prefill']['median']/1000),
             ('total-allocation','Total explicitly allocated GPU buffers','MiB; maximum live explicit sizes, not physical peak VRAM',lambda c,m:m['maximum_live_explicit_buffer_bytes']/2**20),
             ('quality-error','First-token logit difference vs compression','Maximum absolute logit difference; not a model accuracy score',lambda c,m:c['first_logit_error']['max_absolute_error'])]
for name,title,unit,value in definitions:
    error=name=='quality-error';series=[]
    chart_data=decode_data if name=='decode-latency' and decode_data else data
    for path in (['compressed'] if error else ['fp16','compressed']):
        points=[(c['persistent_k_ratio'] if error else c['context_length'],value(c,next(m for m in c['measurements'] if m['path']==path))) for c in chart_data['cases']]
        points=[(x,y) for x,y in points if y is not None]
        series.append((path,sorted(points)))
    allpoints=[p for _,s in series for p in s];xmin=min(x for x,y in allpoints) if error else 256;xmax=max(x for x,y in allpoints) if error else 4096;top=max((y for x,y in allpoints),default=1)*1.12 or 1
    x=lambda t:95+(t-xmin)/max(xmax-xmin,1e-9)*710;y=lambda v:375-v/top*270
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="510"><rect width="960" height="510" fill="white"/><g font-family="sans-serif" fill="#172033">',f'<text x="30" y="30" font-size="22">{title}</text>',f'<text x="30" y="55" font-size="12">{html.escape(unit)}</text>']
    for i in range(6):
        v=top*i/5;parts += [f'<path d="M95 {y(v)} H805" stroke="#e2e8f0"/>',f'<text x="84" y="{y(v)+4}" text-anchor="end" font-size="11">{v:.3g}</text>']
    for tick in ([xmin+(xmax-xmin)*i/4 for i in range(5)] if error else [256,1024,2048,4096]):parts.append(f'<text x="{x(tick)}" y="398" text-anchor="middle" font-size="11">{tick:.3g}</text>')
    for i,(path,points) in enumerate(series):
        color=['#2563eb','#ea580c'][i];coords=' '.join(f'{x(t)},{y(v)}' for t,v in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for t,v in points:parts.append(f'<circle cx="{x(t)}" cy="{y(v)}" r="4" fill="{color}"><title>x={t:.8g}, y={v:.8g}</title></circle>')
        parts.append(f'<text x="{35+i*360}" y="82" font-size="13" fill="{color}">{path}</text>')
    xlabel='Persistent K ratio including calibration tables (FP16 / compressed)' if error else 'Prompt context length'
    source=decode_raw if chart_data is decode_data else raw
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    note='Teacher-forced decode · one trajectory per path/context · includes GPU waits and logits readback' if chart_data is decode_data else 'SmolLM2-135M-Instruct · same device/prompts/greedy settings · raw samples retained'
    parts += [f'<text x="180" y="432" font-size="12">{xlabel}</text>',f'<text x="30" y="468" font-size="11">{note}</text>',f'<text x="30" y="494" font-size="8">{source.name} | SHA256 {source_sha}</text>','</g></svg>']
    (out/f'{name}.svg').write_text('\n'.join(parts),encoding='utf-8')
(out/'source.json').write_text(json.dumps({'raw':raw.name,'sha256':sha,'decode_raw':decode_raw.name if decode_raw else None,'decode_sha256':hashlib.sha256(decode_raw.read_bytes()).hexdigest() if decode_raw else None,'plot_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print(f'Validated final samples and allocations; five model charts in {out}')
