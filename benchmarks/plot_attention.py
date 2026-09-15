"""Validate raw statistics and generate local SVGs without measurements."""
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

raw = Path(sys.argv[1])
data = json.loads(raw.read_text())
assert data['passed'] and len(data['cases']) == 12
for case in data['cases']:
    for p in case['measurements']:
        t = p['latency']; s = t['samples']
        assert len(s) == t['repetition_count']
        for k, v in [('median', statistics.median(s)), ('mean', statistics.mean(s)), ('stddev', statistics.pstdev(s)), ('min', min(s)), ('max', max(s))]:
            assert abs(t[k] - v) < 1e-8
        assert p['persistent_k_bytes'] == sum(a['bytes'] for a in p['allocations'] if a['role'] == 'persistent-k')
out = Path(__file__).parent / 'charts' / 'attention'
out.mkdir(parents=True, exist_ok=True)
definitions = [
    ('persistent-k', 'Persistent K memory', 'KiB; codes and norms (shared tables separate)', lambda c,p: p['persistent_k_bytes']/1024),
    ('latency', 'Attention latency', 'Median microseconds; compressed includes full K reconstruction', lambda c,p: p['latency']['median']),
    ('compression-ratio', 'Effective persistent compression ratio', 'FP16 bytes / (K bytes + shared tables)', lambda c,p: c['context_length']*c['head_dimension']*2/(p['persistent_k_bytes']+p['shared_metadata_bytes'])),
    ('output-error', 'Attention-output error', 'Maximum absolute error versus CPU FP16-K attention', lambda c,p: p['total_output_error_vs_fp16']['max_absolute_error'])]
colors = ['#2563eb','#ea580c','#059669','#9333ea']
sha = hashlib.sha256(raw.read_bytes()).hexdigest()
for name,title,unit,value in definitions:
    series = [(f'{label} D={dim}', sorted((c['context_length'],value(c,next(p for p in c['measurements'] if p['path']==label))) for c in data['cases'] if c['head_dimension']==dim)) for dim in [64,128] for label in ['baseline','compressed']]
    top = max(v for _,points in series for _,v in points)*1.12 or 1
    x = lambda t: 90+(t-128)/3968*690
    y = lambda v: 375-v/top*270
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="490"><rect width="920" height="490" fill="white"/><g font-family="sans-serif" fill="#172033">', f'<text x="30" y="30" font-size="21">{title}</text>',f'<text x="30" y="53" font-size="12">{html.escape(unit)}</text>']
    for i in range(6):
        v=top*i/5
        parts += [f'<path d="M90 {y(v)} H780" stroke="#e2e8f0"/>',f'<text x="80" y="{y(v)+4}" text-anchor="end" font-size="11">{v:.3g}</text>']
    for t in [128,512,1024,2048,4096]:
        parts.append(f'<text x="{x(t)}" y="395" text-anchor="middle" font-size="11">{t}</text>')
    for i,(label,points) in enumerate(series):
        coordinates=' '.join(f'{x(t)},{y(v)}' for t,v in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        for t,v in points:
            parts.append(f'<circle cx="{x(t)}" cy="{y(v)}" r="3" fill="{colors[i]}"><title>T={t}: {v:.8g}</title></circle>')
        parts.append(f'<text x="{35+i*220}" y="80" fill="{colors[i]}" font-size="12">{label}</text>')
    parts += ['<text x="400" y="420" font-size="12">Context length; three noncausal queries</text>', '<text x="30" y="448" font-size="11">Synthetic development evidence. FP32 materialized K is additional scratch memory.</text>',f'<text x="30" y="473" font-size="8">{raw.name} | SHA256 {sha}</text>','</g></svg>']
    (out/f'{name}.svg').write_text('\n'.join(parts),encoding='utf-8')
(out/'source.json').write_text(json.dumps({'raw':raw.name,'sha256':sha},indent=2)+'\n')
print(f'Validated statistics; generated four SVGs in {out}')
