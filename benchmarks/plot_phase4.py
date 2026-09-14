"""Four Phase 4 charts from immutable samples. No browser or timing side effects."""
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

raw=Path(sys.argv[1]); data=json.loads(raw.read_text()); assert data['passed'] and len(data['cases'])==8
for c in data['cases']:
    for p in c['measurements']:
        t=p['timing']
        for metric in [t['total'],t['host_record'],*t['passes']]:
            s=metric['samples']; assert len(s)==t['repetition_count']
            for k,v in [('median',statistics.median(s)),('mean',statistics.mean(s)),('stddev',statistics.pstdev(s)),('min',min(s)),('max',max(s))]:
                assert abs(metric[k]-v)<1e-8
        assert sum(a['bytes'] for a in p['allocations'])==p['total_explicit_buffer_bytes']
        for role,key in [('persistent-k','persistent_k_bytes'),('shared','shared_metadata_bytes'),('scratch','scratch_bytes')]:
            assert sum(a['bytes'] for a in p['allocations'] if a['role']==role)==p[key]
out=Path(__file__).parent/'charts'/'phase4';out.mkdir(parents=True,exist_ok=True)
definitions=[('latency','Attention latency','Median GPU ms; three noncausal queries',lambda p:p['timing']['total']['median']/1000,False),
             ('persistent-k','Persistent K including shared metadata','KiB; actual code, norm and table allocations',lambda p:(p['persistent_k_bytes']+p['shared_metadata_bytes'])/1024,False),
             ('scratch','Temporary / scratch GPU buffers','KiB; excludes workgroup local memory and timestamp probes',lambda p:p['scratch_bytes']/1024,False),
             ('compressed-latency','Phase 3 vs Phase 4 compressed attention','Median GPU ms; same Q/K semantics/V/device',lambda p:p['timing']['total']['median']/1000,True)]
colors=['#2563eb','#ea580c','#059669','#9333ea','#be123c','#0891b2']; sha=hashlib.sha256(raw.read_bytes()).hexdigest()
for name,title,unit,fn,compressed in definitions:
    series=[]
    for dim in [64,128]:
        for label in (['phase3','phase4'] if compressed else ['fp16','phase3','phase4']):
            points=sorted((c['context_length'],fn(next(p for p in c['measurements'] if p['path']==label))) for c in data['cases'] if c['head_dimension']==dim)
            series.append((f'{label} D={dim}',points))
    top=max(v for _,points in series for _,v in points)*1.12 or 1
    x=lambda t:90+(t-512)/3584*720
    y=lambda v:390-v/top*260
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="515"><rect width="960" height="515" fill="white"/><g font-family="sans-serif" fill="#172033">',f'<text x="30" y="30" font-size="22">{html.escape(title)}</text>',f'<text x="30" y="54" font-size="12">{html.escape(unit)}</text>']
    for i in range(6):
        v=top*i/5;parts += [f'<path d="M90 {y(v)} H810" stroke="#e2e8f0"/>',f'<text x="80" y="{y(v)+4}" text-anchor="end" font-size="11">{v:.3g}</text>']
    for t in [512,1024,2048,4096]:parts.append(f'<text x="{x(t)}" y="410" text-anchor="middle" font-size="11">{t}</text>')
    for i,(label,points) in enumerate(series):
        coords=' '.join(f'{x(t)},{y(v)}' for t,v in points)
        # Dash D=128 so overlapping allocation curves remain distinguishable.
        dash=' stroke-dasharray="6 3"' if '128' in label else ''
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i]}" stroke-width="2"{dash}/>')
        for t,v in points:parts.append(f'<circle cx="{x(t)}" cy="{y(v)}" r="3" fill="{colors[i]}"><title>T={t}: {v:.8g}</title></circle>')
        parts.append(f'<text x="{35+(i%3)*290}" y="{80+(i//3)*20}" font-size="12" fill="{colors[i]}">{label}</text>')
    parts += ['<text x="405" y="440" font-size="12">Context length</text>','<text x="30" y="474" font-size="11">Synthetic attention only. No model throughput or total-model memory claim. Samples retained without filtering.</text>',f'<text x="30" y="499" font-size="8">{raw.name} | SHA256 {sha}</text>','</g></svg>']
    (out/f'{name}.svg').write_text('\n'.join(parts),encoding='utf-8')
(out/'source.json').write_text(json.dumps({'raw':raw.name,'sha256':sha},indent=2)+'\n')
print(f'Validated all statistics and buffer ledgers; generated four charts in {out}')
