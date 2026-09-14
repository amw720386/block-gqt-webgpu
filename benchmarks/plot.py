"""Generate five standalone SVG charts directly from an immutable raw run."""
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
COLORS=["#2563eb","#d97706","#08916e","#a855f7"]


class SVG:
    def __init__(self,title,subtitle,source):
        self.parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="650" viewBox="0 0 1060 650">',
                    '<rect width="1060" height="650" fill="white"/>',
                    '<style>text{font-family:Arial,sans-serif;fill:#25334a} .small{font-size:12px}</style>']
        self.text(36,36,title,22);self.text(36,62,subtitle,13)
        self.text(36,608,"Development microbenchmark • no attention/model results",12)
        self.text(36,632,source,11)
        self.parts.append(f'<metadata>{html.escape(source)}</metadata>')

    def text(self,x,y,text,size=12,anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{html.escape(str(text))}</text>')

    def line(self,x1,y1,x2,y2,color="#cbd5e1",dash=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')

    def rect(self,x,y,w,h,color):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{max(h,0)}" fill="{color}"/>')

    def save(self,path):path.write_text('\n'.join(self.parts+['</svg>']),encoding='utf-8')


def bar_chart(title,subtitle,labels,series,names,path,source):
    s=SVG(title,subtitle,source);left,right,top,bottom=85,1020,125,475
    peak=max(max(values) for values in series)*1.17 or 1
    y=lambda v:bottom-v/peak*(bottom-top)
    for tick in range(6):
        value=peak*tick/5;s.line(left,y(value),right,y(value));s.text(left-9,y(value)+4,f'{value:.2f}',11,'end')
    step=(right-left)/len(labels);width=step*.72/len(series)
    for k,(name,color) in enumerate(zip(names,COLORS)):
        s.rect(85+k*285,85,14,14,color);s.text(105+k*285,97,name,12)
    for i,label in enumerate(labels):
        for k,values in enumerate(series):
            x=left+i*step+step*.14+k*width;v=values[i]
            s.rect(x,y(v),width*.92,bottom-y(v),COLORS[k]);s.text(x+width*.46,y(v)-6,f'{v:.2f}',10,'middle')
        for j,line in enumerate(label.split('\n')):s.text(left+(i+.5)*step,500+j*15,line,11,'middle')
    s.save(path)


def latency_chart(gpu,cpu,kernel,path,source):
    title='WGSL reconstruction latency' if kernel=='decode' else 'WGSL quantization latency'
    s=SVG(title,'256 vectors/dispatch • medians and min–max • logarithmic µs axis • CPU boundaries are non-equivalent',source)
    series=[]
    for env,records in [('GPU timestamps',gpu['records']),('CPU wall (non-equivalent)',cpu['records'])]:
        for allocation in ('low','mixed'):
            entries=[]
            for r in records:
                if r['fixture'].startswith(allocation+'_'):
                    t=next(x for x in r['timings'] if x['kernel']==kernel)
                    entries.append((r['workload']['dim'],t['statistics']))
            series.append((f'{env} / {allocation}',sorted(entries)))
    positive=[v for _,points in series for _,stat in points for v in (stat['min'],stat['max']) if v>0]
    if not positive:raise ValueError('No positive timings to plot')
    lo,hi=math.floor(math.log10(min(positive))),math.ceil(math.log10(max(positive)))
    if lo==hi:hi+=1
    left,right,top,bottom=85,1015,150,505
    x=lambda d:left+(math.log2(d)-4)/3*(right-left)
    y=lambda v:bottom-(math.log10(max(v,10**lo))-lo)/(hi-lo)*(bottom-top)
    for exponent in range(lo,hi+1):
        value=10**exponent;s.line(left,y(value),right,y(value));s.text(left-10,y(value)+4,f'{value:g}',11,'end')
    for d in (16,32,64,128):s.text(x(d),531,d,12,'middle')
    s.text(550,558,'Vector dimension',13,'middle')
    for k,(label,points) in enumerate(series):
        color=COLORS[k];s.line(85+(k%2)*455,89+(k//2)*24,110+(k%2)*455,89+(k//2)*24,color,k>=2)
        s.text(116+(k%2)*455,93+(k//2)*24,label,12)
        previous=None
        for d,stat in points:
            if stat['median']<=0:raise ValueError('Timestamp median is zero; increase batching before plotting')
            px,py=x(d),y(stat['median'])
            if previous:s.line(*previous,px,py,color,k>=2)
            s.line(px,y(stat['min']),px,y(stat['max']),color)
            s.rect(px-4,py-4,8,8,color);previous=(px,py)
    s.save(path)


def main(path):
    path=Path(path);gpu=json.loads(path.read_text());cpu=json.loads(Path(str(path).replace('-gpu.json','-cpu.json')).read_text())
    assert gpu['run_id']==cpu['run_id'] and gpu['passed'] and cpu['passed']
    out=ROOT/'charts';out.mkdir(exist_ok=True)
    records=gpu['records'];source=f"Raw: {path.name} • git {gpu['git_commit'][:12]} • {gpu['environment']['vendor']} {gpu['environment']['architecture']} • Chrome {gpu['chrome_version']}"
    labels=[f"D={r['workload']['dim']}\n{r['fixture'].split('_')[0]}" for r in records]
    bar_chart('Actual allocated K storage vs FP16','KiB • 512 reserved / 256 active vectors • shared tables are counted separately',labels,
        [[r['memory']['baseline_fp16_bytes']/1024 for r in records],
         [r['memory']['persistent_bytes']/1024 for r in records],
         [(r['memory']['persistent_bytes']+r['memory']['shared_metadata_bytes'])/1024 for r in records]],
        ['FP16 baseline','K codes + FP16 norms','K + norms + shared tables'],out/'allocated-bytes.svg',source)
    bar_chart('Effective compression ratio including shared metadata','FP16 allocated bytes / (K capacity + norm capacity + shared tables) • below 1 means more storage',
        [f"D={r['workload']['dim']}\n{r['fixture'].split('_')[0]} / {r['workload']['average_bits']:.2f} bits" for r in records],
        [[r['memory']['effective_compression_ratio'] for r in records]],['Effective ratio'],out/'compression-ratio.svg',source)
    latency_chart(gpu,cpu,'decode',out/'reconstruction-latency.svg',source)
    latency_chart(gpu,cpu,'quantize',out/'quantization-latency.svg',source)
    # Use a measured mixed allocation, ordered by original split-half RoPE pair.
    r=next(r for r in records if r['fixture']=='mixed_d16');w=r['workload']['allocation'];avg=sum(w)/len(w)
    s=SVG('Allocated width by RoPE frequency pair','Deterministic synthetic scores • mixed_d16 • average 2.75 bits • widths ≤4 still use nibbles',source)
    left,right,top,bottom=85,1015,125,485;step=(right-left)/len(w);y=lambda v:bottom-v/6*(bottom-top)
    for v in range(7):s.line(left,y(v),right,y(v));s.text(70,y(v)+4,v,12,'end')
    for i,b in enumerate(w):
        s.rect(left+i*step+step*.2,y(b),step*.6,bottom-y(b),COLORS[0] if b<=4 else COLORS[1])
        s.text(left+(i+.5)*step,y(b)-8,b,14,'middle');s.text(left+(i+.5)*step,509,i,12,'middle')
    s.line(left,y(avg),right,y(avg),'#7c3aed',True);s.text(1000,y(avg)-9,'Uniform-average guide: 2.75',12,'end')
    s.text(550,542,'RoPE frequency pair index (original order)',13,'middle')
    s.save(out/'pair-allocation.svg')
    (out/'latest.json').write_text(json.dumps(dict(run_id=gpu['run_id'],gpu_raw='../raw/'+path.name,
        cpu_raw='../raw/'+path.name.replace('-gpu.json','-cpu.json'),gpu_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        generated_assets=sorted(p.name for p in out.glob('*.svg'))),indent=2)+'\n')
    print(f'Generated five SVG charts from {path.name}')


if __name__=='__main__':main(sys.argv[1])
