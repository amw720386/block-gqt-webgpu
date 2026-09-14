"""Validate raw statistics, allocation accounting and chart provenance."""
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main(path):
    path=Path(path)
    for current in (path,Path(str(path).replace('-gpu.json','-cpu.json'))):
        report=json.loads(current.read_text())
        assert report['passed'] and len(report['git_commit'])==40 and report['chrome_version']
        assert report['os'] and report['environment']['adapter_limits'] and report['source_sha256']
        for record in report['records']:
            assert len(record['upstream_commit'])==40
            for t in record['timings']:
                samples=t['samples'];assert len(samples)==t['repetition_count'] and t['warmup_count']>0
                expected=dict(median=statistics.median(samples),mean=statistics.mean(samples),stddev=statistics.pstdev(samples),min=min(samples),max=max(samples))
                for key,value in expected.items():assert abs(value-t['statistics'][key])<=1e-9*max(1,abs(value)),(key,value)
                assert min(samples)>=0
            if current==path:
                memory=record['memory'];ledger=memory['allocated_buffers']
                assert all(x['bytes']%4==0 for x in ledger)
                persistent=sum(x['bytes'] for x in ledger if x['role']=='persistent')
                shared=sum(x['bytes'] for x in ledger if x['role']=='shared_metadata')
                assert persistent==memory['persistent_bytes'] and shared==memory['shared_metadata_bytes']
                assert memory['effective_compression_ratio']==memory['baseline_fp16_bytes']/(persistent+shared)
    manifest=json.loads((path.parent.parent/'charts/latest.json').read_text())
    assert manifest['gpu_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    for name in manifest['generated_assets']:
        assert path.name in (path.parent.parent/'charts'/name).read_text(encoding='utf-8')
    print('Raw statistics, byte accounting and five chart provenance links pass.')


if __name__=='__main__':main(sys.argv[1])
