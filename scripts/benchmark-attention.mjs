import {readFile,writeFile,readdir,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {resolve,relative} from 'node:path';
import os from 'node:os';
import {runBrowser,root} from './browser.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
const hashes={};
async function scan(path){
  for(const e of await readdir(path,{withFileTypes:true})){
    if(['__pycache__','block_gtq_webgpu.egg-info'].includes(e.name))continue;
    const p=resolve(path,e.name);if(e.isDirectory())await scan(p);else hashes[relative(root,p).replaceAll('\\','/')]=hash(await readFile(p));
  }
}
for(const dir of ['src','tests','scripts','fixtures','vendor'])await scan(resolve(root,dir));
for(const file of ['package.json','package-lock.json','tsconfig.json','requirements.txt','pyproject.toml','benchmarks/attention.ts','benchmarks/plot_attention.py'])hashes[file]=hash(await readFile(resolve(root,file)));
const gateFiles=['results/attention-compressed.json','results/attention-cpu-verification.json'];
const gates={};for(const file of gateFiles){const data=await readFile(resolve(root,file));if(!JSON.parse(data).passed)throw new Error(`Correctness gate failed: ${file}`);gates[file]=hash(data);}
const result=await runBrowser('/tests/webgpu/attention-bench.html','attentionBenchmark');
if(!result.passed)throw new Error(JSON.stringify(result));
const time=new Date().toISOString(),run=time.replaceAll(':','-').replace('.','-');
const report={schema_version:1,kind:'synthetic-attention',run_id:run,created_utc:time,
  git_commit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),
  git_dirty:!!execFileSync('git',['status','--porcelain'],{cwd:root,encoding:'utf8'}).trim(),source_sha256:hashes,correctness_gate_sha256:gates,
  os:{platform:os.platform(),release:os.release(),version:os.version(),arch:os.arch(),cpu:os.cpus()[0]?.model},
  measurement:'GPU timestamps from first compute pass to final attention output, including full K reconstruction for compressed path. Excludes append/encoding, uploads, compilation, command construction, submission and readback. Resident repeated inputs; no model throughput interpretation.',
  memory:'Actual GPUBuffer.size. Persistent K is codes plus corrected FP16 norms; shared tables and all runtime scratch listed separately. Both paths coexist and share V. Temporary append and measurement buffers excluded from runtime ledger; timestamp buffers add 32 bytes. Driver/query/pipeline allocation sizes unavailable; not peak GPU memory.',
  ...result};
const file=resolve(root,`benchmarks/raw/${run}-attention.json`);await mkdir(resolve(root,'benchmarks/raw'),{recursive:true});
await writeFile(file,JSON.stringify(report,null,2)+'\n');
console.log(file);
execFileSync(process.env.BLOCKGTQ_PYTHON||'python',[resolve(root,'benchmarks/plot_attention.py'),file],{cwd:root,stdio:'inherit'});
