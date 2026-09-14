import {runBrowser,root} from './browser.mjs';
import {readFile,writeFile,readdir,mkdir} from 'node:fs/promises';
import {resolve,relative} from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import os from 'node:os';
const mode=process.argv[2]||'profile',sha=b=>createHash('sha256').update(b).digest('hex'),hashes={};
async function scan(path){for(const e of await readdir(path,{withFileTypes:true})){if(e.name==='__pycache__'||e.name.endsWith('.egg-info'))continue;const p=resolve(path,e.name);if(e.isDirectory())await scan(p);else hashes[relative(root,p).replaceAll('\\','/')]=sha(await readFile(p));}}
for(const d of ['src','tests','fixtures','vendor','scripts'])await scan(resolve(root,d));
for(const e of await readdir(resolve(root,'benchmarks')))if(/\.(ts|py)$/.test(e))hashes['benchmarks/'+e]=sha(await readFile(resolve(root,'benchmarks',e)));
for(const f of ['package.json','package-lock.json','tsconfig.json','requirements.txt'])hashes[f]=sha(await readFile(resolve(root,f)));
const old=JSON.parse(await readFile(resolve(root,'benchmarks/raw/2026-09-15T09-27-00-112Z-phase4-phase4-bench.json')));
for(const [p,h] of Object.entries(old.source_sha256))if(p.startsWith('src/')&&hashes[p]!==h)throw new Error(`Phase 3 baseline changed: ${p}`);
const correctness_gates={};
if(mode==='phase4-bench'){
  const records=(await readdir(resolve(root,'benchmarks/raw'))).filter(p=>p.endsWith('-phase4-optimized-test-cpu.json')).sort();
  const file=records.at(-1);if(!file)throw new Error('CPU correctness gate required');
  const bytes=await readFile(resolve(root,'benchmarks/raw',file)),gate=JSON.parse(bytes);
  if(!gate.passed||gate.checks.length!==24)throw new Error('Incomplete CPU correctness gate');
  const gpuBytes=await readFile(resolve(root,gate.gpu_record)),gpu=JSON.parse(gpuBytes);
  if(!gpu.passed||sha(gpuBytes)!==gate.gpu_record_sha256)throw new Error('GPU gate hash mismatch');
  for(const [p,h] of Object.entries(hashes))if(p.startsWith('src/')&&gpu.source_sha256[p]!==h)throw new Error(`Untested implementation: ${p}`);
  correctness_gates['benchmarks/raw/'+file]=sha(bytes);correctness_gates[gate.gpu_record]=sha(gpuBytes);
}
const result=await runBrowser(`/tests/webgpu/${mode}.html`,'phase4Result');
const now=new Date().toISOString(),name=now.replaceAll(':','-').replace('.','-');
for(const c of result.cases||[])if(c.capture?.base64){
  const bytes=Buffer.from(c.capture.base64,'base64'),binary=`results/phase4/${name}-${c.name}.bin`;
  await mkdir(resolve(root,'results/phase4'),{recursive:true});await writeFile(resolve(root,binary),bytes);
  delete c.capture.base64;c.capture.binary=binary;c.capture.sha256=sha(bytes);
}
const file=`benchmarks/raw/${name}-phase4-${mode}.json`;
const report={schema_version:1,mode,created_utc:now,git_commit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),git_dirty:true,
  source_sha256:hashes,correctness_gates,phase3_source_unchanged:true,os:{platform:os.platform(),release:os.release(),version:os.version(),arch:os.arch(),cpu:os.cpus()[0]?.model},...result};
await writeFile(resolve(root,file),JSON.stringify(report,null,2)+'\n');console.log(file);if(!result.passed)process.exitCode=1;
if(mode==='optimized-test'&&result.passed)execFileSync(process.env.BLOCKGTQ_PYTHON||'python',['-m','tests.oracles.verify_phase4',resolve(root,file)],{cwd:root,stdio:'inherit'});
if(mode==='phase4-bench'&&result.passed)execFileSync(process.env.BLOCKGTQ_PYTHON||'python',[resolve(root,'benchmarks/plot_phase4.py'),resolve(root,file)],{cwd:root,stdio:'inherit'});
