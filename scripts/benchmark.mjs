import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import os from 'node:os';
import {resolve} from 'node:path';
import {runBrowser,root} from './browser.mjs';

const git=(...args)=>execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
const commit=git('rev-parse','HEAD');
const paths=git('ls-files','src','tests','scripts','vendor','fixtures','package.json','package-lock.json','tsconfig.json','requirements.txt').split('\n').filter(Boolean).sort();
const hashes={};
for(const path of paths) hashes[path]=createHash('sha256').update(await readFile(resolve(root,path))).digest('hex');
const report=await runBrowser('/tests/webgpu/phase2/bench.html','benchmarkResult');
if(!report.passed) throw new Error(JSON.stringify(report));
const runId=new Date().toISOString().replace(/[:.]/g,'-');
const raw={schema_version:1,run_id:runId,created_at:new Date().toISOString(),git_commit:commit,
  git_dirty:git('status','--porcelain')!=='',source_sha256:hashes,
  os:{platform:os.platform(),release:os.release(),version:os.version(),arch:os.arch(),cpu:os.cpus()[0]?.model},
  ...report};
const path=resolve(root,`benchmarks/raw/${runId}-gpu.json`);
await mkdir(resolve(root,'benchmarks/raw'),{recursive:true});
await writeFile(path,JSON.stringify(raw,null,2)+'\n');
const python=process.env.BLOCKGTQ_PYTHON||'python';
execFileSync(python,['-m','tests.oracles.benchmark',path],{cwd:root,env:process.env,stdio:'inherit'});
execFileSync(python,['benchmarks/plot.py',path],{cwd:root,env:process.env,stdio:'inherit'});
console.log(`Saved GPU/CPU raw diagnostics and charts for ${runId}`);
