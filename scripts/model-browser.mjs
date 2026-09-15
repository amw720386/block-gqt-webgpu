import {createServer} from 'node:http';
import {access,readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {resolve,sep,extname,relative} from 'node:path';
import {chromium} from 'playwright';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import os from 'node:os';
const root=process.cwd(),mode=process.argv[2]||'model-test';
const modelManifest=resolve(root,'models/smollm2/manifest.json'),modelWeights=resolve(root,'models/smollm2/fp16.safetensors');
async function modelReady(){try{await access(modelManifest);await access(modelWeights);return true;}catch{return false;}}
async function ensureModelWeights(){
  if(await modelReady())return;
  console.log('SmolLM2 weights not found under models/smollm2/. Running scripts/setup_model.py (one-time ~269 MB download)...');
  execFileSync(process.env.BLOCKGTQ_PYTHON||'python',[resolve(root,'scripts/setup_model.py')],{cwd:root,stdio:'inherit'});
  if(!await modelReady())throw new Error('Model setup failed. Install Python deps (requirements-model.txt) and run: python scripts/setup_model.py');
}
await ensureModelWeights();
const sha=bytes=>createHash('sha256').update(bytes).digest('hex'),source_sha256={};
async function scan(path){for(const e of await readdir(path,{withFileTypes:true})){if(e.name==='__pycache__'||e.name.endsWith('.egg-info'))continue;const p=resolve(path,e.name);if(e.isDirectory())await scan(p);else source_sha256[relative(root,p).replaceAll('\\','/')]=sha(await readFile(p));}}
for(const folder of ['src','tests','scripts','fixtures','vendor'])await scan(resolve(root,folder));
for(const file of ['models/model.json','models/smollm2/manifest.json','package.json','package-lock.json','requirements.txt','requirements-model.txt','benchmarks/model.ts','benchmarks/model-decode.ts','benchmarks/plot_model.py'])source_sha256[file]=sha(await readFile(resolve(root,file)));
const baseline=JSON.parse(await readFile(resolve(root,'benchmarks/raw/2026-09-15T09-27-00-112Z-phase4-phase4-bench.json')));
for(const [file,hash] of Object.entries(baseline.source_sha256))if((file.startsWith('src/attention/')||file.startsWith('src/runtime/'))&&source_sha256[file]!==hash)throw new Error(`Preserved baseline changed: ${file}`);
const correctness_gates={};
if(mode==='model-benchmark'||mode==='model-decode-benchmark'){
  for(const file of ['model-test.json','model-compressed-test.json','model-primitives.json','model-codec-test.json','model-live-test.json','model-cpu-gate.json','model-compressed-cpu-gate.json']){
    const path=`results/model/${file}`,bytes=await readFile(resolve(root,path)),r=JSON.parse(bytes);if(!r.passed)throw new Error(`Failed correctness gate ${file}`);correctness_gates[path]=sha(bytes);
    if(file==='model-test.json'||file==='model-compressed-test.json')for(const [name,hash] of Object.entries(source_sha256))if(name.startsWith('src/')&&r.source_sha256?.[name]!==hash)throw new Error(`Untested source ${name}`);
  }
  for(const [gpu,cpu] of [['model-test.json','model-cpu-gate.json'],['model-compressed-test.json','model-compressed-cpu-gate.json']]){
    const gate=JSON.parse(await readFile(resolve(root,'results/model',cpu)));if(gate.gpu_report_sha256!==correctness_gates['results/model/'+gpu])throw new Error(`Stale CPU gate ${cpu}`);
  }
}
const server=createServer(async(req,res)=>{
  try{const target=resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
    if(!['src','dist','tests/webgpu','fixtures','models','examples','benchmarks/raw'].some(d=>target.startsWith(resolve(root,d)+sep))){res.writeHead(403).end();return;}
    const bytes=await readFile(target);res.writeHead(200,{'Content-Type':({'.html':'text/html','.js':'text/javascript','.json':'application/json','.wgsl':'text/plain'})[extname(target)]||'application/octet-stream','Cache-Control':'no-store'}).end(bytes);
  }catch{res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:process.env.BROWSER_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try{
  const page=await browser.newPage();page.on('console',msg=>console.log(msg.text()));const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${server.address().port}/tests/webgpu/${mode}.html`);await page.waitForFunction(()=>window.modelResult!==undefined,undefined,{timeout:60000});
  const result=await page.evaluate(async()=>await window.modelResult);
  await mkdir(resolve(root,'results/model'),{recursive:true});
  for(const c of result.cases||[])if(c.capture?.base64){const bytes=Buffer.from(c.capture.base64,'base64'),binary=`results/model/${mode}-${c.name}.bin`;await writeFile(resolve(root,binary),bytes);delete c.capture.base64;c.capture.binary=binary;c.capture.sha256=createHash('sha256').update(bytes).digest('hex');}
  const report={schema_version:1,created_utc:new Date().toISOString(),source_sha256,correctness_gates,git_commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),git_dirty:true,
    os:{platform:os.platform(),release:os.release(),version:os.version(),arch:os.arch(),cpu:os.cpus()[0]?.model},...result,chrome_version:browser.version(),headless:true,page_errors:errors,passed:result.passed&&errors.length===0};
  await mkdir(resolve(root,'results/model'),{recursive:true});await writeFile(resolve(root,`results/model/${mode}.json`),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.passed,cases:report.cases?.length,error:report.error}));if(!report.passed)process.exitCode=1;
  if(mode==='model-benchmark'){
    const raw=`benchmarks/raw/${report.created_utc.replaceAll(':','-').replace('.','-')}-model.json`;await writeFile(resolve(root,raw),JSON.stringify(report,null,2)+'\n');console.log(raw);
    if(report.passed)execFileSync(process.env.BLOCKGTQ_PYTHON||'python',[resolve(root,'benchmarks/plot_model.py'),resolve(root,raw)],{stdio:'inherit'});
  }
  if(mode==='model-decode-benchmark'){
    const raw=`benchmarks/raw/${report.created_utc.replaceAll(':','-').replace('.','-')}-model-decode.json`;await writeFile(resolve(root,raw),JSON.stringify(report,null,2)+'\n');console.log(raw);
  }
}finally{await browser.close();await new Promise(r=>server.close(r));}
