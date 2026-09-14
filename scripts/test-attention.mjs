import {writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {root,runBrowser} from './browser.mjs';
const compressed=process.argv.includes('--compressed');
let report;
try{report=await runBrowser(`/tests/webgpu/attention.html?compressed=${+compressed}`,'attentionResult');}
catch(e){report={passed:false,error:String(e)};}
for(const c of report.cases||[]){
  const bytes=Buffer.from(c.capture.base64,'base64');
  await mkdir(resolve(root,'results/attention-gpu'),{recursive:true});
  const binary=`attention-gpu/${c.name}-${compressed?'compressed':'baseline'}.bin`;
  await writeFile(resolve(root,'results',binary),bytes);
  delete c.capture.base64;c.capture.binary=binary;c.capture.sha256=createHash('sha256').update(bytes).digest('hex');
}
await writeFile(resolve(root,`results/attention-${compressed?'compressed':'baseline'}.json`),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,error:report.error,cases:report.cases?.map(c=>({name:c.name,comparisons:c.comparisons.map(r=>({path:r.path,causal:r.causal,output_error:r.numerical_error.output.max_absolute_error}))}))},null,2));
if(!report.passed)process.exitCode=1;
