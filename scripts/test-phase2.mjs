import {mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {runBrowser,root} from './browser.mjs';
let report;
try {report=await runBrowser('/tests/webgpu/phase2/phase2.html','phase2Result');}
catch(e) {report={passed:false,error:String(e)};}
await mkdir(resolve(root,'results'),{recursive:true});
await writeFile(resolve(root,'results/phase2-webgpu.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,error:report.error,cases:report.cases?.map(c=>({name:c.name,
  packed_bytes_exact:c.packed_bytes_exact,norm_bit_mismatches:c.norm_bit_mismatches,roundtrip:c.roundtrip}))},null,2));
if(!report.passed)process.exitCode=1;
