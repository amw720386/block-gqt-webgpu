import {writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {runBrowser,root} from './browser.mjs';
let report;
try {report=await runBrowser('/tests/webgpu/phase1/index.html','phase1Result');}
catch(e) {report={passed:false,error:String(e)};}
await writeFile(resolve(root,'results/webgpu.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,error:report.error,tq:report.tq?.map(({actual,expected,...r})=>r),packing_cases:report.packing_cases?.length},null,2));
if(!report.passed)process.exitCode=1;
