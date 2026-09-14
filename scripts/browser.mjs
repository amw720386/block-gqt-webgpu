import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
export const root=fileURLToPath(new URL('../',import.meta.url));

export async function runBrowser(pagePath,resultKey) {
  const mime={'.html':'text/html','.js':'text/javascript','.wgsl':'text/plain','.json':'application/json','.bin':'application/octet-stream'};
  const server=createServer(async(req,res)=>{
    try {
      const path=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
      const target=resolve(root,'.'+path);
      if(!['src','tests/webgpu','examples','dist','fixtures'].some(dir=>target.startsWith(resolve(root,dir)+sep))) {res.writeHead(403).end();return;}
      const body=await readFile(target);res.writeHead(200,{'Content-Type':mime[extname(target)]||'application/octet-stream','Cache-Control':'no-store'}).end(body);
    } catch {res.writeHead(404).end();}
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  let browser;
  try {
    const executablePath=process.env.BROWSER_PATH||(process.platform==='win32'?'C:/Program Files/Google/Chrome/Application/chrome.exe':undefined);
    browser=await chromium.launch({executablePath,headless:true});
    const page=await browser.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${server.address().port}${pagePath}`);
    await page.waitForFunction(key=>key in window,resultKey,{timeout:30000});
    const result=await page.evaluate(async key=>await window[key],resultKey);
    return {...result,chrome_version:browser.version(),headless:true,browser_executable:executablePath||'Playwright default',page_errors:errors,
      passed:result.passed&&errors.length===0};
  } finally {await browser?.close();await new Promise(r=>server.close(r));}
}
