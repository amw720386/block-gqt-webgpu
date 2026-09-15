import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
const root=process.cwd(),mime={'.html':'text/html','.js':'text/javascript','.wgsl':'text/plain','.json':'application/json','.bin':'application/octet-stream'};
createServer(async(req,res)=>{
  try{
    const path=new URL(req.url,'http://localhost').pathname;
    const target=resolve(root,'.'+(path==='/'?'/examples/attention.html':decodeURIComponent(path)));
    if(!['src','tests/webgpu','examples','dist','fixtures'].some(dir=>target.startsWith(resolve(root,dir)+sep))){res.writeHead(403).end();return;}
    const body=await readFile(target);res.writeHead(200,{'Content-Type':mime[extname(target)]||'application/octet-stream'}).end(body);
  }catch{res.writeHead(404).end();}
}).listen(8080,'127.0.0.1',()=>console.log('Synthetic attention example: http://127.0.0.1:8080'));
