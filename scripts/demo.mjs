import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
const root=process.cwd();
createServer(async(req,res)=>{
  try{const path=new URL(req.url,'http://localhost').pathname,target=resolve(root,'.'+(path==='/'?'/examples/demo.html':decodeURIComponent(path)));
    if(!['src','dist','models','fixtures','examples'].some(d=>target.startsWith(resolve(root,d)+sep))){res.writeHead(403).end();return;}
    const data=await readFile(target);res.writeHead(200,{'Content-Type':({'.html':'text/html','.js':'text/javascript','.json':'application/json','.wgsl':'text/plain'})[extname(target)]||'application/octet-stream'}).end(data);
  }catch{res.writeHead(404).end();}
}).listen(8080,'127.0.0.1',()=>console.log('Model demo: http://127.0.0.1:8080'));
