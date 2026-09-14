import {execFileSync} from 'node:child_process';
import {readdir,mkdir,copyFile} from 'node:fs/promises';
import {join,dirname} from 'node:path';
execFileSync(process.execPath,['node_modules/typescript/bin/tsc'],{stdio:'inherit'});
async function assets(path){for(const entry of await readdir(path,{withFileTypes:true})){
  const file=join(path,entry.name);if(entry.isDirectory())await assets(file);
  else if(file.endsWith('.wgsl')){const target=join('dist',file);await mkdir(dirname(target),{recursive:true});await copyFile(file,target);}
}}
await assets('src');
