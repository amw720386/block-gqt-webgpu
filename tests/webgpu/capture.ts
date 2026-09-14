// Test-only readback serialization. Never imported by runtime/library modules.
export function capture(arrays:Record<string,Float32Array>){
  const descriptors:Record<string,{offset:number;length:number}>={};let size=0;
  for(const [name,array] of Object.entries(arrays)){descriptors[name]={offset:size,length:array.length};size+=array.byteLength;}
  const bytes=new Uint8Array(size);
  for(const [name,array] of Object.entries(arrays))bytes.set(new Uint8Array(array.buffer,array.byteOffset,array.byteLength),descriptors[name].offset);
  let encoded="";for(let i=0;i<bytes.length;i+=8192)encoded+=String.fromCharCode(...bytes.subarray(i,i+8192));
  return {arrays:descriptors,base64:btoa(encoded),byte_length:size};
}
