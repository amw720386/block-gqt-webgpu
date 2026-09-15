struct P {rows:u32,start:u32,capacity:u32,heads:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> cache:array<u32>;
@compute @workgroup_size(128)
fn append(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.rows*p.heads*32u){return;}let row=i/(p.heads*32u);let head=(i/32u)%p.heads;let pair=i%32u;
  let at=(head*p.capacity+p.start+row)*32u+pair;
  cache[at]=half_bits(input[i*2u])|(half_bits(input[i*2u+1u])<<16u);
}
