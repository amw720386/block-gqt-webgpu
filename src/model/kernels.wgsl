struct P {rows:u32,n:u32,k:u32,mode:u32,start:u32,heads:u32,capacity:u32,pad:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> x:array<u32>;
@group(0) @binding(2) var<storage,read> w:array<u32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<storage,read> extra:array<f32>;
fn xf(i:u32)->f32{return bitcast<f32>(x[i]);}
fn wf(i:u32)->f32{return half_value((w[i/2u]>>((i%2u)*16u))&65535u);}
@compute @workgroup_size(128)
fn embedding(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<p.rows*p.n){y[i]=wf(x[i/p.n]*p.n+i%p.n);}}
var<workgroup> aa:array<f32,256>;
var<workgroup> bb:array<f32,256>;
// 16x16 output tile, four outputs per invocation; FP16 weights, FP32 sums.
@compute @workgroup_size(8,8)
fn matmul(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) lane:vec3<u32>){
  let m=group.y*16u+lane.y*2u;let n=group.x*16u+lane.x*2u;let tid=lane.y*8u+lane.x;
  var sums=vec4<f32>(0.0);
  for(var base=0u;base<p.k;base+=16u){
    for(var j=tid;j<256u;j+=64u){
      let ar=group.y*16u+j/16u;let ak=base+j%16u;aa[j]=0.0;if(ar<p.rows&&ak<p.k){aa[j]=xf(ar*p.k+ak);}
      let bk=base+j/16u;let bn=group.x*16u+j%16u;bb[j]=0.0;if(bn<p.n&&bk<p.k){bb[j]=wf(bn*p.k+bk);}
    }
    workgroupBarrier();
    for(var t=0u;t<16u;t++){
      let a0=aa[lane.y*32u+t];let a1=aa[lane.y*32u+16u+t];let b0=bb[t*16u+lane.x*2u];let b1=bb[t*16u+lane.x*2u+1u];
      sums+=vec4<f32>(a0*b0,a0*b1,a1*b0,a1*b1);
    }
    workgroupBarrier();
  }
  if(m<p.rows&&n<p.n){y[m*p.n+n]=sums.x;}if(m<p.rows&&n+1u<p.n){y[m*p.n+n+1u]=sums.y;}
  if(m+1u<p.rows&&n<p.n){y[(m+1u)*p.n+n]=sums.z;}if(m+1u<p.rows&&n+1u<p.n){y[(m+1u)*p.n+n+1u]=sums.w;}
}
var<workgroup> reduction:array<f32,128>;
@compute @workgroup_size(128)
fn norm(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let row=group.x;var sum=0.0;for(var i=lane;i<p.n;i+=128u){let v=xf(row*p.n+i);sum+=v*v;}
  reduction[lane]=sum;workgroupBarrier();for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){reduction[lane]+=reduction[lane+stride];}workgroupBarrier();}
  let scale=inverseSqrt(reduction[0]/f32(p.n)+1e-5);
  for(var i=lane;i<p.n;i+=128u){y[row*p.n+i]=xf(row*p.n+i)*scale*wf(i);}
}
@compute @workgroup_size(128)
fn element(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.rows*p.n){return;}let a=xf(i);
  if(p.mode==0u){y[i]=a+extra[i];}else{y[i]=(a/(1.0+exp(-a)))*extra[i];}
}
@compute @workgroup_size(128)
fn rope(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.rows*p.heads*32u){return;}
  let row=i/(p.heads*32u);let head=(i/32u)%p.heads;let pair=i%32u;let base=(row*p.heads+head)*64u;
  let theta=f32(p.start+row)*pow(100000.0,-f32(pair)/32.0);let c=cos(theta);let s=sin(theta);
  let a=xf(base+pair);let b=xf(base+pair+32u);y[base+pair]=a*c-b*s;y[base+pair+32u]=b*c+a*s;
}
