import {ModelOps} from "../../src/model/ops.js";
import {storage,upload,readBuffer} from "../../src/webgpu/buffers.js";
import {loadFixture} from "./fixture.js";
import {compare} from "./compare.js";
import {modelAdapter} from "../../src/model/adapter.js";
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));const ops=await ModelOps.create(device),f=await loadFixture('/fixtures/model/rope.json'),cases=[];
  try{
    const qwen=modelAdapter({model_type:'qwen2',hidden_size:896,intermediate_size:4864,num_hidden_layers:24,num_attention_heads:14,num_key_value_heads:2,rope_theta:1000000,rms_norm_eps:1e-6,vocab_size:151936,tie_word_embeddings:true,use_sliding_window:false,eos_token_id:151645});
    if(qwen.queryHeads!==14||qwen.kvHeads!==2||qwen.qkvBias!==true||qwen.attentionBias('model.layers.0.','q_proj')!=='model.layers.0.self_attn.q_proj.bias'||qwen.ropeTheta!==1000000||qwen.rmsNormEps!==1e-6)throw new Error('Qwen adapter mismatch');
    let slidingRejected=false;try{modelAdapter({model_type:'qwen2',hidden_size:896,intermediate_size:4864,num_hidden_layers:24,num_attention_heads:14,num_key_value_heads:2,rope_theta:1000000,rms_norm_eps:1e-6,vocab_size:151936,tie_word_embeddings:true,use_sliding_window:true});}catch{slidingRejected=true;}if(!slidingRejected)throw new Error('Sliding-window Qwen accepted');
    for(const c of (f.manifest as any).cases){
    const input=f.f32(c.name+'_x'),x=storage(device,input.byteLength,'RoPE input'),y=storage(device,input.byteLength,'RoPE output'),dummy=storage(device,4,'unused');
    try{
      upload(device,x,input);const cmd=ops.commands();cmd.dispatch('rope',[c.rows,64,0,0,c.start,c.heads,100000],[x,dummy,y,dummy],Math.ceil(c.rows*c.heads*32/128));await cmd.submit();
      const actual=new Float32Array(await readBuffer(device,y));cases.push({name:c.name,error:compare(actual,f.f32(c.name+'_y'))});
      compare(actual,f.f32(c.name+'_y'),[0.001,0.0001]);
    }finally{x.destroy();y.destroy();dummy.destroy();}
  }return {passed:errors.length===0,errors,cases,qwen_adapter:true};}finally{device.destroy();}
}
