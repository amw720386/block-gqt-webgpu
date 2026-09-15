export type ModelConfig={
  model_type?:unknown;architectures?:unknown;hidden_size?:unknown;intermediate_size?:unknown;num_hidden_layers?:unknown;
  num_attention_heads?:unknown;num_key_value_heads?:unknown;head_dim?:unknown;rope_theta?:unknown;rms_norm_eps?:unknown;
  vocab_size?:unknown;tie_word_embeddings?:unknown;use_sliding_window?:unknown;eos_token_id?:unknown;
};

export interface ModelAdapter {
  readonly family:"smollm2"|"qwen2";readonly name:string;readonly layers:number;readonly hiddenSize:number;readonly intermediateSize:number;
  readonly queryHeads:number;readonly kvHeads:number;readonly headDim:number;readonly ropeTheta:number;readonly rmsNormEps:number;
  readonly vocabSize:number;readonly qkvBias:boolean;readonly eosTokenIds:readonly number[];
  attentionBias(prefix:string,projection:"q_proj"|"k_proj"|"v_proj"):string|undefined;
  chat(prompt:string):string;
}

function integer(config:ModelConfig,key:keyof ModelConfig){const value=config[key];if(!Number.isInteger(value)||Number(value)<1)throw new Error(`Invalid model config: ${String(key)}`);return Number(value);}
function positive(config:ModelConfig,key:keyof ModelConfig){const value=Number(config[key]);if(!Number.isFinite(value)||value<=0)throw new Error(`Invalid model config: ${String(key)}`);return value;}
function common(config:ModelConfig){
  if(config.tie_word_embeddings!==true)throw new Error("Untied output embeddings are unsupported");
  const hiddenSize=integer(config,"hidden_size"),queryHeads=integer(config,"num_attention_heads"),kvHeads=integer(config,"num_key_value_heads");
  const headDim=config.head_dim===undefined?hiddenSize/queryHeads:integer(config,"head_dim");
  if(headDim!==64||hiddenSize!==queryHeads*headDim||queryHeads%kvHeads)throw new Error("Unsupported attention geometry");
  const eos=Array.isArray(config.eos_token_id)?config.eos_token_id:[config.eos_token_id];
  const ropeTheta=positive(config,"rope_theta"),rmsNormEps=positive(config,"rms_norm_eps");
  if(!Number.isInteger(ropeTheta)||![1e-5,1e-6].includes(rmsNormEps))throw new Error("Unsupported RoPE or RMSNorm configuration");
  return {layers:integer(config,"num_hidden_layers"),hiddenSize,intermediateSize:integer(config,"intermediate_size"),queryHeads,kvHeads,headDim,
    ropeTheta,rmsNormEps,vocabSize:integer(config,"vocab_size"),
    eosTokenIds:eos.filter(Number.isInteger).map(Number)};
}

export function modelAdapter(config:ModelConfig):ModelAdapter{
  const values=common(config),modelType=String(config.model_type??"");
  if(modelType==="qwen2"){
    if(config.use_sliding_window===true)throw new Error("Sliding-window Qwen models are unsupported");
    return {...values,family:"qwen2",name:"Qwen2",qkvBias:true,
      attentionBias:(prefix,projection)=>prefix+`self_attn.${projection}.bias`,
      chat:prompt=>`<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`};
  }
  if(modelType==="llama")return {...values,family:"smollm2",name:"SmolLM2",qkvBias:false,
    attentionBias:()=>undefined,
    chat:prompt=>`<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`};
  throw new Error(`Unsupported model type: ${modelType||"missing"}`);
}
