// Exact tokenizer.json BPE data; narrow implementation of this model's
// Digits(individual) -> GPT-2 ByteLevel pre-tokenizer. No network service.
export class Tokenizer {
  private bytes:string[]=[];private inverse=new Map<string,number>();private ranks=new Map<string,number>();private vocabulary:Record<string,number>;private tokens:string[]=[];private specials=new Map<string,number>();private sequenceByteLevel=false;
  private constructor(t:any,private template:(prompt:string)=>string,private modelVocabularySize?:number){
    const pre=t.pre_tokenizer,smol=pre?.pretokenizers?.[0]?.type==="Digits"&&pre.pretokenizers[0].individual_digits===true;
    this.sequenceByteLevel=pre?.type==="Sequence"&&pre.pretokenizers?.[0]?.type==="Split"&&pre.pretokenizers?.[1]?.type==="ByteLevel";
    const byteLevelConfig=this.sequenceByteLevel&&t.normalizer?.type==="NFC"&&t.post_processor?.type==="ByteLevel";
    if(t.model.type!=="BPE"||(!smol&&!byteLevelConfig))throw new Error("Unsupported tokenizer configuration");
    this.vocabulary=t.model.vocab;for(const [token,id] of Object.entries(this.vocabulary))this.tokens[id as number]=token;
    const ordinary=[...Array.from({length:94},(_,i)=>i+33),...Array.from({length:12},(_,i)=>i+161),...Array.from({length:82},(_,i)=>i+174)];let next=256;
    for(let b=0;b<256;b++){const c=String.fromCodePoint(ordinary.includes(b)?b:next++);this.bytes[b]=c;this.inverse.set(c,b);}
    t.model.merges.forEach((pair:string|[string,string],i:number)=>this.ranks.set(typeof pair==="string"?pair:pair.join(" "),i));
    for(const s of t.added_tokens){if(s.lstrip||s.rstrip||s.single_word||s.normalized)throw new Error("Unsupported added-token behavior");this.specials.set(s.content,s.id);}
  }
  static async load(base="/models/smollm2",template=(prompt:string)=>`<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`,modelVocabularySize?:number){const r=await fetch(`${base}/tokenizer.json`);if(!r.ok)throw new Error("Tokenizer unavailable");return new Tokenizer(await r.json(),template,modelVocabularySize);}
  encode(text:string):Uint32Array{
    const escaped=[...this.specials.keys()].sort((a,b)=>b.length-a.length).map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"));
    const chunks=text.split(new RegExp(`(${escaped.join("|")})`,"gu")),ids:number[]=[];
    for(const chunk of chunks){
      if(!chunk)continue;const special=this.specials.get(chunk);if(special!==undefined){ids.push(special);continue;}
      const normalized=this.sequenceByteLevel?chunk.normalize("NFC"):chunk;
      const matches=this.sequenceByteLevel?normalized.matchAll(/(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu):Array.from(normalized.split(/(\p{N})/gu)).flatMap(part=>Array.from(part.matchAll(/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu)));
      for(const match of matches){
        let pieces=Array.from(new TextEncoder().encode(match[0]),b=>this.bytes[b]);
        while(pieces.length>1){
          let rank=Infinity,index=-1;for(let i=0;i<pieces.length-1;i++){const r=this.ranks.get(pieces[i]+" "+pieces[i+1]);if(r!==undefined&&r<rank){rank=r;index=i;}}
          if(index<0)break;const a=pieces[index],b=pieces[index+1],merged:string[]=[];
          for(let i=0;i<pieces.length;i++){if(i+1<pieces.length&&pieces[i]===a&&pieces[i+1]===b){merged.push(a+b);i++;}else merged.push(pieces[i]);}pieces=merged;
        }
        for(const piece of pieces){const id=this.vocabulary[piece];if(id===undefined)throw new Error("Missing BPE token");ids.push(id);}
      }
    }
    return new Uint32Array(ids);
  }
  decode(ids:ArrayLike<number>,skipSpecial=true){
    const bytes:number[]=[];let out="";const flush=()=>{out+=new TextDecoder().decode(new Uint8Array(bytes));bytes.length=0;};
    for(let i=0;i<ids.length;i++){const id=ids[i],token=this.tokens[id];if(token===undefined){if(this.modelVocabularySize!==undefined&&Number.isInteger(id)&&id>=0&&id<this.modelVocabularySize)continue;throw new Error("Invalid token ID");}if(this.specials.has(token)){if(!skipSpecial){flush();out+=token;}continue;}for(const c of token){const b=this.inverse.get(c);if(b===undefined)throw new Error("Invalid byte token");bytes.push(b);}}
    flush();return out;
  }
  chat(prompt:string){return this.template(prompt);}
}
