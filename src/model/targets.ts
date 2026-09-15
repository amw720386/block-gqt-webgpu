export type ModelTarget={id:string;label:string;base:string;calibrationBase:string};
export const modelTargets:readonly ModelTarget[]=[
  {id:"smollm2",label:"SmolLM2-135M-Instruct",base:"/models/smollm2",calibrationBase:"/fixtures/model"},
  {id:"qwen2.5-0.5b",label:"Qwen2.5-0.5B-Instruct",base:"/models/qwen2.5-0.5b",calibrationBase:"/models/qwen2.5-0.5b"}
];
export function modelTarget(id:string){const target=modelTargets.find(model=>model.id===id);if(!target)throw new Error(`Unsupported model: ${id}`);return target;}
