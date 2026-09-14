export function compare(actual:Float32Array,expected:Float32Array,tolerance?:[number,number],bound?:Float64Array){
  if(actual.length!==expected.length)throw new Error(`Shape mismatch: ${actual.length} != ${expected.length}`);
  let maximum=0,sum=0,scale=0,worst=0,masked=0;
  for(let i=0;i<actual.length;i++){
    if(actual[i]===-Infinity&&expected[i]===-Infinity){masked++;continue;}
    const error=Math.abs(actual[i]-expected[i]);
    if(!Number.isFinite(error))throw new Error(`Nonfinite mismatch at ${i}: ${actual[i]} vs ${expected[i]}`);
    if(error>maximum){maximum=error;worst=i;}sum+=error*error;scale+=expected[i]*expected[i];
    if(tolerance&&error>tolerance[0]+tolerance[1]*Math.abs(expected[i])+(bound?.[i]??0))throw new Error(`Mismatch at ${i}: ${actual[i]} vs ${expected[i]}; error ${error}, limit ${tolerance}`);
  }
  return {max_absolute_error:maximum,relative_l2_error:Math.sqrt(sum)/Math.max(Math.sqrt(scale),1e-30),worst_index:worst,
    worst_actual:actual[worst],worst_expected:expected[worst],masked_positions:masked};
}
export function exact(a:ArrayLike<number>,b:ArrayLike<number>,label:string):void{
  if(a.length!==b.length)throw new Error(`${label} length mismatch`);
  for(let i=0;i<a.length;i++)if(a[i]!==b[i])throw new Error(`${label} differs at ${i}`);
}
