export type Block = {
  rows: number; dim: number; bits: number; rowBytes: number; correction: boolean;
  packed: Uint8Array; norms: Float32Array; rotation: Float32Array; centroids: Float32Array;
};

export async function createReconstructor(device: GPUDevice) {
  const response = await fetch("/tests/webgpu/phase1/reconstruct.wgsl");
  if (!response.ok) throw new Error("Shader fetch failed");
  const shader = device.createShaderModule({ code: await response.text() });
  const diagnostics = await shader.getCompilationInfo();
  const errors = diagnostics.messages.filter(m => m.type === "error");
  if (errors.length) throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module: shader, entryPoint: "main" },
  });
  return async (b: Block) => {
    const expectedRowBytes = b.bits === 3 ? Math.ceil(b.dim / 8) * 3 : Math.ceil(b.dim * b.bits / 8);
    if (![b.rows, b.dim, b.bits, b.rowBytes].every(Number.isSafeInteger) ||
        b.rows < 1 || b.rows > device.limits.maxComputeWorkgroupsPerDimension || b.dim < 1 || b.dim > 16 ||
        b.bits < 1 || b.bits > 4 || b.rowBytes !== expectedRowBytes ||
        b.packed.length !== b.rows * b.rowBytes || b.norms.length !== b.rows ||
        b.rotation.length !== b.dim * b.dim || b.centroids.length !== 2 ** b.bits)
      throw new Error("Invalid tiny block contract");
    const buffers: GPUBuffer[] = [];
    const make = (size: number, usage: GPUBufferUsageFlags) => {
      const buffer = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });
      buffers.push(buffer); return buffer;
    };
    const upload = (data: Uint8Array | Uint32Array | Float32Array, usage = GPUBufferUsage.STORAGE) => {
      const buffer = make(data.byteLength, usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const padded = new Uint8Array(buffer.size);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      device.queue.writeBuffer(buffer, 0, padded); return buffer;
    };
    device.pushErrorScope("validation");
    let scopePopped = false;
    try {
      const packed = upload(b.packed);
      const output = make(b.rows * b.dim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const codes = make(output.size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const params = upload(new Uint32Array([b.rows, b.dim, b.bits, b.rowBytes, +b.correction, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const bindings = [packed, upload(b.norms), upload(b.rotation), upload(b.centroids), output, codes, params];
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const sources = [output, codes, packed];
      const readbacks = sources.map(x => make(x.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(b.rows); pass.end();
      sources.forEach((x, i) => encoder.copyBufferToBuffer(x, 0, readbacks[i], 0, x.size));
      device.queue.submit([encoder.finish()]);
      await Promise.all(readbacks.map(x => x.mapAsync(GPUMapMode.READ)));
      const result = readbacks.map(x => x.getMappedRange().slice(0));
      const error = await device.popErrorScope(); scopePopped = true;
      if (error) throw new Error(error.message);
      return { values: new Float32Array(result[0]), codes: new Uint32Array(result[1]), packed: new Uint8Array(result[2]) };
    } finally {
      if (!scopePopped) await device.popErrorScope();
      buffers.forEach(x => x.destroy());
    }
  };
}
