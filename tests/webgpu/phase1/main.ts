import { loadFixture } from "../fixture.js";
import { createReconstructor, type Block } from "./reconstruct.js";

function exact(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string) {
  if (actual.length !== expected.length || Array.from(actual).some((x, i) => x !== expected[i]))
    throw new Error(`${label}: exact comparison failed`);
}
function compare(actual: Float32Array, expected: Float32Array, atol: number, rtol: number) {
  if (actual.length !== expected.length) throw new Error("Float shape mismatch");
  const errors = Array.from(actual, (x, i) => Math.abs(x - expected[i]));
  errors.forEach((e, i) => {
    if (!Number.isFinite(e) || e > atol + rtol * Math.abs(expected[i]))
      throw new Error(`Float mismatch at ${i}: ${actual[i]} vs ${expected[i]}`);
  });
  return { max_abs_reference_error: Math.max(...errors), actual: Array.from(actual), expected: Array.from(expected) };
}
function checkPacked(actual: Uint8Array, expected: Uint8Array) {
  exact(actual.subarray(0, expected.length), expected, "GPU packed readback");
  if (actual.subarray(expected.length).some(x => x !== 0)) throw new Error("Upload alignment is not zero");
}

async function run() {
  if (!navigator.gpu) throw new Error("WebGPU unavailable in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter available");
  const device = await adapter.requestDevice();
  const uncaptured: string[] = [];
  device.addEventListener("uncapturederror", e => uncaptured.push(e.error.message));
  try {
    const reconstruct = await createReconstructor(device);
    const fixture = await loadFixture("/fixtures/tq3.json");
    const m = fixture.manifest;
    if (m.coordinate_space !== "synthetic-rope-free" || m.packing !== "uniform-lsb-byte-v1" || m.norm_dtype !== "f32")
      throw new Error("Unsupported TQ fixture semantics");
    const block: Block = { rows: m.rows, dim: m.dim, bits: m.bits, rowBytes: m.row_bytes,
      correction: true, packed: fixture.u8("packed"), norms: fixture.f32("norms"),
      rotation: fixture.f32("rotation"), centroids: fixture.f32("centroids") };
    const tq = [];
    for (const correction of [true, false]) {
      const actual = await reconstruct({ ...block, correction });
      exact(actual.codes, fixture.u32("indices"), "TQ indices");
      checkPacked(actual.packed, block.packed);
      exact(actual.values.subarray(0, m.dim), new Float32Array(m.dim), "Zero row");
      tq.push({ correction, ...compare(actual.values, fixture.f32(correction ? "reconstructed" : "uncorrected"),
        m.tolerance.atol, m.tolerance.rtol), indices_exact: true, packed_bytes_exact: true });
    }
    const packing = await loadFixture("/fixtures/packing.json");
    const packingCases = [];
    for (const c of packing.manifest.cases) {
      const codes = packing.u8(`${c.name}_codes`);
      const packed = packing.u8(`${c.name}_packed`);
      const rows = codes.length / c.dim;
      const rotation = new Float32Array(c.dim * c.dim);
      for (let i = 0; i < c.dim; i++) rotation[i * c.dim + i] = 1;
      const actual = await reconstruct({ rows, dim: c.dim, bits: c.bits, rowBytes: packed.length / rows,
        correction: false, packed, norms: new Float32Array(rows).fill(1), rotation,
        centroids: Float32Array.from({ length: 2 ** c.bits }, (_, i) => i) });
      exact(actual.codes, codes, c.name); exact(actual.values, codes, `${c.name} identity reconstruction`);
      checkPacked(actual.packed, packed);
      packingCases.push({ name: c.name, indices_exact: true, reconstruction_exact: true, packed_bytes_exact: true });
    }
    await device.queue.onSubmittedWorkDone();
    if (uncaptured.length) throw new Error(uncaptured.join("\n"));
    const info = adapter.info;
    return { passed: true, upstream_commit: m.upstream_commit, fixture_sha256: m.sha256,
      packing_fixture_sha256: packing.manifest.sha256, tolerance: m.tolerance,
      environment: { user_agent: navigator.userAgent, adapter: { vendor: info.vendor,
        architecture: info.architecture, device: info.device, description: info.description,
        is_fallback_adapter: info.isFallbackAdapter }, features: Array.from(device.features),
        limits: { maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxBufferSize: device.limits.maxBufferSize, maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension } },
      tq, packing_cases: packingCases };
  } finally { device.destroy(); }
}

const completion = run().catch(error => ({ passed: false, error: String(error), stack: error?.stack }));
Object.assign(window, { phase1Result: completion });
void completion.then(result => { document.getElementById("result")!.textContent = JSON.stringify(result, null, 2); });
