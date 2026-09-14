struct Params {
  rows: u32, dim: u32, bits: u32, row_bytes: u32,
  correction: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> norms: array<f32>;
@group(0) @binding(2) var<storage, read> rotation: array<f32>;
@group(0) @binding(3) var<storage, read> centroids: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read_write> codes: array<u32>;
@group(0) @binding(6) var<uniform> p: Params;

fn byte_at(address: u32) -> u32 {
  return (packed[address / 4u] >> (8u * (address % 4u))) & 255u;
}

// Intentionally serial within each tiny row: this is the correctness baseline.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= p.rows) { return; }
  var decoded: array<f32, 16>;
  var squared_length = 0.0;
  for (var i = 0u; i < p.dim; i++) {
    let bit = i * p.bits;
    let address = row * p.row_bytes + bit / 8u;
    let shift = bit % 8u;
    var raw = byte_at(address);
    if (shift + p.bits > 8u) { raw |= byte_at(address + 1u) << 8u; }
    let code = (raw >> shift) & ((1u << p.bits) - 1u);
    codes[row * p.dim + i] = code;
    decoded[i] = centroids[code];
    squared_length += decoded[i] * decoded[i];
  }
  var divisor = 1.0;
  let length = sqrt(squared_length);
  if (p.correction != 0u && length > 1e-10) { divisor = length; }
  for (var i = 0u; i < p.dim; i++) { decoded[i] /= divisor; }
  // Upstream inverse transform is decoded @ rotation, then original norm.
  for (var j = 0u; j < p.dim; j++) {
    var value = 0.0;
    for (var i = 0u; i < p.dim; i++) { value += decoded[i] * rotation[i * p.dim + j]; }
    output[row * p.dim + j] = value * norms[row];
  }
}
