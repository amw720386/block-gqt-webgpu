type Descriptor = { dtype: "f32" | "u32" | "u8" | "f16" | "u16"; shape: number[]; offset: number; nbytes: number };
export type Manifest = {
  version: number; endianness: string; binary: string; byte_length: number;
  sha256: string; upstream_commit: string; arrays: Record<string, Descriptor>;
  rows: number; dim: number; bits: number; row_bytes: number;
  coordinate_space: string; packing: string; norm_dtype: string;
  tolerance: { atol: number; rtol: number };
  cases: { name: string; bits: number; dim: number }[];
  name: string; n_groups: number; norm_stride: number; capacity: number;
  nopack_start: number; max_centroids: number; logical_row_bytes: number;
  average_bits: number; norm_conversion_max_abs: number; quantization_max_abs: number;
};

export async function loadFixture(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fixture HTTP ${response.status}`);
  const manifest: Manifest = await response.json();
  if (![1, 2].includes(manifest.version) || manifest.endianness !== "little") throw new Error("Unsupported fixture");
  if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) throw new Error("Little-endian host required");
  const payload = await fetch(new URL(manifest.binary, new URL(url, location.href)));
  if (!payload.ok) throw new Error(`Payload HTTP ${payload.status}`);
  const binary = await payload.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", binary))]
    .map(x => x.toString(16).padStart(2, "0")).join("");
  if (hash !== manifest.sha256 || binary.byteLength !== manifest.byte_length) throw new Error("Fixture integrity failure");
  function bytes(name: string, dtype: Descriptor["dtype"]) {
    const d = manifest.arrays[name];
    if (!d || d.dtype !== dtype) throw new Error(`Missing or invalid ${name}`);
    const size = dtype === "u8" ? 1 : ["f16", "u16"].includes(dtype) ? 2 : 4;
    if (!d.shape.every(x => Number.isSafeInteger(x) && x >= 0) ||
        !Number.isSafeInteger(d.offset) || d.offset < 0 || d.offset % 4 ||
        d.nbytes !== d.shape.reduce((a, b) => a * b, 1) * size ||
        d.offset + d.nbytes > binary.byteLength) throw new Error(`Invalid descriptor ${name}`);
    return binary.slice(d.offset, d.offset + d.nbytes);
  }
  return {
    manifest,
    f32: (name: string) => new Float32Array(bytes(name, "f32")),
    u32: (name: string) => new Uint32Array(bytes(name, "u32")),
    u8: (name: string) => new Uint8Array(bytes(name, "u8")),
    u16: (name: string) => new Uint16Array(bytes(name, "u16")),
    halfBits: (name: string) => new Uint16Array(bytes(name, "f16")),
  };
}
