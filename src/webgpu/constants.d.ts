// TypeScript 7's DOM library includes the interfaces but omits these namespaces.
// These are browser-provided constants; no runtime values are substituted.
declare const GPUBufferUsage: {
  readonly MAP_READ: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly UNIFORM: number;
  readonly STORAGE: number;
  readonly QUERY_RESOLVE: number;
};
declare const GPUMapMode: { readonly READ: number };
