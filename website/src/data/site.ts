export const repositoryUrl = "https://github.com/OrbitInfra/OrbitKV";

export const sourceUrl = (path: string) =>
  `${repositoryUrl}/blob/${import.meta.env.PUBLIC_SOURCE_REF}/${path}`;
export const localUrl = (path: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Docs", href: "/docs/" },
  { label: "Architecture", href: "/architecture/" },
  { label: "Integration", href: "/integration/" },
];

export const layers = [
  {
    name: "orbitkv-state",
    role: "Name the state.",
    detail:
      "Versioned model/storage keys and compiled rules for demanded page ranges and leased-evidence validation, consumed by engine adapters within their qualified model profiles.",
    path: "crates/orbitkv-state/src/lib.rs",
  },
  {
    name: "orbitkv-channel",
    role: "Control the Cache Manager.",
    detail:
      "Rust ownership of query revisions, warming and transfer waits; versioned iceoryx2 requests plus UDS lifecycle for both adapters.",
    path: "crates/orbitkv-channel/src/lib.rs",
  },
  {
    name: "orbitkv-core",
    role: "Own the blocks.",
    detail:
      "KV blocks, leases, tiers, batched GPU codecs with reusable workspace, CPU SIMD, and encoded cuFile storage.",
    path: "crates/orbitkv-core/src/lib.rs",
  },
  {
    name: "orbitkv-catalog",
    role: "Locate replicas.",
    detail:
      "Scoped local inventory indexes and member admission; the server owns inventory streams and leased etcd membership.",
    path: "crates/orbitkv-catalog/src/lib.rs",
  },
  {
    name: "orbitkv-transfer",
    role: "Move the bytes.",
    detail:
      "Pinned Mooncake TENT with registered-region ownership and terminal batch drain for shared-cache transfers between Managers.",
    path: "crates/orbitkv-transfer/README.md",
  },
  {
    name: "Cache Manager",
    role: "Share the cache.",
    detail:
      "Independent per-node service sharing external DRAM/SSD capacity across attached engines, with automatic SSD selection, health and peer transfer control.",
    path: "crates/orbitkv-server/README.md",
  },
  {
    name: "orbitkv.sglang",
    role: "Link SGLang GPU pages.",
    detail:
      "Official UnifiedRadixCache external linker with validated selected-range reads; public lifecycle and complete hybrid model recovery remain open.",
    path: "python/orbitkv/sglang/linker.py",
  },
  {
    name: "orbitkv.vllm",
    role: "Connect vLLM.",
    detail:
      "Independent cache connector; native NIXL and MultiConnector handle candidate P/D composition.",
    path: "python/orbitkv/vllm/connector.py",
  },
];
