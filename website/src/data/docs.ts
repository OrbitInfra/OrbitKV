export const docGroups = [
  {
    title: "Getting started",
    description: "Build a wheel, connect an engine, and verify an external hit.",
    items: [
      { slug: "goals", title: "Overview" },
      { slug: "features", title: "Key features" },
      { slug: "single-node", title: "Installation & quickstarts" },
    ],
  },
  {
    title: "Supported engines and models",
    description: "Check release pins, implemented layouts and qualification limits.",
    items: [
      { slug: "compatibility", title: "Engine support matrix" },
      { slug: "models", title: "Model recipes & limits" },
    ],
  },
  {
    title: "Configuration",
    description: "Configure the Manager and engine adapters; keep preparation opt-in.",
    items: [
      { slug: "server", title: "Manager configuration" },
      { slug: "adapters", title: "Engine adapters: configuration & contracts" },
      { slug: "request-preparation", title: "Request preparation" },
      { slug: "queued-warming", title: "Queued-request warming" },
    ],
  },
  {
    title: "Storage tiers",
    description: "Choose DRAM retention, SSD file transfers and storage formats.",
    items: [
      { slug: "cache-policies", title: "DRAM retention & SSD admission" },
      { slug: "gds", title: "SSD / io_uring & GDS" },
      { slug: "storage-formats", title: "Formats & compression" },
    ],
  },
  {
    title: "Distributed KV cache",
    description: "Share compatible replicas through OrbiTE and authoritative grants.",
    items: [
      { slug: "p2p", title: "P2P deployment" },
      { slug: "peer-control", title: "Peer control & source grants" },
      { slug: "distributed-cache", title: "Distributed global index" },
      { slug: "distributed-design", title: "Inventory & recovery design" },
    ],
  },
  {
    title: "Disaggregated prefill",
    description: "Compose cache reuse with the inference engine's native P/D transport.",
    items: [{ slug: "pd", title: "Prefill/decode composition" }],
  },
  {
    title: "Production",
    description: "Plan deployment and operations; these guides do not imply qualification.",
    items: [
      { slug: "deployment", title: "Deployment requirements & recipes" },
      { slug: "metrics", title: "Metrics & observability" },
      { slug: "fault-qualification", title: "Fault behavior & qualification" },
      { slug: "shared-cache-qualification", title: "Shared-cache qualification" },
      { slug: "releases", title: "Release artifacts & installation gates" },
    ],
  },
  {
    title: "Benchmarks",
    description: "Read historical results with their limits and reproduce matched measurements.",
    items: [
      { slug: "performance", title: "Performance & historical results" },
      { slug: "benchmark-evidence", title: "Methodology & evidence" },
      { slug: "kernel-optimization", title: "CUDA kernel measurements" },
      { slug: "ssd-performance", title: "SSD recovery methodology" },
      { slug: "distributed-comparison", title: "Distributed cache comparisons" },
    ],
  },
  {
    title: "Developer guide",
    description: "Follow ownership and state contracts, then the single execution queue. Adapter contracts are under Configuration.",
    items: [
      { slug: "architecture", title: "System architecture" },
      { slug: "transport", title: "Transport & completion" },
      { slug: "engine-local-restore", title: "Engine-local restore" },
      { slug: "state-identity", title: "State identity" },
      { slug: "state-planning", title: "Demand & transfer planning" },
      { slug: "engine-release-audit", title: "Released interface audit" },
      { slug: "vllm-request-state-machine", title: "vLLM request lifecycle" },
      { slug: "hybrid-recovery", title: "Hybrid recovery contracts" },
      { slug: "hybrid-cache-design", title: "Hybrid cache design (unexecuted)" },
      { slug: "design-v2", title: "Design v2: state compiler & agent cache (proposal)" },
      { slug: "contributing", title: "Contributing" },
      { slug: "completion-plan", title: "Completion plan" },
    ],
  },
];

// Every canonical document has one group; sidebar, index and search share this map.
export const docPages = docGroups.flatMap((group) =>
  group.items.map((item) => ({ ...item, group: group.title })),
);
