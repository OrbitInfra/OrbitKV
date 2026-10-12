export const docGroups = [
  {
    title: "Getting started",
    description: "Install, select a supported profile, and verify an external hit.",
    items: [
      { slug: "goals", title: "Overview" },
      { slug: "single-node", title: "Installation & quickstart" },
      { slug: "compatibility", title: "Compatibility & support" },
    ],
  },
  {
    title: "Deployment",
    description: "Connect engines, share capacity, and select a topology.",
    items: [
      { slug: "adapters", title: "Engine adapter contracts" },
      { slug: "deployment", title: "Deployment recipes" },
      { slug: "models", title: "Model qualification" },
      { slug: "p2p", title: "Cross-host shared cache" },
      { slug: "pd", title: "Native P/D composition" },
    ],
  },
  {
    title: "Operations",
    description: "Configure capacity, inspect metrics, and diagnose failures.",
    items: [
      { slug: "server", title: "Manager configuration" },
      { slug: "metrics", title: "Metrics & observability" },
      { slug: "fault-qualification", title: "Fault qualification" },
    ],
  },
  {
    title: "Architecture",
    description: "Understand identity, ownership, and storage paths.",
    items: [
      { slug: "architecture", title: "System architecture" },
      { slug: "transport", title: "Process & network transport" },
      { slug: "state-identity", title: "State identity" },
      { slug: "hybrid-recovery", title: "Hybrid recovery contracts" },
      { slug: "cache-policies", title: "Retention & SSD admission" },
      { slug: "gds", title: "GPU storage & GDS limits" },
      { slug: "storage-formats", title: "Storage codecs" },
    ],
  },
  {
    title: "Benchmarks",
    description: "Read scoped results and reproduce matched measurements.",
    items: [
      { slug: "performance", title: "Performance overview" },
      { slug: "benchmark-evidence", title: "Methodology & evidence" },
      { slug: "kernel-optimization", title: "CUDA kernel work" },
      { slug: "ssd-performance", title: "SSD recovery methodology" },
    ],
  },
  {
    title: "Development",
    description: "Contribute against the single execution queue.",
    items: [
      { slug: "contributing", title: "Contributor guide" },
      { slug: "completion-plan", title: "Completion plan" },
      { slug: "engine-release-audit", title: "Released interface audit" },
      { slug: "releases", title: "Package releases" },
    ],
  },
];

// Keep stable reference routes and search entries outside the primary sidebar.
const referencePages = [
  { slug: "engine-local-restore", title: "Engine-local restore ownership", group: "Reference" },
  { slug: "state-planning", title: "Demand & transfer planning", group: "Reference" },
  { slug: "queued-warming", title: "Queued-request warming", group: "Reference" },
  { slug: "request-preparation", title: "Consumer-owned preparation", group: "Reference" },
  { slug: "vllm-request-state-machine", title: "vLLM request lifecycle", group: "Reference" },
  { slug: "peer-control", title: "Peer control boundary", group: "Reference" },
  { slug: "shared-cache-qualification", title: "Shared-cache qualification", group: "Reference" },
  { slug: "distributed-cache", title: "Distributed global index", group: "Reference" },
  { slug: "distributed-design", title: "Inventory and recovery design", group: "Reference" },
  { slug: "distributed-comparison", title: "LMCache, FlexKV & Mooncake comparison", group: "Reference" },
  { slug: "pd-mooncake-push", title: "Retired P/D protocol", group: "Reference" },
  { slug: "client-performance", title: "Client control overhead", group: "Reference" },
  { slug: "communication-performance", title: "Communication measurements", group: "Reference" },
  { slug: "single-node-performance", title: "Single-node comparisons", group: "Reference" },
  { slug: "concurrent-performance", title: "Concurrent query budgets", group: "Reference" },
  { slug: "sustained-performance", title: "Sustained serving", group: "Reference" },
  { slug: "recovery-performance", title: "Recovery stage profile", group: "Reference" },
  { slug: "rust-quality", title: "Rust quality gates", group: "Reference" },
];

export const docPages = [
  ...docGroups.flatMap((group) =>
    group.items.map((item) => ({ ...item, group: group.title })),
  ),
  ...referencePages,
];
