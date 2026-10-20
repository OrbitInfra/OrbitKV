export const docGroups = [
  {
    title: "Getting started",
    description: "Install, select a supported profile, and verify an external hit.",
    items: [
      { slug: "goals", title: "Overview" },
      { slug: "features", title: "Features" },
      { slug: "single-node", title: "Installation & quickstart" },
      { slug: "compatibility", title: "Compatibility & support" },
    ],
  },
  {
    title: "Models & deployment",
    description: "Choose a model, connect an engine, and share cache capacity.",
    items: [
      { slug: "models", title: "Model recipes" },
      { slug: "deployment", title: "Deployment recipes" },
      { slug: "p2p", title: "Cross-host shared cache" },
      { slug: "pd", title: "Native P/D composition" },
      { slug: "storage-formats", title: "Storage formats & compression" },
      { slug: "cache-policies", title: "Retention & SSD admission" },
    ],
  },
  {
    title: "Operations",
    description: "Configure capacity, inspect metrics, and diagnose failures.",
    items: [
      { slug: "server", title: "Manager configuration" },
      { slug: "metrics", title: "Metrics & observability" },
    ],
  },
  {
    title: "Architecture",
    description: "Understand identity, ownership, and storage paths.",
    items: [
      { slug: "architecture", title: "System architecture" },
      { slug: "transport", title: "Process & network transport" },
      { slug: "state-identity", title: "State identity" },
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
      { slug: "adapters", title: "Engine adapter contracts" },
      { slug: "engine-release-audit", title: "Released interface audit" },
      { slug: "releases", title: "Package releases" },
    ],
  },
];

// Keep stable reference routes and search entries outside the primary sidebar.
const referencePages = [
  { slug: "hybrid-recovery", title: "Hybrid recovery contracts", group: "Reference" },
  { slug: "gds", title: "GPU storage & GDS limits", group: "Reference" },
  { slug: "fault-qualification", title: "Fault qualification", group: "Reference" },
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
];

export const docPages = [
  ...docGroups.flatMap((group) =>
    group.items.map((item) => ({ ...item, group: group.title })),
  ),
  ...referencePages,
];
