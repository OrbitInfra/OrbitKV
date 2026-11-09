import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import { repositoryMarkdown } from "./src/markdown.mjs";
const sourceRevision =
  process.env.PUBLIC_SOURCE_REF || process.env.GITHUB_SHA || "main";

const base = "/OrbitKV";

export default defineConfig({
  site: "https://orbitinfra.github.io",
  base,
  output: "static",
  trailingSlash: "always",
  markdown: {
    processor: unified({
      remarkPlugins: [[repositoryMarkdown, { base, sourceRevision }]],
    }),
    shikiConfig: { theme: "github-light", langAlias: { promql: "text" } },
  },
  redirects: {
    "/evidence/": `${base}/docs/performance/`,
    ...Object.fromEntries(
      ["client-performance", "communication-performance", "single-node-performance",
        "concurrent-performance", "sustained-performance", "recovery-performance"]
        .map((slug) => [`/docs/${slug}/`, `${base}/docs/performance/`]),
    ),
    "/docs/pd-mooncake-push/": `${base}/docs/pd/`,
    "/docs/rust-quality/": `${base}/docs/contributing/`,
    "/docs/distributed-comparison/": `${base}/docs/completion-plan/`,
    "/docs/distributed-design/": `${base}/docs/distributed-cache/`,
    "/docs/engine-release-audit/": `${base}/docs/adapters/`,
    "/docs/kernel-optimization/": `${base}/docs/transport/`,
    "/docs/ssd-performance/": `${base}/docs/gds/`,
    "/docs/state-planning/": `${base}/docs/architecture/`,
    "/docs/queued-warming/": `${base}/docs/request-preparation/`,
    "/docs/shared-cache-qualification/": `${base}/docs/fault-qualification/`,
    "/docs/benchmark-evidence/": `${base}/docs/performance/`,
    "/docs/peer-control/": `${base}/docs/distributed-cache/`,
  },
  vite: {
    define: {
      "import.meta.env.PUBLIC_SOURCE_REF": JSON.stringify(sourceRevision),
    },
  },
});
