import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import { repositoryMarkdown } from "./src/markdown.mjs";
const sourceRevision =
  process.env.PUBLIC_SOURCE_REF || process.env.GITHUB_SHA || "main";

const base = "/orbitkv";

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
    "/evidence/": `${base}/integration/`,
    ...Object.fromEntries(
      ["client-performance", "communication-performance", "single-node-performance",
        "concurrent-performance", "sustained-performance", "recovery-performance"]
        .map((slug) => [`/docs/${slug}/`, `${base}/docs/performance/`]),
    ),
    "/docs/pd-mooncake-push/": `${base}/docs/pd/`,
    "/docs/rust-quality/": `${base}/docs/contributing/`,
  },
  vite: {
    define: {
      "import.meta.env.PUBLIC_SOURCE_REF": JSON.stringify(sourceRevision),
    },
  },
});
