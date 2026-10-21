import { relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const docs = resolve(root, "docs");
const assets = resolve(root, "website/public");
const inside = (parent, child) => child.startsWith(parent + sep);
const escape = (text) =>
  text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

export function repositoryMarkdown({ base, sourceRevision }) {
  return (tree, file) => {
    const visit = (node) => {
      if (node.type === "code" && node.lang === "mermaid") {
        node.type = "html";
        node.value = `<pre class="mermaid" tabindex="0">${escape(node.value)}</pre>`;
      }
      if (
        node.url &&
        file.path &&
        !/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(node.url)
      ) {
        const target = new URL(node.url, pathToFileURL(file.path));
        const path = fileURLToPath(target);
        if (inside(docs, path) && path.endsWith(".md")) {
          node.url = `${base}/docs/${relative(docs, path).replace(/\.md$/, "")}/${target.search}${target.hash}`;
        } else if (inside(assets, path)) {
          node.url = `${base}/${relative(assets, path)}${target.search}${target.hash}`;
        } else if (inside(root, path)) {
          node.url = `https://github.com/OrbitInfra/orbitkv/blob/${sourceRevision}/${relative(root, path)}${target.hash}`;
        }
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}
