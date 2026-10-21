import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, globSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const dist = fileURLToPath(new URL("../dist/", import.meta.url));
const repository = resolve(dist, "../..");
const base = "https://orbitinfra.github.io/orbitkv/";
const decode = (value) =>
  value.replaceAll("&amp;", "&").replaceAll("&quot;", '"');
const files = globSync("**/*.html", { cwd: dist });

test("published pages have valid internal links, fragments and repository targets", () => {
  assert.ok(
    files.length > 20,
    "Build the site before running publication checks",
  );
  const failures = [];
  const historicalTargets = new Set();
  for (const file of files) {
    const html = readFileSync(join(dist, file), "utf8");
    const current = new URL(file.replace(/index\.html$/, ""), base);
    for (const [, attribute] of html.matchAll(/\b(?:href|src)="([^"]*)"/g)) {
      const target = new URL(decode(attribute), current);
      if (target.origin === "https://github.com") {
        const match = target.pathname.match(
          /^\/OrbitInfra\/orbitkv\/blob\/([^/]+)\/(.+)$/,
        );
        const [, revision, path] = match ?? [];
        // Historical snapshots can contain evidence removed from the current tree.
        const historical = revision && /^[a-f0-9]{40}$/.test(revision);
        if (path && historical) {
          const object = `${revision}:${decodeURIComponent(path)}`;
          if (!historicalTargets.has(object)) {
            historicalTargets.add(object);
            const result = spawnSync("git", ["cat-file", "-e", object], {
              cwd: repository,
              encoding: "utf8",
            });
            if (result.status !== 0)
              failures.push(`${file}: missing historical target ${object}; fetch Git history before checking`);
          }
        } else if (path && !existsSync(join(repository, decodeURIComponent(path)))) {
          failures.push(`${file}: missing repository file ${path}`);
        }
        continue;
      }
      if (target.origin !== current.origin) continue;
      if (!target.pathname.startsWith("/orbitkv/")) {
        failures.push(`${file}: URL outside site base ${target}`);
        continue;
      }
      let path = join(
        dist,
        decodeURIComponent(target.pathname.slice("/orbitkv/".length)),
      );
      if (existsSync(path) && statSync(path).isDirectory())
        path = join(path, "index.html");
      if (!existsSync(path)) {
        failures.push(`${file}: missing published target ${target}`);
      } else if (target.hash && path.endsWith(".html")) {
        const ids = new Set(
          [...readFileSync(path, "utf8").matchAll(/\bid="([^"]*)"/g)].map(
            (match) => decode(match[1]),
          ),
        );
        if (!ids.has(decodeURIComponent(target.hash.slice(1)))) {
          failures.push(`${file}: missing fragment ${target}`);
        }
      }
    }
  }
  assert.deepEqual(failures, []);
});

test("search publishes every canonical document with an internal URL and its content", () => {
  const pages = JSON.parse(
    readFileSync(join(dist, "docs/search.json"), "utf8"),
  );
  const sources = globSync("**/*.md", { cwd: join(repository, "docs") });
  assert.equal(pages.length, sources.length);
  assert.equal(new Set(pages.map((page) => page.url)).size, sources.length);
  for (const source of sources) {
    const url = `/orbitkv/docs/${source.replace(/\.md$/, "")}/`;
    const page = pages.find((page) => page.url === url);
    assert.ok(page, `Missing searchable document: ${source}`);
    assert.ok(
      page.title && page.body.includes("# "),
      `Empty search content: ${source}`,
    );
    assert.ok(
      existsSync(join(dist, url.slice("/orbitkv/".length), "index.html")),
    );
  }
});
