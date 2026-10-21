# OrbitKV website

Static Astro pages and documentation. The site keeps OrbitKV's paper, green and
rust palette, typography and SVG identity. Content is organized around engine
setup, capabilities, deployment, architecture and reproducible measurements.

## Develop

Use the Node and npm versions in `package.json`. From this directory:

```sh
npm ci
npm run dev
```

Open the printed URL under `/OrbitKV/`. Before publishing:

```sh
npm run check
npm run build
npm test
npm run preview
```

Inspect desktop and mobile layouts, keyboard navigation, search, rendered
diagrams and internal links. Browser tooling stays outside production
dependencies. Search loads a static index on first use; Mermaid loads only on
pages containing diagrams. The document text and navigation work without JavaScript.

## Content and layout

```text
../docs/*.md                 canonical technical documentation
src/content.config.ts       Astro collection loading repository docs
src/data/docs.ts             documentation groups and page titles
src/data/site.ts             navigation, URLs and crate descriptions
src/markdown.mjs             repository links and Mermaid code fences
src/layouts/                shared site and documentation layouts
src/components/             cache illustration and document search
src/pages/                  overview, architecture, integration and docs
src/styles/                 existing visual tokens and document styles
tests/                      publication and link checks
public/                     identity assets and shared architecture SVG
```

Edit technical content in `../docs/`, without copying it into the website.
Register a new document in `src/data/docs.ts`; the build rejects missing or
unlisted pages. Markdown links between documents become site routes, shared
public assets stay local, and other repository links bind to the build's
source revision. Page headings supply the table of contents and search reads
the same collection. README and the site share `public/architecture.svg`.

Keep capability claims tied to evidence. Distinguish validated single-node
behavior, experimental shared-cache/P-D paths, and planned catalog HA and
routing. Update affected docs, README and site copy with behavioral changes.

## Publish

CI validates the site on pull requests. The Pages workflow checks, builds and
deploys `website/` or `docs/` changes merged into `main` at
[orbitinfra.github.io/OrbitKV](https://orbitinfra.github.io/OrbitKV/).

Repository links bind to the build's `GITHUB_SHA`. Local previews use `main`
unless `PUBLIC_SOURCE_REF` is set. Navigation and search respect Astro's
`/OrbitKV/` base path. No external font or search service is required.
