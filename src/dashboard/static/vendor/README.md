# Vendored dashboard assets

Committed deliberately: CLAUDE.md rule 1 requires zero CDN/external calls, so the
dashboard serves these locally instead of loading them from jsdelivr at runtime.
There is no build step — `src/api/app.py` mounts this directory verbatim.

To update: re-fetch from the URL below, re-verify the sha256, refresh this table.

| file | version | bytes | sha256 | source |
|---|---|---|---|---|
| `chart.umd.js` | 4.4.7 | 205615 | `2812cb8825fdc57469eb2f7bb055e9429244e599920511ee477e828499b632cb` | https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.js |
| `vis-network.min.js` | 9.1.9 | 476620 | `f6ee3a6560dd2bc7fce0b08149a30f2111e78255c5bb677e02c6517acc770379` | https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js |
| `vis-network.min.css` | 9.1.9 | 220163 | `2e82d445ad5878ea881652470ce632601f8f55f1b99e6ebecdff8614600e6d0e` | https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/dist/vis-network.min.css |
| `fonts/inter-latin-wght-normal.woff2` | 5.2.5 | 48444 | `f052ee44c3728dfd23aba8a4567150bc314d23903026fbb6ad089422c2df56af` | https://cdn.jsdelivr.net/npm/@fontsource-variable/inter@5.2.5/files/inter-latin-wght-normal.woff2 |
| `fonts/jetbrains-mono-latin-wght-normal.woff2` | 5.2.5 | 40404 | `18be452724bfdc236c074ca94a249a7f41a86752c7d04ab258ce9ed5651f6a7e` | https://cdn.jsdelivr.net/npm/@fontsource-variable/jetbrains-mono@5.2.5/files/jetbrains-mono-latin-wght-normal.woff2 |

All five hashes were verified against jsDelivr's published hashes
(`data.jsdelivr.com/v1/packages/npm/<pkg>@<ver>?structure=flat`) at fetch time.

## Licenses

`LICENSES/` holds the upstream text for each: chart.js (MIT), vis-network
(Apache-2.0 + MIT), Inter and JetBrains Mono (both OFL-1.1). The font licenses come
from the font projects themselves, not from the fontsource npm packages — those ship
a generic OFL file crediting "Google Inc.", which is the wrong copyright holder for
both faces, and OFL-1.1 requires the real notice.

## Notes

- The fonts are **variable** (wght 100–900) latin-subset files: one file each covers
  every weight the CSS uses (400/500/600/700). Non-latin text falls back to system fonts.
- No italic face is vendored; `font-style: italic` renders as synthesized oblique.
- `vis-network.min.css` genuinely lives at `dist/dist/` in the upstream package —
  that is not a typo, and it is a different file from `styles/vis-network.min.css`.
- chart.js is vendored as `chart.umd.js`, not `.min.js`: the minified path only exists
  because jsDelivr minifies on the fly, so it has no upstream hash to verify against.
- The vendored libraries contain `https://` strings in their own license headers
  (chartjs.org, apache.org, …) and `http://www.w3.org/2000/svg`, which is an XML
  namespace identifier, not a URL that is fetched. Their `sourceMappingURL` comments
  point at relative `.map` filenames that are not vendored, so devtools 404s locally
  rather than reaching out. **Do not strip these** — editing the files would void both
  the hash verification above and the license notices. The rule that matters is holding:
  zero external network calls.
