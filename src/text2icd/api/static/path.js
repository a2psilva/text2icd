// The path view (task 10.10): a CodingResult drawn as the chunks of the phrase, then one lane per
// diagnosis from its chunks through the Alphabetic Index and the Tabular List to its code.

import { h, percent, plural } from "./dom.js";

const ROLES = {
  diagnosis: "diagnosis",
  manifestation: "manifestation",
  code_first: "code first",
  use_additional: "use additional code",
  code_also: "code also",
  external_cause: "external cause",
};
const NOTE_KINDS = {
  codeFirst: "code first",
  useAdditionalCode: "use additional code",
  codeAlso: "code also",
};
const BOOKS = {
  index: "Alphabetic Index",
  eindex: "External Cause Index",
  neoplasm: "Table of Neoplasms",
  drug: "Table of Drugs and Chemicals",
  retrieval: "local retrieval (L2)",
};
const LAYERS = { index: "L1", retrieval: "L2", llm: "L3" };
const LAYER_NAMES = { index: "L1, the Index walk", retrieval: "L2, local retrieval", llm: "L3, the LLM" };

// `open` shows a term or a code in the side panel: open({term: id}) or open({code: "E11.22"}).
export function renderResult(result, open) {
  const detail = result.detail;
  return h(
    "div",
    { class: "result" },
    summary(result),
    annotated(result),
    codesTable(result, open),
    detail ? lanes(result, open) : null,
    detail && detail.pool.length ? pool(detail.pool, open) : null,
    detail && detail.decisions.length ? decisions(detail.decisions) : null,
    trace(result, open),
  );
}

function summary(result) {
  const primary = result.codes[0];
  return h(
    "section",
    { class: "summary" },
    h(
      "div",
      { class: "primary" },
      primary
        ? [h("span", { class: "code big" }, primary.code), h("span", {}, primary.description)]
        : h("span", { class: "muted" }, noCode(result)),
    ),
    h(
      "dl",
      { class: "facts" },
      fact("confidence", meter(result.confidence), "Calibrated: the chance that the primary code is right."),
      fact("primary code from", LAYER_NAMES[result.resolver] ?? result.resolver),
      result.escalated.length
        ? fact(
            "escalated",
            result.escalated.map((l) => LAYERS[l] ?? l).join(", "),
            "These layers' confidence was below their threshold, so the next layer was asked.",
          )
        : null,
      fact("setting", result.setting),
      result.flags.length ? fact("flags", result.flags.join(", ")) : null,
      result.cached ? fact("cached", "yes", "Answered from the in-memory cache.") : null,
      result.llm_calls ? fact("L3", `${plural(result.llm_calls, "call")}, $${result.cost_usd.toFixed(4)}`) : null,
    ),
  );
}

function fact(name, value, title) {
  return h("div", { title }, h("dt", {}, name), h("dd", {}, value));
}

function meter(value) {
  if (value === null || value === undefined) return "–";
  const level = value >= 0.9 ? "high" : value >= 0.6 ? "mid" : "low";
  const bar = h("span", { class: "bar" });
  bar.style.width = `${Math.round(value * 100)}%`; // a property, which the page's CSP allows
  return h("span", { class: `meter ${level}` }, h("span", { class: "track" }, bar), h("span", { class: "label" }, percent(value)));
}

// -- The phrase, with its chunks ------------------------------------------------------------------

function annotated(result) {
  const input = result.input;
  const chunkAt = new Array(input.length).fill(-1);
  const marksAt = Array.from({ length: input.length }, () => []);
  const marks = [];
  for (const chunk of result.chunks) {
    for (let i = chunk.start; i < chunk.end; i++) chunkAt[i] = chunk.id;
    for (const m of chunk.modifiers) marks.push({ ...m, label: `${m.kind}: ${m.value}`, type: "modifier" });
    for (const c of chunk.cues) marks.push({ ...c, label: `${c.kind} (written ${c.value} the condition)`, type: "cue" });
  }
  for (const e of result.expansions) {
    if (e.start === null || e.start === undefined) continue;
    const others = e.alternatives.length ? `; also ${e.alternatives.join(", ")}` : "";
    marks.push({ ...e, label: `${e.written} read as ${e.expansion}${others}`, type: "expansion" });
  }
  marks.forEach((mark, index) => {
    for (let i = mark.start; i < mark.end && i < input.length; i++) marksAt[i].push(index);
  });

  const line = h("p", { class: "phrase", "aria-label": "the phrase, with its chunks" });
  const byId = new Map(result.chunks.map((c) => [c.id, c]));
  let i = 0;
  while (i < input.length) {
    const id = chunkAt[i];
    let j = i;
    while (j < input.length && chunkAt[j] === id) j++;
    const pieces = runs(input, marksAt, marks, i, j);
    if (id < 0) {
      line.append(...pieces);
    } else {
      const chunk = byId.get(id);
      const status = [chunk.status !== "current" ? chunk.status.replace("_", " ") : null,
        chunk.certainty !== "affirmed" ? chunk.certainty : null].filter(Boolean);
      line.append(
        h(
          "span",
          {
            class: `chunk d${chunk.diagnosis % 6}${chunk.reportable ? "" : " unreported"}`,
            dataset: { chunk: String(id) },
            title: `chunk ${id + 1}: ${chunk.condition}${status.length ? ` (${status.join(", ")})` : ""}${
              chunk.reportable ? "" : ": not coded"}`,
            tabindex: "0",
            onmouseenter: () => light(String(id), true),
            onmouseleave: () => light(String(id), false),
            onfocus: () => light(String(id), true),
            onblur: () => light(String(id), false),
          },
          pieces,
          status.length ? h("span", { class: "tag" }, status.join(", ")) : null,
        ),
      );
    }
    i = j;
  }
  const legend = h(
    "p",
    { class: "legend muted" },
    "Each colour is a diagnosis; ",
    h("span", { class: "mark expansion" }, "dotted"),
    ": an abbreviation read; ",
    h("span", { class: "mark modifier" }, "underlined"),
    ": a modifier or cue. Hover a chunk to find its lane.",
  );
  return h(
    "section",
    { class: "chunks" },
    h("h2", {}, "Chunks"),
    line,
    result.normalized !== result.input ? h("p", { class: "normalized muted" }, "read as: ", result.normalized) : null,
    legend,
  );
}

function runs(input, marksAt, marks, start, end) {
  const out = [];
  let i = start;
  while (i < end) {
    const keyOf = (k) => marksAt[k].join(",");
    const key = keyOf(i);
    let j = i;
    while (j < end && keyOf(j) === key) j++;
    const text = input.slice(i, j);
    if (!key) {
      out.push(text);
    } else {
      const here = marksAt[i].map((m) => marks[m]);
      const types = [...new Set(here.map((m) => m.type))];
      out.push(h("span", { class: `mark ${types.join(" ")}`, title: here.map((m) => m.label).join("\n") }, text));
    }
    i = j;
  }
  return out;
}

function light(id, on) {
  for (const el of document.querySelectorAll(`[data-chunks~="${id}"], [data-chunk="${id}"]`)) {
    el.classList.toggle("lit", on);
  }
}

// -- The codes ------------------------------------------------------------------------------------

function codesTable(result, open) {
  if (!result.codes.length) return null;
  const chunkText = new Map(result.chunks.map((c) => [c.id, c.text]));
  return h(
    "section",
    { class: "codes" },
    h("h2", {}, "Codes, in coding order"),
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, ["#", "code", "description", "role", "chunks", "brought in by"].map((t) => h("th", {}, t)))),
      h(
        "tbody",
        {},
        result.codes.map((code, n) =>
          h(
            "tr",
            {
              dataset: { chunks: code.chunks.join(" ") },
              class: code.primary ? "primary-row" : "",
            },
            h("td", {}, n + 1),
            h("td", {}, codeButton(code.code, open)),
            h("td", {}, code.description),
            h("td", {}, h("span", { class: `role ${code.role}` }, ROLES[code.role] ?? code.role)),
            h("td", { class: "muted" }, code.chunks.map((c) => chunkText.get(c)).join(" · ")),
            h("td", {}, code.source ? codeButton(code.source, open) : "–"),
          ),
        ),
      ),
    ),
  );
}

function codeButton(code, open, title) {
  return h("button", { class: "code link", type: "button", title, onclick: () => open({ code }) }, code);
}

function joined(items, separator = ", ") {
  return items.flatMap((item, n) => (n ? [separator, item] : [item]));
}

// -- One lane per diagnosis -----------------------------------------------------------------------

function lanes(result, open) {
  const detail = result.detail;
  const chunks = new Map(result.chunks.map((c) => [c.id, c]));
  return h(
    "section",
    { class: "lanes" },
    h("h2", {}, "How each diagnosis was coded"),
    detail.paths.map((path, n) => lane(result, path, n, chunks, open)),
    detail.excluded.length ? excluded(detail.excluded, open) : null,
  );
}

function lane(result, path, n, chunks, open) {
  const diagnosis = path.chunks.length ? chunks.get(path.chunks[0])?.diagnosis ?? n : n;
  const steps = [];
  const start = h(
    "div",
    { class: "node chunk-node" },
    h("span", { class: "kind" }, path.chunks.length === 1 ? "chunk" : "chunks"),
    path.chunks.map((c) => h("span", { class: "chunk-text" }, chunks.get(c)?.text ?? "")),
  );
  if (path.book === "retrieval" || !path.steps.length) {
    const hit = (result.detail.pool || []).find((p) => p.code === path.code);
    steps.push(edge(""));
    steps.push(
      h(
        "li",
        { class: "node retrieval" },
        h("span", { class: "kind" }, BOOKS[path.book] ?? path.book),
        hit ? `closest Tabular text, similarity ${hit.similarity.toFixed(2)}` : "no Index path",
      ),
    );
  }
  path.steps.forEach((step, k) => {
    const via = step.via ? step.via.replace("_", " ") : "";
    steps.push(edge(via, step.note));
    const coded = k === path.coded_step;
    steps.push(
      h(
        "li",
        { class: `node term${k === 0 ? " main" : ""}${coded ? " coded" : ""}` },
        h(
          "button",
          {
            type: "button",
            class: "node-button",
            onclick: () => open({ term: step.term }),
            title: step.took.length ? `matches the phrase words: ${step.took.join(", ")}` : "Show this term",
          },
          k === 0 ? h("span", { class: "kind" }, `${BOOKS[path.book] ?? path.book}: main term`) : null,
          h("span", { class: "title" }, step.title),
          step.column !== null && step.column !== undefined ? h("span", { class: "muted" }, ` column ${step.column}`) : null,
          coded ? h("span", { class: "written" }, path.written) : null,
        ),
      ),
    );
  });
  for (const step of path.trace.filter((t) => t.layer === "tabular")) {
    steps.push(edge(step.action === "complete" ? "Tabular List" : ""));
    steps.push(
      h(
        "li",
        { class: "node tabular" },
        h(
          "button",
          {
            type: "button",
            class: "node-button",
            onclick: () => step.code && open({ code: step.code }),
            title: step.guideline ? `Guideline ${step.guideline}` : "Show this code",
          },
          h("span", { class: "kind" }, step.action.replaceAll("_", " ")),
          h("span", { class: "title", title: step.detail }, tabularTitle(step)),
        ),
      ),
    );
  }
  const end = h(
    "div",
    { class: "node final" },
    h(
      "button",
      { type: "button", class: "node-button", onclick: () => open({ code: path.code }) },
      h("span", { class: "kind" }, ROLES[path.role] ?? path.role),
      h("span", { class: "code" }, path.code, path.manifestation ? ` [${path.manifestation}]` : ""),
      h("span", { class: "title" }, path.description),
    ),
  );

  const companions = result.detail.companions.filter((c) => c.source === path.code || c.source === path.manifestation);
  return h(
    "article",
    { class: `lane d${diagnosis % 6}`, dataset: { chunks: path.chunks.join(" ") } },
    h(
      "header",
      {},
      h("span", { class: "code" }, path.code),
      h("span", {}, path.description),
      path.flags.map((f) => h("span", { class: "tag" }, f.replaceAll("_", " "))),
    ),
    h("div", { class: "lane-row" }, start, h("ol", { class: "steps" }, steps), h("div", { class: "edge to-code" }, h("span", { class: "label" }, " "), h("span", { class: "arrow" })), end),
    companions.length
      ? h(
          "ul",
          { class: "branches" },
          companions.map((c) =>
            h(
              "li",
              { class: `companion ${c.status}` },
              h("span", { class: "kind" }, NOTE_KINDS[c.kind] ?? c.kind),
              " ",
              h("span", { class: "muted" }, `“${c.note}”`),
              " → ",
              c.code ? codeButton(c.code, open, c.description) : h("span", { class: "muted" }, "no code"),
              " ",
              h("span", { class: `status ${c.status}` }, c.status.replaceAll("_", " ")),
            ),
          ),
        )
      : null,
    path.alternates.length
      ? h(
          "p",
          { class: "alternates muted" },
          "also reached: ",
          joined(path.alternates.map((a) => codeButton(a.code, open, a.description))),
        )
      : null,
  );
}

// A Tabular step, short: "S72.30 → S72.301H" rather than the code's whole description.
function tabularTitle(step) {
  const completed = step.detail.match(/^(\S+) -> (\S+)/);
  if (completed) return `${completed[1]} → ${completed[2]}`;
  return step.detail.length > 70 ? `${step.detail.slice(0, 68)}…` : step.detail;
}

function edge(label, note) {
  return h(
    "li",
    { class: `edge${label === "see" || label === "see also" ? " jump" : ""}`, "aria-hidden": label ? null : "true" },
    h("span", { class: "label", title: note ? `written: ${note}` : null }, label || " "),
    h("span", { class: "arrow" }),
  );
}

function excluded(rows, open) {
  return h(
    "div",
    { class: "excluded" },
    h("h3", {}, "Taken out by Excludes1"),
    h(
      "ul",
      {},
      rows.map((e) =>
        h("li", {}, h("s", {}, codeButton(e.code, open)), " because ", codeButton(e.kept, open), " excludes it: ",
          h("span", { class: "muted" }, `“${e.note}”`)),
      ),
    ),
  );
}

// -- L2 and L3 ------------------------------------------------------------------------------------

function pool(rows, open) {
  return h(
    "details",
    { class: "pool" },
    h("summary", {}, `What L2 weighed (${plural(rows.length, "code")})`),
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, ["code", "description", "from", "similarity", "probability"].map((t) => h("th", {}, t)))),
      h(
        "tbody",
        {},
        rows.map((row) =>
          h(
            "tr",
            {},
            h("td", {}, codeButton(row.code, open)),
            h("td", {}, row.description),
            h("td", { class: "muted" }, row.origin),
            h("td", {}, row.similarity === null || row.similarity === undefined ? "–" : row.similarity.toFixed(3)),
            h("td", {}, meter(row.probability)),
          ),
        ),
      ),
    ),
  );
}

function decisions(rows) {
  return h(
    "details",
    { class: "decisions", open: true },
    h("summary", {}, `What L3 was asked (${plural(rows.length, "decision")})`),
    h("ol", {}, rows.map((d) => h("li", {}, h("pre", {}, JSON.stringify(d, null, 2))))),
  );
}

// -- Every step -----------------------------------------------------------------------------------

function trace(result, open) {
  return h(
    "details",
    { class: "trace" },
    h("summary", {}, `Every step (${plural(result.trace.length, "step")})`),
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, ["layer", "step", "detail", "guideline"].map((t) => h("th", {}, t)))),
      h(
        "tbody",
        {},
        result.trace.map((step) =>
          h(
            "tr",
            {},
            h("td", { class: "muted" }, step.layer),
            h("td", {}, step.action.replaceAll("_", " ")),
            h(
              "td",
              {},
              step.term !== null && step.term !== undefined
                ? h("button", { type: "button", class: "link", onclick: () => open({ term: step.term }) }, step.detail)
                : step.code
                  ? h("button", { type: "button", class: "link", onclick: () => open({ code: step.code }) }, step.detail)
                  : step.detail,
            ),
            h("td", { class: "muted" }, step.guideline ?? ""),
          ),
        ),
      ),
    ),
  );
}

// Why a result has no code: the reasons its conditions are not reported (uncertain in the
// outpatient setting, negated...), or that the Index codes none of them.
function noCode(result) {
  const unreported = result.chunks.length > 0 && result.chunks.every((c) => !c.reportable);
  const why = unreported ? result.trace.filter((t) => t.layer === "text" && t.guideline) : [];
  if (!why.length) return "No code: the phrase states no condition the Index codes.";
  return `No code: ${why.map((t) => `${t.detail} (Guideline ${t.guideline})`).join("; ")}.`;
}
