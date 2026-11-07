// The text2icd UI: code a phrase and see each chunk's path to its code (#/), or explore the graph
// (#/explore, #/search?q=, #/term/<id>, #/code/<code>). Plain ES modules, no build step. Served
// with `--review`, it is the hand review instead (#/review/<n>).

import { clear, h } from "./dom.js";
import { codeView, holdOnly, searchView, termView } from "./explore.js";
import { renderResult } from "./path.js";
import { reviewer } from "./review.js";
import { makeSource } from "./source.js";

// The demo lists its own examples, by group (resources/demo.toml); these are the service's.
const EXAMPLES = [
  {
    group: null,
    phrases: [
      "DM2 w/ CKD 3a, hx of MI, aspiration pneumonia 2/2 food",
      "type 2 DM with diabetic CKD stage 3a",
      "aspiration pneumonia due to food",
      "open fracture of shaft of right femur, subsequent encounter with delayed healing",
      "malignant neoplasm of right upper lobe of lung",
      "adverse effect of digoxin, initial encounter",
      "amyloid heart disease",
      "constipation and fecal impaction",
    ],
  },
];
const REPOSITORY = "https://github.com/a2psilva/text2icd";
const BOOKS = [
  ["", "every book"],
  ["index", "Alphabetic Index"],
  ["eindex", "External Cause Index"],
  ["neoplasm", "Table of Neoplasms"],
  ["drug", "Table of Drugs"],
];

const source = makeSource();
const main = document.getElementById("main");
const panel = document.getElementById("panel");
const state = { release: null, result: null, examples: EXAMPLES, review: null };
const REVIEWS = { audit: "Silver-label audit", gold: "Gold annotation" };

async function start() {
  try {
    const health = await source.health();
    state.release = health.release;
    document.getElementById("release").textContent = `${health.release}${source.kind === "static" ? " · demo" : ""}`;
    if (source.examples) state.examples = await source.examples();
    if (source.has) holdOnly(source.has);
    if (source.kind === "static") document.querySelector(".top nav").append(h("a", { href: REPOSITORY }, "Source"));
    if (health.review) reviewMode(health.review);
  } catch (error) {
    clear(main, h("p", { class: "error" }, error.message));
    return;
  }
  window.addEventListener("hashchange", route);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) closePanel();
  });
  route();
}

function route() {
  const [path, query] = location.hash.replace(/^#/, "").split("?");
  const params = new URLSearchParams(query ?? "");
  const term = path.match(/^\/term\/(\d+)$/);
  const code = path.match(/^\/code\/(.+)$/);
  const section = state.review && !term && !code ? "review" : /^\/(explore|search|term|code)/.test(path) ? "explore" : "code";
  for (const link of document.querySelectorAll("[data-nav]")) link.classList.toggle("on", link.dataset.nav === section);
  if (term) return explorePage({ term: Number(term[1]) });
  if (code) return explorePage({ code: decodeURIComponent(code[1]) });
  if (state.review) {
    // Only items, terms and codes: no phrase and no search goes in the address.
    const item = path.match(/^\/review\/(\d+)$/);
    return state.review.show(main, item && Number(item[1]) > 0 ? Number(item[1]) - 1 : null);
  }
  if (path.startsWith("/search")) return explorePage({ search: params.get("q") ?? "", book: params.get("book") ?? "" });
  if (path.startsWith("/explore")) return explorePage({});
  return codePage(params);
}

function reviewMode(mode) {
  state.review = reviewer(source, state.release, mode);
  document.title = `text2icd · ${REVIEWS[mode]}`;
  clear(document.querySelector(".top nav"), h("a", { href: "#/review", "data-nav": "review" }, REVIEWS[mode]));
  document.querySelector(".brand").setAttribute("href", "#/review");
  document.getElementById("release").textContent = `${state.release} · review`;
}

// -- Coding a phrase ------------------------------------------------------------------------------

function codePage(params) {
  closePanel(); // it shows a page of the previous result
  const phrase = params.get("q") ?? "";
  const setting = params.get("setting") === "outpatient" ? "outpatient" : "inpatient";
  const episode = params.get("episode") ?? "";
  const box = h("textarea", {
    id: "phrase",
    rows: "2",
    maxlength: "1000",
    placeholder: "A diagnosis phrase or a list, such as: DM2 w/ CKD 3a, hx of MI, HTN",
    "aria-label": "diagnosis phrase",
    onkeydown: (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) form.requestSubmit();
    },
  });
  box.value = phrase;
  const settingSelect = select("setting", [["inpatient", "inpatient"], ["outpatient", "outpatient"]], setting,
    "Inpatient codes uncertain diagnoses as if established (Section II.H); outpatient does not (IV.H).");
  const episodeSelect = select("episode", [["", "7th character: as stated"], ["initial", "initial encounter"], ["subsequent", "subsequent encounter"], ["sequela", "sequela"]], episode,
    "The episode of care assumed when an injury phrase states none.");
  const form = h(
    "form",
    {
      class: "ask",
      onsubmit: (event) => {
        event.preventDefault();
        const next = new URLSearchParams({ q: box.value.trim() });
        if (settingSelect.value !== "inpatient") next.set("setting", settingSelect.value);
        if (episodeSelect.value) next.set("episode", episodeSelect.value);
        if (!box.value.trim()) return;
        const hash = `#/?${next}`;
        if (location.hash === hash) codePage(next);
        else location.hash = hash;
      },
    },
    box,
    h("div", { class: "options" }, settingSelect, episodeSelect, h("button", { type: "submit", class: "primary-button" }, "Code")),
    h(
      "p",
      { class: "examples" },
      "Try: ",
      chips(state.examples).map((example) => h("button", { type: "button", class: "chip", onclick: () => pick(example) }, example)),
    ),
  );
  const pick = (example) => {
    box.value = example;
    form.requestSubmit();
  };
  const out = h("div", { id: "result", "aria-live": "polite" });
  clear(main, form, out);
  if (!phrase) {
    clear(out, intro(pick));
    box.focus();
    return;
  }
  const request = { phrase, setting, episode: episode || null, explain: true };
  clear(out, h("p", { class: "muted" }, "Coding…"));
  source
    .code(request)
    .then((result) => {
      state.result = result;
      clear(out, toolbar(result), renderResult(result, openPanel));
    })
    .catch((error) => clear(out, h("p", { class: "error" }, error.message)));
}

function select(name, options, value, title) {
  const el = h("select", { name, title, "aria-label": name }, options.map(([v, label]) => h("option", { value: v }, label)));
  el.value = value;
  return el;
}

// One chip per group, or the phrases of the only group.
function chips(groups) {
  return groups.length === 1 ? groups[0].phrases : groups.map((g) => g.phrases[0]);
}

function intro(pick) {
  const count = state.examples.reduce((n, g) => n + g.phrases.length, 0);
  return h(
    "div",
    { class: "intro" },
    h(
      "div",
      { class: "muted" },
      h("p", {}, "text2icd codes the way a coder does: it splits the phrase into its conditions, looks each one up in the Alphabetic Index, follows the subterms and cross-references, and checks the code in the Tabular List."),
      h("p", {}, "The result shows each chunk of the phrase and its path to the code. Click a step to see that Index term or code, or open the explorer."),
      source.kind === "static"
        ? h("p", {}, `This demo holds the answers to ${count} example phrases, computed in advance by the deterministic coder (no model, no API). To code any phrase, `, h("a", { href: REPOSITORY }, "install text2icd"), " and run text2icd serve.")
        : null,
    ),
    state.examples.length > 1
      ? h(
          "div",
          { class: "example-groups" },
          state.examples.map((g) =>
            h("section", {}, h("h4", {}, g.group), h("ul", {}, g.phrases.map((p) => h("li", {}, h("button", { type: "button", class: "link", onclick: () => pick(p) }, p))))),
          ),
        )
      : null,
  );
}

function toolbar(result) {
  return h(
    "div",
    { class: "toolbar" },
    h("button", { type: "button", onclick: () => download(result) }, "Download JSON"),
    h("span", { class: "muted" }, `${result.release} · graph ${result.graph.slice(0, 12)} · text2icd ${result.version}`),
  );
}

function download(result) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const link = h("a", { href: URL.createObjectURL(blob), download: `text2icd-${result.codes[0]?.code ?? "result"}.json` });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

// -- The side panel -------------------------------------------------------------------------------

async function openPanel(target) {
  panel.hidden = false;
  document.body.classList.add("with-panel");
  const close = h("button", { type: "button", class: "close", "aria-label": "close", onclick: closePanel }, "×");
  const page = target.term !== undefined ? `#/term/${target.term}` : `#/code/${encodeURIComponent(target.code)}`;
  clear(panel, h("div", { class: "panel-head" }, h("a", { href: page }, "open in the explorer"), close), h("p", { class: "muted" }, "Loading…"));
  try {
    const body = await nodeView(target, openPanel);
    clear(panel, h("div", { class: "panel-head" }, h("a", { href: page, onclick: closePanel }, "open in the explorer"), close), body, related(target));
  } catch (error) {
    clear(panel, h("div", { class: "panel-head" }, close), h("p", { class: "error" }, error.message));
  }
}

function closePanel() {
  panel.hidden = true;
  document.body.classList.remove("with-panel");
}

// The trace steps of the shown result that are about this term or code.
function related(target) {
  const steps = (state.result?.trace ?? []).filter((t) => (target.term !== undefined ? t.term === target.term : t.code === target.code));
  if (!steps.length) return null;
  return h(
    "section",
    { class: "related" },
    h("h4", {}, "In this result"),
    h("ul", {}, steps.map((t) => h("li", {}, h("span", { class: "tag" }, t.action.replaceAll("_", " ")), " ", t.detail, t.guideline ? h("span", { class: "muted" }, ` (Guideline ${t.guideline})`) : null))),
  );
}

async function nodeView(target, go) {
  if (target.term !== undefined) {
    const chosen = new Set((state.result?.detail?.paths ?? []).flatMap((p) => p.steps.map((s) => s.term)));
    return termView(await source.termPage(state.release, target.term), go, chosen);
  }
  return codeView(await source.codePage(state.release, target.code), go);
}

// -- The explorer ---------------------------------------------------------------------------------

async function explorePage(target) {
  closePanel();
  const q = h("input", { type: "search", name: "q", value: target.search ?? "", placeholder: "Search the Index and the Tabular List", "aria-label": "search", maxlength: "200" });
  const book = select("book", BOOKS, target.book ?? "", "Search one book of the Index, or every book and the Tabular List.");
  const form = h(
    "form",
    {
      class: "explore-search",
      onsubmit: (event) => {
        event.preventDefault();
        if (!q.value.trim()) return;
        const next = new URLSearchParams({ q: q.value.trim() });
        if (book.value) next.set("book", book.value);
        location.hash = `#/search?${next}`;
      },
    },
    q,
    book,
    h("button", { type: "submit" }, "Search"),
  );
  const go = (next) => {
    location.hash = next.term !== undefined ? `#/term/${next.term}` : `#/code/${encodeURIComponent(next.code)}`;
  };
  const out = h("div", { class: "explore-body" });
  // In the review mode the search is the review page's lookup pane, which keeps it out of the
  // address; the demo cannot search.
  const top = state.review ? h("p", {}, h("a", { href: "#/review" }, "‹ back to the review")) : source.kind === "static" ? null : form;
  clear(main, top, out);
  try {
    if (target.term !== undefined || target.code !== undefined) {
      clear(out, await nodeView(target, go));
    } else if (target.search) {
      clear(out, searchView(await source.search(state.release, target.search, target.book || null), go));
    } else if (source.kind === "static") {
      clear(out, h("p", { class: "muted" }, "The demo holds the Index terms and codes its examples touch: open one from an example's result, or start at ", h("a", { href: "#/code/E11.22" }, "E11.22"), ". Searching the whole Index and Tabular List needs text2icd serve."));
    } else {
      clear(out, h("p", { class: "muted" }, "Search for a condition (", h("a", { href: "#/search?q=pneumonia" }, "pneumonia"), "), or open a code (", h("a", { href: "#/code/E11.22" }, "E11.22"), ")."));
      q.focus();
    }
  } catch (error) {
    clear(out, h("p", { class: "error" }, error.message));
  }
}

start();
