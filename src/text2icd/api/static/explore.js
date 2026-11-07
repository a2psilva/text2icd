// The graph explorer (task 10.11): an Index term or a Tabular code, as lists that link onward.
// The side panel of the path view and the explorer page draw the same views.

import { h } from "./dom.js";

const NOTES = {
  inclusionTerm: "Inclusion terms",
  includes: "Includes",
  excludes1: "Excludes1",
  excludes2: "Excludes2",
  codeFirst: "Code first",
  useAdditionalCode: "Use additional code",
  codeAlso: "Code also",
  notes: "Notes",
};
const VIA = { see: "see", see_also: "see also", see_condition: "see condition" };

// The static demo holds some pages only: a link to any other is shown as text.
let held = () => true;
export function holdOnly(has) {
  held = has;
}

// `go` opens a node: go({term: 123}) or go({code: "E11.22"}).
export function termView(view, go, chosen = new Set()) {
  const term = view.term;
  return h(
    "div",
    { class: "node-view" },
    h("p", { class: "crumbs" }, crumbs(view.ancestors.slice().reverse(), go), h("strong", {}, term.title)),
    h("p", { class: "muted" }, `${bookName(term.book)} · term ${term.id} · level ${term.level}`),
    term.nonessential.length
      ? h("p", {}, "Nonessential: ", term.nonessential.map((m) => h("span", { class: "tag" }, `(${m})`)))
      : null,
    section(
      "Codes it gives",
      view.pointers.map((p) =>
        h(
          "li",
          {},
          p.code ? codeLink(p.code, go) : h("span", { class: "muted" }, p.written),
          p.code && p.written !== p.code ? h("span", { class: "muted" }, ` written ${p.written}`) : null,
          p.column ? h("span", { class: "muted" }, ` · column ${p.column}`) : null,
        ),
      ),
    ),
    section(
      "Sends the coder to",
      view.xrefs.map((x) =>
        h(
          "li",
          {},
          h("span", { class: "tag" }, VIA[x.kind] ?? x.kind),
          " ",
          x.target !== null ? termLink(x.target, x.target_path ?? x.written, go) : h("span", { class: "muted" }, x.written),
        ),
      ),
    ),
    section(
      "Subterms",
      view.subterms.map((s) =>
        h("li", { class: chosen.has(s.id) ? "chosen" : "" }, termLink(s.id, s.title, go), s.nonessential.length ? h("span", { class: "muted" }, ` (${s.nonessential.join(") (")})`) : null),
      ),
    ),
    section(
      "Terms that send the coder here",
      view.xrefs_to.map((x) =>
        h("li", {}, termLink(x.term, x.source_path ?? String(x.term), go), " ", h("span", { class: "muted" }, VIA[x.kind] ?? x.kind)),
      ),
    ),
  );
}

export function codeView(view, go) {
  const code = view.code;
  const notes = groupNotes(view.notes);
  const inherited = groupNotes(view.inherited);
  return h(
    "div",
    { class: "node-view" },
    h("p", { class: "crumbs" }, crumbs(view.ancestors.slice().reverse(), go), h("strong", {}, code.key)),
    h("p", { class: "description" }, code.description),
    h(
      "p",
      {},
      h("span", { class: `tag ${code.billable ? "billable" : ""}` }, code.billable ? "billable" : "not billable"),
      " ",
      h("span", { class: "tag" }, code.kind),
      code.placeholder ? h("span", { class: "tag" }, "placeholder X") : null,
    ),
    view.seventh.length
      ? section("7th characters", view.seventh.map(([c, meaning]) => h("li", {}, h("span", { class: "code" }, c), " ", meaning)))
      : null,
    Object.entries(notes).map(([kind, rows]) => noteSection(NOTES[kind] ?? kind, rows, view.edges, go)),
    Object.keys(inherited).length
      ? h(
          "details",
          {},
          h("summary", {}, "Notes it inherits"),
          Object.entries(inherited).map(([kind, rows]) => noteSection(`${NOTES[kind] ?? kind} (from ${rows[0].node})`, rows, [], go)),
        )
      : null,
    section("Codes under it", view.children.map((c) => h("li", {}, codeLink(c.key, go), " ", c.description))),
    section(
      "Index terms that give it",
      view.terms.map((t) => h("li", {}, termLink(t.term, t.path, go), t.written !== code.key ? h("span", { class: "muted" }, ` (${t.written})`) : null)),
    ),
    section(
      "Notes elsewhere that name it",
      view.named_by.map((e) => h("li", {}, codeLink(e.source, go), " ", h("span", { class: "muted" }, `${NOTES[e.kind] ?? e.kind}: “${e.note}”`))),
    ),
  );
}

export function searchView(view, go) {
  return h(
    "div",
    { class: "search-results" },
    section(
      "Alphabetic Index",
      view.index.map((hit) =>
        h("li", {}, termLink(hit.term.id, hit.term.path, go), hit.term.book !== "index" ? h("span", { class: "muted" }, ` ${bookName(hit.term.book)}`) : null),
      ),
      "No Index term matches.",
    ),
    view.tabular.length || !view.index.length
      ? section(
          "Tabular List",
          view.tabular.map((hit) => h("li", {}, codeLink(hit.code.key, go), " ", hit.text, hit.kind !== "description" ? h("span", { class: "muted" }, ` (${hit.kind})`) : null)),
          "No Tabular text matches.",
        )
      : null,
  );
}

function noteSection(title, rows, edges, go) {
  return section(
    title,
    rows.map((note) => {
      const named = edges.filter((e) => e.note_id === note.id && e.target);
      return h("li", {}, note.text, named.length ? h("span", { class: "refs" }, " → ", named.map((e) => codeLink(e.target, go))) : null);
    }),
  );
}

function groupNotes(notes) {
  const out = {};
  for (const note of notes) (out[note.kind] ??= []).push(note);
  return out;
}

function section(title, items, empty) {
  if (!items.length && !empty) return null;
  return h("section", {}, h("h4", {}, title), items.length ? h("ul", {}, items) : h("p", { class: "muted" }, empty));
}

function crumbs(ancestors, go) {
  return ancestors.flatMap((a) => [
    "id" in a ? termLink(a.id, a.title, go) : codeLink(a.key, go, a.description),
    h("span", { class: "sep" }, " › "),
  ]);
}

function termLink(id, label, go) {
  if (!held({ term: id })) return h("span", { class: "unheld", title: "Not in the demo" }, label);
  return h("a", { href: `#/term/${id}`, onclick: (event) => follow(event, () => go({ term: id })) }, label);
}

function codeLink(key, go, title) {
  if (!held({ code: key })) {
    return h("span", { class: "code unheld", title: title ? `${title} (not in the demo)` : "Not in the demo" }, key);
  }
  return h("a", { class: "code", href: `#/code/${encodeURIComponent(key)}`, title, onclick: (event) => follow(event, () => go({ code: key })) }, key);
}

function follow(event, action) {
  if (event.metaKey || event.ctrlKey || event.shiftKey) return; // a new tab follows the href
  event.preventDefault();
  action();
}

function bookName(book) {
  return { index: "Alphabetic Index", eindex: "External Cause Index", neoplasm: "Table of Neoplasms", drug: "Table of Drugs and Chemicals" }[book] ?? book;
}
