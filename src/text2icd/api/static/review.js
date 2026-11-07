// The local review mode (task 10.12): the silver-label audit or the gold annotation, one item at a
// time, beside a lookup pane for the Index and the Tabular List. The phrases are MIMIC-IV text: only
// the item's number goes in the address, and lookups stay inside the page, so the browser's history
// never holds a phrase or the words looked up.

import { clear, h, plural } from "./dom.js";
import { codeView, searchView, termView } from "./explore.js";

const VERDICTS = [
  ["y", "Right", "The silver code is right for the phrase"],
  ["n", "Wrong", "The silver code is wrong for the phrase"],
  ["?", "Can't tell", "The phrase does not settle it"],
];
const ANSWERED = { coded: "coded", skip: "skipped: not a codable diagnosis", unsure: "unsure" };
const CODE = /^[A-TV-Z][0-9][0-9A-Z](\.?[0-9A-Z]{1,4})?$/i;

// One per page: the lookup pane and the keyboard keep their state from one item to the next.
export function reviewer(source, release, mode) {
  let keys = null;
  let codes = null; // the gold answer's codes field, which the lookup pane adds to
  const lookup = lookupPane(source, release, mode === "gold" ? (code) => codes && addCode(codes, code) : null);
  document.addEventListener("keydown", (event) => keys?.(event));

  async function show(main, n) {
    keys = null;
    let progress;
    try {
      progress = await source.review.progress();
    } catch (error) {
      clear(main, h("p", { class: "error" }, error.message));
      return;
    }
    if (!progress.total) {
      clear(main, h("p", { class: "muted" }, "The review file has no rows."));
      return;
    }
    if (n === null || n >= progress.total) {
      location.replace(`#/review/${(progress.next ?? progress.total - 1) + 1}`);
      return;
    }
    const go = (m) => {
      if (m >= 0 && m < progress.total) location.hash = `#/review/${m + 1}`;
    };
    let item;
    try {
      item = await source.review.item(n);
    } catch (error) {
      clear(main, h("p", { class: "error" }, error.message));
      return;
    }
    const next = () => (n + 1 < progress.total ? go(n + 1) : show(main, n));
    const view = mode === "audit" ? auditItem(item, source, lookup, next) : goldItem(item, source, lookup, next);
    codes = view.codes ?? null;
    keys = (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
        if (event.key === "Escape") event.target.blur();
        return;
      }
      if (event.key === "ArrowLeft") go(n - 1);
      else if (event.key === "ArrowRight") go(n + 1);
      else if (event.key === "/") {
        event.preventDefault();
        lookup.focus();
      } else view.key?.(event);
    };
    clear(
      main,
      h("p", { class: "privacy muted" }, "MIMIC-IV text: it stays on this machine. Do not copy a phrase into an issue, a chat or a screenshot."),
      h(
        "div",
        { class: "review" },
        h("section", { class: "review-item", "aria-label": "item" }, bar(progress, n, go, mode), view.el),
        lookup.el,
      ),
    );
    view.focus?.();
  }
  return { show };
}

function bar(progress, n, go, mode) {
  const meter = h("progress", { max: String(progress.total), value: String(progress.answered) });
  return h(
    "div",
    { class: "review-bar" },
    h("button", { type: "button", onclick: () => go(n - 1), disabled: n === 0, title: "Previous (←)" }, "‹"),
    h("span", { class: "position" }, `${mode === "audit" ? "Row" : "Phrase"} ${n + 1} of ${progress.total}`),
    h("button", { type: "button", onclick: () => go(n + 1), disabled: n + 1 >= progress.total, title: "Next (→)" }, "›"),
    progress.next !== null && progress.next !== n
      ? h("button", { type: "button", class: "link", onclick: () => go(progress.next) }, "first unanswered")
      : null,
    h("span", { class: "progress" }, meter, `${progress.answered} of ${progress.total} answered`),
  );
}

// -- The silver-label audit (task 3.6) ------------------------------------------------------------

function auditItem(item, source, lookup, next) {
  const comment = h("input", { type: "text", maxlength: "500", "aria-label": "comment", autocomplete: "off" });
  comment.value = item.comment;
  const status = h("p", { class: "status-line", "aria-live": "polite" });
  const buttons = VERDICTS.map(([key, label, title]) =>
    h("button", { type: "button", class: `verdict${item.verdict === key ? " on" : ""}`, title, onclick: () => save(key) }, label, h("kbd", {}, key)),
  );
  async function save(verdict) {
    buttons.forEach((b) => (b.disabled = true));
    try {
      await source.review.answer(item.n, { verdict, comment: comment.value });
      next();
    } catch (error) {
      clear(status, h("span", { class: "error" }, error.message));
      buttons.forEach((b) => (b.disabled = false));
    }
  }
  const el = h(
    "div",
    {},
    h("p", { class: "review-phrase" }, item.phrase),
    h("p", { class: "silver" }, h("span", { class: "muted" }, "Silver code "), codeButton(item.code, lookup), " ", item.description),
    h("p", { class: "muted question" }, "Is this code right for the phrase alone?"),
    h("div", { class: "verdicts" }, buttons),
    h("label", { class: "field" }, "Comment (optional)", comment),
    status,
    item.verdict
      ? h("p", { class: "answered" }, "Verdict: ", h("strong", {}, VERDICTS.find(([k]) => k === item.verdict)?.[1] ?? item.verdict), item.comment ? ` · ${item.comment}` : null)
      : null,
  );
  return {
    el,
    key: (event) => {
      if (VERDICTS.some(([k]) => k === event.key)) save(event.key);
    },
  };
}

// -- The gold annotation (task 3.7) ---------------------------------------------------------------

function goldItem(item, source, lookup, next) {
  const codes = h("input", {
    type: "text",
    class: "codes",
    maxlength: "500",
    placeholder: "1 2, E78.5",
    "aria-label": "codes",
    autocomplete: "off",
    spellcheck: "false",
  });
  const comment = h("input", { type: "text", maxlength: "500", "aria-label": "comment", autocomplete: "off" });
  if (item.answer) comment.value = item.answer.comment;
  const preview = h("div", { class: "preview", "aria-live": "polite" });
  const status = h("p", { class: "status-line", "aria-live": "polite" });
  let timer = null;
  let asked = 0;
  async function check() {
    const line = codes.value.trim();
    const ask = ++asked;
    if (!line) return clear(preview);
    try {
      const checked = await source.review.check(item.n, line);
      if (ask === asked) clear(preview, describe(checked));
    } catch (error) {
      if (ask === asked) clear(preview, h("p", { class: "error" }, error.message));
    }
  }
  codes.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(check, 200);
  });
  codes.addEventListener("change", check);
  codes.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      save(codes.value);
    }
  });
  const buttons = [
    h("button", { type: "button", class: "primary-button", onclick: () => save(codes.value), title: "Save the codes (Enter)" }, "Save the codes"),
    h("button", { type: "button", onclick: () => save("s"), title: "Not a codable diagnosis: a procedure, a plan, a fragment, a negation" }, "Skip"),
    h("button", { type: "button", onclick: () => save("?"), title: "The phrase could mean different things" }, "Unsure"),
  ];
  async function save(line) {
    if (!line.trim()) {
      clear(status, h("span", { class: "error" }, "Type the codes first, or skip."));
      codes.focus();
      return;
    }
    buttons.forEach((b) => (b.disabled = true));
    try {
      await source.review.answer(item.n, { line: line.trim(), comment: comment.value });
      next();
    } catch (error) {
      clear(status, h("span", { class: "error" }, error.message));
      buttons.forEach((b) => (b.disabled = false));
    }
  }
  const el = h(
    "div",
    {},
    h("p", { class: "review-phrase" }, item.phrase),
    h("h4", {}, "Billed for the admission"),
    item.choices.length
      ? h(
          "ol",
          { class: "choices" },
          item.choices.map((choice, i) =>
            h(
              "li",
              {},
              h("button", { type: "button", class: "add", title: `Add ${choice.code} to the codes`, onclick: () => addCode(codes, choice.code) }, String(i + 1)),
              codeButton(choice.code, lookup),
              h("span", {}, choice.description),
            ),
          ),
        )
      : h("p", { class: "muted" }, "No codes were billed for the admission."),
    h("label", { class: "field" }, "Codes for the phrase alone: primary first, a comma between diagnoses", codes),
    preview,
    h("label", { class: "field" }, "Comment (optional)", comment),
    h("div", { class: "actions" }, buttons),
    status,
    item.answer ? answered(item.answer) : null,
    rules(),
  );
  return { el, codes, focus: () => codes.focus() };
}

function describe(checked) {
  if (checked.error) return h("p", { class: "error" }, checked.error);
  if (checked.verdict !== "coded") return h("p", { class: "muted" }, ANSWERED[checked.verdict]);
  return [
    h("ul", {}, checked.codes.map((c) => h("li", {}, h("span", { class: "code" }, c.code), " ", c.description))),
    checked.diagnoses > 1 ? h("p", { class: "muted" }, plural(checked.diagnoses, "diagnosis", "diagnoses")) : null,
  ];
}

function answered(answer) {
  const shown = answer.verdict === "coded" ? answer.codes.map((c) => h("span", { class: "code", title: c.description }, c.code)) : [ANSWERED[answer.verdict]];
  return h(
    "p",
    { class: "answered" },
    "Answered: ",
    shown.flatMap((part, i) => (i ? [" ", part] : [part])),
    answer.diagnoses > 1 ? ` · ${plural(answer.diagnoses, "diagnosis", "diagnoses")}` : null,
    answer.comment ? ` · ${answer.comment}` : null,
    h("span", { class: "muted" }, ". Saving again replaces it."),
  );
}

function rules() {
  return h(
    "details",
    { class: "rules" },
    h("summary", {}, "How to answer"),
    h(
      "ul",
      {},
      h("li", {}, "Code the phrase alone, as if it were all the documentation. The billed codes are suggestions."),
      h("li", {}, "Primary code first. The codes of one diagnosis go together (E11.22 N18.31); a comma separates diagnoses (I10, E78.5)."),
      h("li", {}, "Uncertain diagnoses are coded as if established; injuries take the 7th character A unless the phrase says otherwise."),
      h("li", {}, "Skip what is not a codable diagnosis (a plan, a procedure, a negation); mark Unsure what the phrase does not settle."),
      h("li", {}, "Look up the Index and the Tabular List on the right (/ to search). The full rules are in docs/annotation-guide.md."),
    ),
  );
}

function addCode(field, code) {
  const text = field.value.trimEnd();
  field.value = text && !text.endsWith(",") ? `${text} ${code}` : `${text}${text ? " " : ""}${code}`;
  field.dispatchEvent(new Event("input"));
  field.focus();
}

function codeButton(code, lookup) {
  return h("button", { type: "button", class: "code link", title: `Show ${code}`, onclick: () => lookup.open({ code }) }, code);
}

// -- The lookup pane ------------------------------------------------------------------------------

function lookupPane(source, release, onAdd) {
  const trail = [];
  const q = h("input", { type: "search", maxlength: "200", placeholder: "Index terms, or a code", "aria-label": "look up" });
  const back = h("button", { type: "button", class: "link back", hidden: true, onclick: () => (trail.pop(), open(trail.pop())) }, "‹ back");
  const body = h(
    "div",
    { class: "lookup-body" },
    h("p", { class: "muted" }, "Search the Alphabetic Index and the Tabular List, or open a code. Lookups stay in this page."),
  );
  const form = h(
    "form",
    {
      class: "lookup-search",
      onsubmit: (event) => {
        event.preventDefault();
        const text = q.value.trim();
        if (text) open(CODE.test(text) ? { code: dotted(text) } : { search: text });
      },
    },
    q,
    h("button", { type: "submit" }, "Look up"),
  );
  async function open(target) {
    if (!target) return;
    trail.push(target);
    back.hidden = trail.length < 2;
    clear(body, h("p", { class: "muted" }, "Loading…"));
    try {
      if (target.search !== undefined) {
        clear(body, searchView(await source.search(release, target.search, null), open));
      } else if (target.term !== undefined) {
        clear(body, termView(await source.termPage(release, target.term), open));
      } else {
        const view = await source.codePage(release, target.code);
        clear(body, onAdd ? addButton(view.code, onAdd) : null, codeView(view, open));
      }
    } catch (error) {
      clear(body, h("p", { class: "error" }, error.message));
    }
  }
  return { el: h("aside", { class: "lookup", "aria-label": "lookup" }, form, back, body), open, focus: () => q.focus() };
}

function addButton(code, onAdd) {
  if (!code.billable) return h("p", { class: "muted add-hint" }, `${code.key} is not billable: pick a code under it.`);
  return h("p", { class: "add-hint" }, h("button", { type: "button", onclick: () => onAdd(code.key) }, `Add ${code.key} to the codes`));
}

function dotted(text) {
  const code = text.toUpperCase().replace(".", "");
  return code.length > 3 ? `${code.slice(0, 3)}.${code.slice(3)}` : code;
}
