// Where the UI reads its data: the running service (`text2icd serve`), or, in the static demo,
// the files exported next to the page (task 10.13). Both answer with the same JSON.

export function makeSource() {
  const meta = document.querySelector('meta[name="text2icd-source"]');
  return meta && meta.content === "static" ? staticSource() : liveSource();
}

function liveSource() {
  return {
    kind: "live",
    health: () => call("healthz"),
    code: (request) => call("v1/code", json("POST", request)),
    codePage: (release, key) => call(`v1/${enc(release)}/codes/${enc(key)}`),
    termPage: (release, id) => call(`v1/${enc(release)}/terms/${enc(id)}`),
    search: (release, q, book) => {
      const query = new URLSearchParams({ q, k: "30" });
      if (book) query.set("book", book);
      return call(`v1/${enc(release)}/search?${query}`);
    },
    // The local review mode (`text2icd serve --review`, task 10.12).
    review: {
      progress: () => call("v1/review"),
      item: (n) => call(`v1/review/${n}`),
      check: (n, line) => call(`v1/review/${n}/check`, json("POST", { line })),
      answer: (n, body) => call(`v1/review/${n}`, json("PUT", body)),
    },
  };
}

function json(method, body) {
  return { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
}

function staticSource() {
  // The demo's files (`text2icd demo export`): data/index.json lists the answers and the pages it
  // holds; each answer and each page is a file.
  let listing = null;
  let held = null;
  const read = (path) => call(`data/${path}`);
  const index = async () => {
    if (!listing) {
      listing = await read("index.json");
      held = { codes: new Set(listing.pages.codes), terms: new Set(listing.pages.terms) };
    }
    return listing;
  };
  return {
    kind: "static",
    health: async () => (await index()).health,
    examples: async () => (await index()).examples,
    // Whether a term or code page is in the demo (after health()); the UI does not link the others.
    has: (target) => !held || (target.term !== undefined ? held.terms.has(target.term) : held.codes.has(target.code)),
    code: async (request) => {
      const { results } = await index();
      const found = results[key(request)];
      if (found) return read(found);
      const phrase = normal(request.phrase);
      const settings = Object.keys(results)
        .filter((k) => k.endsWith(`|${phrase}`))
        .map((k) => k.split("|")[0]);
      if (settings.length) {
        throw new Error(`The demo holds this phrase ${settings.join(" and ")}, with the 7th character as stated: choose that, or run text2icd serve.`);
      }
      throw new Error("The demo holds its example phrases only. To code any phrase, install text2icd and run text2icd serve.");
    },
    codePage: (release, code) => read(`${release}/codes/${code}.json`).catch(missing),
    termPage: (release, id) => read(`${release}/terms/${id}.json`).catch(missing),
    search: async () => {
      throw new Error("Search needs the service: run text2icd serve.");
    },
  };
}

// The answer's key in the demo's index.json (the export computes the same, `demo.key`).
export function key(request) {
  return `${request.setting}|${request.episode ?? ""}|${normal(request.phrase)}`;
}

function normal(phrase) {
  return phrase.trim().split(/\s+/).join(" ").toLowerCase();
}

function missing() {
  throw new Error("Not in the demo: it holds the pages its examples touch.");
}

async function call(path, init) {
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("The service is not reachable. Is text2icd serve running?");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(detail(body) || `HTTP ${response.status}`);
  return body;
}

function detail(body) {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
  }
  return "";
}

const enc = encodeURIComponent;
