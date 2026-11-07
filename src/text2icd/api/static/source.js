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
  // The demo's files: data/index.json lists the phrases it holds; each answer is a file.
  let listing = null;
  const read = (path) => call(`data/${path}`);
  const index = async () => (listing ??= await read("index.json"));
  return {
    kind: "static",
    health: async () => (await index()).health,
    examples: async () => (await index()).phrases,
    code: async (request) => {
      const found = (await index()).results[key(request)];
      if (!found) throw new Error("The demo holds the example phrases only: run text2icd serve to code others.");
      return read(found);
    },
    codePage: (release, code) => read(`${release}/codes/${code}.json`).catch(missing),
    termPage: (release, id) => read(`${release}/terms/${id}.json`).catch(missing),
    search: async () => {
      throw new Error("Search needs the service: run text2icd serve.");
    },
  };
}

export function key(request) {
  return `${request.setting}|${request.episode ?? ""}|${request.phrase}`;
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
