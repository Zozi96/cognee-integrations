import { CogneeHttpClient } from "../src/client";

// ---------------------------------------------------------------------------
// fetchAPI success-path responseParser (gh #195 / SDK-242).
//
// fetchAPI accepts a responseParser so callers can read non-JSON bodies, but
// the success path used to hardcode response.json() and ignore it — only the
// 401-relogin path honored it. visualise() passes a text parser to read the
// graph HTML the server returns, so every healthy 200 threw
// "SyntaxError: Unexpected token < in JSON". These tests pin the success path
// to the caller-supplied parser: they FAIL against the pre-fix client (json()
// throws on the HTML body) and pass now.
// ---------------------------------------------------------------------------

let mockFetch: jest.Mock;
let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  mockFetch = jest.fn();
  (globalThis as unknown as { fetch: unknown }).fetch = mockFetch;
});

afterEach(() => {
  (globalThis as unknown as { fetch: unknown }).fetch = originalFetch;
  jest.restoreAllMocks();
});

const GRAPH_HTML = "<!doctype html><html><body><svg>graph</svg></body></html>";

// A 200 Response whose body is HTML: text() returns it, json() throws exactly
// as a real Response.json() does on a non-JSON body.
function httpHtml(): Response {
  return {
    ok: true,
    status: 200,
    text: async () => GRAPH_HTML,
    json: async () => {
      throw new SyntaxError("Unexpected token < in JSON at position 0");
    },
  } as unknown as Response;
}

// apiKey is set so ensureAuth() short-circuits without a login round-trip.
function makeClient(): CogneeHttpClient {
  return new CogneeHttpClient("http://test", "key");
}

describe("fetchAPI success-path responseParser (gh #195)", () => {
  it("routes the success path through the supplied responseParser instead of forcing JSON", async () => {
    jest.useFakeTimers();
    try {
      const client = makeClient();
      mockFetch.mockResolvedValue(httpHtml());

      const body = await client.fetchAPI<string>(
        "/api/v1/visualize?dataset_id=d",
        { method: "GET" },
        5_000,
        async (r) => await r.text(),
        0, // retries disabled: one fetch settles the call
      );

      expect(body).toBe(GRAPH_HTML);
    } finally {
      jest.useRealTimers();
    }
  });

  it("visualise() resolves to the raw graph HTML rather than throwing a JSON parse error", async () => {
    jest.useFakeTimers();
    try {
      const client = makeClient();
      mockFetch.mockResolvedValue(httpHtml());

      await expect(client.visualise("dataset-1")).resolves.toBe(GRAPH_HTML);
    } finally {
      jest.useRealTimers();
    }
  });
});
