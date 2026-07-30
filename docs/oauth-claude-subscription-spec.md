# Claude Pro/Max subscription OAuth — implementation spec

**Status:** specification only. Nothing in this document is implemented in grain today.
**Audience:** the engineer implementing the follow-on package. Assumes no prior context.
**Ledger item:** G7 ("OAuth parsed but login flow not wired" — see *Correction to the ledger*).

Upstream reference is the TypeScript `pi` repo at pinned commit **`34239180`**. Every constant,
header and algorithm below was read out of that source; each carries its `file:line`. Where this
document says "upstream", it means that commit. **Upstream is the specification** — where its
behavior looks odd, match it anyway and note the oddity rather than improving on it.

---

## 0. Read this first: what this work actually is

The obvious reading of G7 is "grain's OAuth constants are wrong, fix them." That reading is
wrong and will cause the package to under-scope.

grain's Anthropic OAuth config is aimed at an entirely different authorization server
(`console.anthropic.com`, the API-key console) than the one that issues Claude Pro/Max
subscription tokens (`claude.ai` + `platform.claude.com`). Correcting the constants is necessary
and is roughly a day's work. But even with perfect constants, **the resulting token cannot
currently reach the wire correctly**, for three independent reasons:

1. A subscription token must be sent as `Authorization: Bearer …`, *not* `x-api-key`. grain's
   only auth channel is `AuthData::from_single(token)`
   (`grain-llm-genai/src/builder.rs:243`), which the genai Anthropic adapter renders as
   `x-api-key`.
2. Four request headers must be present (`anthropic-beta`, `user-agent`, `x-app`, plus `accept`).
   grain-llm-genai has **no** custom-header injection path at all — confirmed by grep across
   `builder.rs` and `provider.rs`.
3. The request body must carry a Claude Code identity system block *prepended* to the caller's
   system prompt. Nothing in grain shapes the outbound body this way.

So the real shape of this work is **"OAuth credential lifecycle" + "a transport that can express
Bearer auth, custom headers, and a prepended system block."** See §7 for why that should almost
certainly land on the native Anthropic transport rather than fighting genai, and §8 for the
staging that lets the credential half be built and tested before the transport half exists.

### Correction to the ledger

Two things the debt ledger says about G7 are inaccurate, and the follow-on package should not
plan against them:

- **"Login flow not wired" is false.** It is wired: `grain-ai-agent-headless/src/cli.rs:265` and
  `grain-ai-agent-tui/src/agent_worker.rs:3068` both call
  `grain_llm_genai::oauth::start_login_flow`, and `grain-llm-genai/src/builder.rs:243` consumes
  the stored token. A complete PKCE + local-callback + refresh implementation exists in
  `grain-llm-genai/src/oauth.rs` (913 lines).
- **The real defect is worse than "not wired."** The flow runs end to end and *appears* to
  succeed against the wrong authorization server. A user can complete a browser login and get a
  stored token that cannot authorize inference — the failure surfaces later, as an auth error on
  the first request, not at login. The scope missing is not the flow, it is the flow's target and
  the transport underneath it.

---

## 1. Constants

All from `packages/ai/src/auth/oauth/anthropic.ts` unless noted.

| Constant | Value | Source |
|---|---|---|
| `CLIENT_ID` | `9d1c250a-e61b-44d9-88ed-5944d1962f5e` | `:29` |
| `AUTHORIZE_URL` | `https://claude.ai/oauth/authorize` | `:30` |
| `TOKEN_URL` | `https://platform.claude.com/v1/oauth/token` | `:31` |
| `CALLBACK_HOST` | env `PI_OAUTH_CALLBACK_HOST`, else `127.0.0.1` | `:32` |
| `CALLBACK_PORT` | `53692` (fixed, not dynamic) | `:33` |
| `CALLBACK_PATH` | `/callback` | `:34` |
| `REDIRECT_URI` | `http://localhost:53692/callback` | `:35` |
| `SCOPES` | `org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload` | `:36-37` |

Three details that are easy to get wrong:

- **The client id is base64-encoded in upstream source** as
  `decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")` (`:28-29`). The decoded value is
  in the table. The encoding is obfuscation, not a secret — this is a public client, which is why
  PKCE is mandatory. Store it decoded; do not reproduce the base64 dance.
- **`REDIRECT_URI` says `localhost` but the server binds `127.0.0.1`** (`:35` vs `:157`). These
  are deliberately not the same string. The redirect URI is registered with the authorization
  server and must be sent byte-exact; the bind address is a local concern. Do not "fix" the
  mismatch by making them agree.
- **The port is fixed.** grain currently binds a random port. A fixed port is required because
  the redirect URI is pre-registered — a random port produces a redirect-URI mismatch and the
  authorization server rejects the exchange. The cost is that login fails if 53692 is occupied;
  upstream accepts that, and so should we (surface a clear error naming the port).

---

## 2. PKCE

`packages/ai/src/auth/oauth/pkce.ts:21-34`.

```
verifier  = base64url( 32 cryptographically-random bytes )     // no padding
challenge = base64url( SHA-256( utf8_bytes(verifier) ) )       // no padding
method    = "S256"
```

base64url = standard base64 with `+`→`-`, `/`→`_`, `=` stripped (`pkce.ts:9-15`). The SHA-256 is
computed over the **ASCII/UTF-8 bytes of the verifier string**, not over the raw 32 bytes.

### `state` is the verifier

`anthropic.ts:247` — the authorize request sends `state: verifier`. This is unusual and is
load-bearing in three places, so do not "improve" it by generating an independent nonce:

- the local callback compares the returned `state` against the verifier (`:138`, via
  `startCallbackServer(verifier)` at `:231`);
- the manual-paste path validates `parsed.state !== verifier` (`:279`, `:289`);
- **the token exchange sends `state` in the request body** (`:202`) — the authorization server
  expects it, so an independent nonce would have to be threaded there anyway.

grain currently generates `state` as a separate random value (`oauth.rs:609`), which breaks the
exchange against this server.

---

## 3. Authorization request

`anthropic.ts:239-251`. `GET`/browser-open to `AUTHORIZE_URL` with these query parameters. Order
is as upstream builds them; parameter order is not protocol-significant, but matching it makes
diffing against upstream trivial.

| Parameter | Value |
|---|---|
| `code` | `true` |
| `client_id` | `CLIENT_ID` |
| `response_type` | `code` |
| `redirect_uri` | `REDIRECT_URI` |
| `scope` | `SCOPES` (space-separated, URL-encoded) |
| `code_challenge` | PKCE challenge |
| `code_challenge_method` | `S256` |
| `state` | the PKCE **verifier** |

`code=true` is non-standard and required — it is what makes the server also render the code for
manual copy/paste. grain does not send it today.

---

## 4. Obtaining the code — two concurrent paths

Upstream races a local callback server against a manual paste prompt (`anthropic.ts:229-302`);
whichever arrives first wins, and the other is cancelled. The manual path exists for the real
case where the browser is on a different machine from the CLI (headless server, SSH, container).

**Local callback** (`:99-168`): HTTP server on `CALLBACK_HOST:CALLBACK_PORT`. Requests to a path
other than `CALLBACK_PATH` → 404. Then, in order: an `error` query parameter → 400 with the error
text; missing `code` or `state` → 400; `state !== expectedState` → 400 "State mismatch"; otherwise
200 with a success page and the `{code, state}` is resolved.

**Manual paste** (`parseAuthorizationInput`, `:52-80`) accepts four input shapes, tried in order:

1. a full redirect **URL** → read `code` and `state` query params;
2. a string containing `#` → **split on the first `#` into `code`, `state`** (this is the
   `code#state` convention the authorization server's copy button produces);
3. a string containing `code=` → parse as a query string;
4. anything else → treat the whole string as the bare code, with `state` defaulting to the
   verifier (`:281`).

If a state is present and does not equal the verifier, fail with `OAuth state mismatch`
(`:279`, `:289`). grain implements none of this — it has no manual fallback at all.

---

## 5. Token exchange and refresh

Both are `POST` with a **JSON** body — `Content-Type: application/json`, `Accept:
application/json`, 30-second timeout (`postJson`, `:170-188`). **grain sends form-encoded for
both** (`oauth.rs:324-329` exchange, `oauth.rs:452-457` refresh); that is a wire-format mismatch
against this server, not a stylistic difference.

**Exchange** (`:198-205`):

```json
{ "grant_type": "authorization_code", "client_id": "…", "code": "…",
  "state": "…", "redirect_uri": "…", "code_verifier": "…" }
```

Note `state` is in the body, and `client_id` is in the body (not an `Authorization` header).

**Refresh** (`:311-315`):

```json
{ "grant_type": "refresh_token", "client_id": "…", "refresh_token": "…" }
```

Both responses parse as `{ access_token, refresh_token, expires_in }` (refresh may also carry
`scope`, which upstream reads but ignores — `:320`). **Refresh returns a rotated refresh token;
persist it.** Both paths compute expiry identically (`:225`, `:338`):

```
expires = now_ms + expires_in * 1000 - 5 * 60 * 1000
```

i.e. a **5-minute safety margin is baked into the stored value**.

### Credential shape

`OAuthCredential` (`packages/ai/src/auth/types.ts:32-34`), persisted per provider:

```
{ type: "oauth", access: string, refresh: string, expires: number /* epoch ms */ }
```

Keep grain's existing store location and permissions —
`~/.config/grain/oauth/<profile>.json`, `0o600` on Unix (`oauth.rs:5`, `:277-301`). That part is
sound and does not need to change. grain's `StoredTokens` (`oauth.rs:137-147`) is a superset
(`id_token`, `api_key`, `token_type`) retained for the OpenAI/Codex profile; leave it alone and
map the Anthropic fields onto it. Note grain stores `expires_at` in **seconds**, upstream stores
`expires` in **milliseconds** — pick one and be consistent; if you keep seconds, convert at the
boundary and say so in a comment.

### When refresh fires

`packages/ai/src/auth/resolve.ts:95-138`:

```
DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000                    // :95
expiresSoon(c) = Date.now() + minimumValidityMs >= c.expires         // :110
```

Combined with the 5 minutes already subtracted at issue time, **a refresh fires roughly 10
minutes before the provider's real expiry.** grain currently subtracts 60 seconds at issue
(`oauth.rs:379`, `:502`) and refreshes only once `now >= expires_at` (`:535`, `:564`) — about a
60-second margin, which is too tight for a long tool-execution phase and will produce
mid-turn auth failures.

Upstream also does **single-flight refresh under a lock with a double check** (`resolve.ts:113-131`):
take the lock, re-read the credential, re-test `expiresSoon`, and bail if another process already
refreshed — then persist the rotated credential before releasing. This matters because the refresh
token rotates: two concurrent refreshes race, and the loser persists a refresh token the server
has already invalidated, logging the user out. grain has no such locking. The session file lock in
`grain-agent-harness/src/session_jsonl.rs` (`fs2` advisory lock) is a working in-repo pattern to
copy.

---

## 6. Applying the credential to requests

This is the part with no grain equivalent whatsoever.

**Detection** (`packages/ai/src/api/anthropic-messages.ts:843-845`):

```ts
function isOAuthToken(apiKey: string): boolean {
    return apiKey.includes("sk-ant-oat");
}
```

Upstream deliberately routes the access token through the ordinary `apiKey` channel
(`anthropic.ts:347-349`: `toAuth` returns `{ apiKey: credential.access }`) and then
*sniffs the token* at client-construction time to switch modes. Substring, not prefix.

**Client construction when the token is OAuth** (`anthropic-messages.ts:891-912`):

- `apiKey: null` and `authToken: <token>` — i.e. **`Authorization: Bearer …`, and `x-api-key` is
  not sent at all**. Sending both is an auth error.
- headers:

| Header | Value |
|---|---|
| `accept` | `application/json` |
| `anthropic-dangerous-direct-browser-access` | `true` |
| `anthropic-beta` | `claude-code-20250219,oauth-2025-04-20` + any other active betas, comma-joined |
| `user-agent` | `claude-cli/2.1.75` |
| `x-app` | `cli` |

`claudeCodeVersion = "2.1.75"` is a literal at `anthropic-messages.ts:76`. Pin it as a named
constant so the version bump is a one-line change.

Note the OAuth branch merges `model.headers` and `optionsHeaders` but — unlike the API-key branch
at `:915-926` — **does not merge `dynamicHeaders`, and does not add the session-affinity header**.
That asymmetry is upstream's; reproduce it rather than tidying it.

**Body shaping** (`anthropic-messages.ts:976-988`): when the token is OAuth, the Claude Code
identity block is **prepended as the first system block**, and the caller's system prompt is
appended as a **second** block:

```
system[0] = { type: "text", text: "You are Claude Code, Anthropic's official CLI for Claude." }
system[1] = { type: "text", text: <caller's system prompt> }        // only if non-empty
```

It is **not** a replacement of the caller's prompt — a detail worth stating because the phrase
"forced system prompt" suggests otherwise. Each block carries `cache_control` when caching is
active. Omitting or reordering this block is understood to cause the subscription endpoint to
reject the request.

---

## 7. The transport problem, and the recommended resolution

Sections 1–5 are a self-contained credential-lifecycle rewrite inside
`grain-llm-genai/src/oauth.rs`. Section 6 is not implementable there, because:

- genai's `AuthData::from_single(token)` maps to `x-api-key` for the Anthropic adapter; there is
  no `authToken`/Bearer equivalent exposed through the path grain uses
  (`grain-llm-genai/src/builder.rs:243`);
- grain-llm-genai injects no custom headers anywhere, so the four required headers have no route
  to the wire;
- the prepended system block is outbound body shaping that grain's outbound mapping
  (`grain-llm-genai/src/mapping/outbound.rs`) does not model.

**Recommendation: land §6 on the native Anthropic transport, not on genai.** A native transport
was already being scoped when this spec was written (it is the consumer of the `response_id` /
`response_model` slots added in WP19). It owns its own HTTP client, so Bearer auth, arbitrary
headers and body shaping are all natural there, and the seam gaps that motivated it (S-2, S-6,
and the S-3 usage double-count) are the same gaps that make the genai path unattractive.

If the native transport is not available in time, the fallback options, least bad first:

1. **Per-call header channel.** WP19 added `StreamOptions.headers` and `metadata` to
   `grain-agent-core` (`grain-agent-core/src/stream.rs`), with upstream's null-suppression
   semantics. Core-side plumbing exists and is tested; what is missing is genai-side consumption
   plus a Bearer path. Only viable if genai exposes both — verify before committing to it.
2. **A custom `ServiceTarget` / request interceptor in genai 0.6.5** (`Cargo.toml:36`), if one
   exists that can override auth and headers. Unverified — check before planning.
3. **Do not** attempt to smuggle a Bearer token through `AuthData`. Even if a header ends up
   looking right, `x-api-key` would still be sent alongside it.

---

## 8. Suggested staging

Two stages, because stage A is independently valuable, independently testable, and unblocked:

**Stage A — credential lifecycle** (~1–1.5 days), entirely in `grain-llm-genai/src/oauth.rs`:
constants (§1), PKCE with `state == verifier` (§2), authorize params incl. `code=true` (§3),
fixed-port callback + manual-paste parsing (§4), JSON exchange/refresh, 5-minute margins, rotated
refresh-token persistence, single-flight locking (§5). Fully testable offline against a stub
authorization server — no subscription account needed.

**Stage B — request shaping** (~0.5–1 day *on the native transport*; unbounded on genai until
option 1 or 2 in §7 is verified): token sniffing, Bearer without `x-api-key`, the four headers,
the prepended identity block (§6).

**Total: ~2–2.5 days**, assuming Stage B lands on the native transport. Add a live verification
pass with a real Claude Pro/Max account — the whole failure mode here is that everything looks
fine until a real subscription token meets a real endpoint, so a green offline suite is
necessary but not sufficient.

### Coordination

`grain-llm-genai/src/oauth.rs` was **not** touched by WP19. WP19's only change to that crate is a
two-line mechanical `None` fill at `mapping/inbound.rs:346` and `:618`, marked
`WP19 mechanical fill only`, carrying no semantics. There is no conflict with this work.

---

## 9. Divergence summary

| # | Aspect | grain today | upstream requires | File to change |
|---|---|---|---|---|
| 1 | Authorize URL | `console.anthropic.com/oauth/authorize` (`oauth.rs:89`) | `claude.ai/oauth/authorize` | `oauth.rs` |
| 2 | Token URL | `console.anthropic.com/oauth/token` (`:90`) | `platform.claude.com/v1/oauth/token` | `oauth.rs` |
| 3 | Client id | `f7a5c308-…` (`:91`) | `9d1c250a-e61b-44d9-88ed-5944d1962f5e` | `oauth.rs` |
| 4 | Scopes | `openid profile email` (`:92`) | the 6-scope set incl. **`user:inference`** | `oauth.rs` |
| 5 | `code=true` | absent (`:620-628`) | required | `oauth.rs` |
| 6 | `state` | independent random (`:609`) | **equals the PKCE verifier** | `oauth.rs` |
| 7 | Redirect URI | dynamic random port (`:611-616`) | fixed `http://localhost:53692/callback` | `oauth.rs` |
| 8 | Exchange encoding | form-encoded (`:324-329`) | JSON, incl. `state` in body | `oauth.rs` |
| 9 | Refresh encoding | form-encoded (`:452-457`) | JSON | `oauth.rs` |
| 10 | Expiry margin | 60 s (`:379`, `:502`) | 5 min at issue **+** 5 min at check ≈ 10 min | `oauth.rs` |
| 11 | Concurrent refresh | unguarded | single-flight, double-checked, persist-before-release | `oauth.rs` |
| 12 | Manual paste | absent | URL / `code#state` / `code=…` / bare code | `oauth.rs` |
| 13 | Auth header | `x-api-key` via `AuthData::from_single` (`builder.rs:243`) | `Authorization: Bearer`, **no** `x-api-key` | transport (§7) |
| 14 | `anthropic-beta` | absent | `claude-code-20250219,oauth-2025-04-20` + active betas | transport |
| 15 | `user-agent` | absent | `claude-cli/2.1.75` | transport |
| 16 | `x-app` | absent | `cli` | transport |
| 17 | Identity system block | absent | prepended as `system[0]`, caller's prompt second | transport |

Rows 1–12 are Stage A; rows 13–17 are Stage B.

---

## 10. Test checklist

Offline, no account required:

- PKCE: verifier is 43 chars of base64url from 32 bytes, no padding, no `+` or `/`; challenge
  equals base64url(SHA-256(verifier-as-utf8)) for a **known fixed vector** — pin a hardcoded
  verifier→challenge pair so a future refactor cannot silently change the algorithm.
- Authorize URL contains all eight parameters, `code=true`, `code_challenge_method=S256`, and
  `state` byte-equal to the verifier.
- `parseAuthorizationInput` covers all four shapes plus: a `#` with an empty state, a URL with no
  `code`, and a state that mismatches the verifier (must error).
- Callback server: wrong path → 404; `error=` → 400; missing code → 400; mismatched state → 400;
  happy path → 200 and resolves.
- Exchange and refresh send JSON (assert `Content-Type` and the parsed body keys) against a stub
  server; assert `state` present in the exchange body.
- Expiry: stored expiry is `issue + expires_in − 5min`; `expiresSoon` fires at ≈10 min before real
  expiry; a rotated refresh token is persisted.
- Concurrency: two simultaneous resolves trigger exactly **one** refresh; the loser observes the
  winner's credential.
- Token store keeps `0o600` on Unix.
- Detection: `isOAuthToken` matches on the `sk-ant-oat` **substring**, anywhere in the string.

Requires a real Claude Pro/Max account (do not skip — see §8):

- End-to-end browser login → stored credential → a real inference request succeeding.
- Assert on the wire that `Authorization: Bearer` is present and `x-api-key` is **absent**.
- Assert `system[0]` is the identity block and the caller's prompt is `system[1]`.
- A forced refresh (rewind stored expiry) followed by a successful request.
