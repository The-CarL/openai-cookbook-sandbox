# openai-cookbook-sandbox

Personal study repo for the OpenAI API stack — Solutions Engineer onboarding
material. Each exercise is a single runnable script that demonstrates one
concept end-to-end. `.py` files are canonical; the `notebooks/` copies cover
the first 18 and lag behind.

## Setup

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run python openai_lab/01_basic_response.py
```

## Exercises

Numbered to be read in order — each builds on the previous.

### Foundations
| # | Topic | Why |
|---|---|---|
| 01 | Basic Responses API call | The core primitive: `client.responses.create()` |
| 02 | Multi-turn via `previous_response_id` | API-managed conversation state |
| 03 | Streaming events | TTFT, event types, `response.completed` |
| 04 | Model comparison (4.1 / 5.4 / 5.5 / 5.6 / 6) | Cost vs latency vs quality picker |

### Built-in tools
| # | Topic | Why |
|---|---|---|
| 05 | Web search | Citations, annotations, output items |
| 06 | Code interpreter | Sandboxed Python for data work |
| 07 | File search + vector stores | Enterprise RAG without infra |
| 08 | Multi-tool composition | web_search + code_interpreter in one call |

### Structured output & functions
| # | Topic | Why |
|---|---|---|
| 09 | Structured output (`text.format`) | Strict-mode JSON schemas |
| 10 | Nested schemas | Real-world enterprise objects |
| 11 | Function calling | Custom function definitions, call_id flow |
| 12 | Agentic loop | Multi-step tool use until done |
| 13 | Parallel function calls | One round-trip, many tools |

### Embeddings & retrieval
| # | Topic | Why |
|---|---|---|
| 14 | Embeddings + dimension reduction | `text-embedding-3-*`, free dim reduction |
| 15 | Similarity search from scratch | What `file_search` does under the hood |

### Production patterns
| # | Topic | Why |
|---|---|---|
| 16 | State management patterns | `previous_response_id` vs DB-managed vs hybrid |
| 17 | Error handling | Common failure modes + retry/backoff |
| 18 | Cost tracking | Token accounting + cached input math |

### Newer model capabilities
| # | Topic | Why |
|---|---|---|
| 19 | Image generation | `image_generation` tool, options, multi-tool |
| 20 | Reasoning effort | GPT-5.x reasoning levels + o-series comparison |
| 21 | MCP | Remote tool servers (DeepWiki, dmcp, custom) |
| 22 | Shell tool | Hosted container, `gpt-5.5`, env options |
| 23 | Computer use | GA `{"type": "computer"}`, action loop pattern |
| 24 | Agents SDK | `Agent`, `Runner`, handoffs, guardrails |

### 2026 SOTA additions
| # | Topic | Why |
|---|---|---|
| 25 | Prompt caching | Verifying cache hits, the 10% rule, invalidation |
| 26 | Context compaction (Feb 2026) | `context_management` + `responses.compact()` + `prompt_cache_retention` |
| 27 | Agent skills (Feb 2026) | `SKILL.md` bundles, composing with tools |
| 28 | Evals | Programmatic checks + LLM-as-judge, model A/B |
| 29 | Apply patch (Mar 2026) | Codex-style file editing via V4A diffs |
| 30 | Tool search (Mar 2026) | `namespace` + `defer_loading` for huge tool surfaces |
| 31 | `phase` field (Feb 2026) | Separate `commentary` from `final_answer` in agent UIs |
| 32 | gpt-image-2 (Apr 2026) | Direct Images API: generation, editing, token pricing, Batch |
| 33 | Realtime API v2 (May 2026) | `gpt-realtime-2` / translate / whisper WebSocket voice agents |
| 34 | Inline moderation (Jun 2026) | Safety scores alongside `responses.create()` in one call |
| 35 | GPT-5.6 family (Jul 2026) | Sol/Terra/Luna naming, cache breakpoints, post-launch price cuts |

## Model lineup snapshot (verified October 4, 2026)

| Model | Input $/M | Output $/M | Context | When to reach for it |
|---|---|---|---|---|
| `gpt-4.1-nano` | 0.10 | 0.40 | 1M | Classification, routing, cheap extraction |
| `gpt-4.1-mini` | 0.40 | 1.60 | 1M | High-volume production where 5.x is overkill |
| `gpt-4.1` | 2.00 | 8.00 | 1M | 1M context without needing reasoning |
| `gpt-5.4-nano` | 0.20 | 1.25 | — | Budget reasoning. Compaction only (no tool search / computer) |
| `gpt-5.4-mini` | 0.75 | 4.50 | 400K | Legacy agentic workloads. Tool search, computer, compaction |
| `gpt-5.4` | 2.50 | 15.00 | 1M | Cheaper than 5.5; computer use, image gen, native compaction |
| `gpt-5.4-pro` | — | — | 1M | March 5: computationally intensive problems |
| `gpt-5.5` | 5.00 | 30.00 | 1M | Apr 24 flagship. Token-efficient → often cheaper end-to-end |
| `gpt-5.5-pro` | 30.00 | 180.00 | 1M | Hardest reasoning, unchanged from 5.4 Pro pricing |
| `gpt-5.6-luna` | 0.20 | 1.20 | 1.05M | GA Jul 9. Cost-efficient 5.6 tier; cache writes 1.25× input |
| `gpt-5.6-terra` | 2.00 | 12.00 | 1.05M | GA Jul 9. Balanced 5.6 tier; stronger coding/cybersecurity |
| `gpt-5.6-sol` | 4.00† | 20.00† | 1.05M | GA Jul 9. Flagship 5.6; promo price through Nov 2026 |
| `gpt-6-luna` | 0.10 | 0.50 | — | Sep 22. 50% cheaper than 5.6 Luna; text+image input |
| `gpt-6-sol` | 2.00 | 10.00 | — | Sep 22. 50% cheaper than 5.6 Sol; default for new high-quality flows |
| `gpt-6.1-sol` | 2.00 | 10.00 | 1.05M | Sep 29. Complex coding/professional work; Multi-agent beta; same tier as gpt-6-sol |
| `gpt-6-astra` | 10.00 | 50.00 | 1M | Sep 3. Hardest reasoning/coding/computer use/research; Fast + Ultrafast modes |
| `gpt-5.3-codex` | — | — | — | Feb 24: dedicated agentic coding model |
| `gpt-5.2-codex` | — | — | — | Jan 14: earlier codex generation |
| `o3` | 2.00 | 8.00 | — | Dedicated reasoning, complex proofs |
| `o4-mini` | 1.10 | 4.40 | — | Fast reasoning, math/code/visual |

† GPT-5.6 Sol promotional price (was $5/$30); active through at least Nov 21, 2026.

### Caching gotchas
- Cached input is ~10% of standard input across the GPT families.
- Verify hits via `usage.input_tokens_details.cached_tokens` (Exercise 25).
- **GPT-5.5 only supports extended prompt caching — in-memory caching is unsupported.**
- GPT-5.5 reasoning effort defaults to `medium`.
- **GPT-5.6+ cache writes cost 1.25× the input rate** (new billing model). Minimum cache lifetime: 30 min. Cached reads remain ~10% of input. Applies to GPT-6 as well. GPT-5.6 requires explicit cache breakpoints (Exercise 35).
- **GPT-6 Astra:** cached input $1.00/M; cache writes $12.50/M. Fast mode is 2× the standard rate ($20/$100 per 1M). **Ultrafast mode** (`service_tier: "ultrafast"`) reduces inter-token latency — pricing TBD. Sessions >272K input tokens have the same long-context surcharge as GPT-5.5 (2× input, 1.5× output for the full session).

### Other 2026 API updates and follow-up exercises

The following platform updates are relevant; items without a dedicated exercise are worth follow-up:

- **GPT Image models** (covered by ex. 32) — gpt-image-1.5, gpt-image-1-mini also available; Batch 50% off. **`dall-e-2` and `dall-e-3` removed May 12, 2026.**
- **Sora 2 / sora-2-pro** (Mar 12) — video gen up to 20s, 1080p, video extensions, Batch
- **`gpt-audio-1.5`** (Feb 23) — Chat Completions audio model
- **GPT-5.6 family** (GA July 9, 2026) — covered by Exercise 35; Sol/Terra/Luna tiers, explicit cache breakpoints, and post-launch price cuts.
- **GPT-6 Sol / Luna** (Sep 22, 2026) — gpt-6-sol ($2/$10/M) and gpt-6-luna ($0.10/$0.50/M); text+image input, text output; Responses and Chat Completions APIs. At least 50% lower per-token cost than GPT-5.6 counterparts. Same 1.25× cache-write billing.
- **GPT-6 Astra** (Sep 3, 2026) — `gpt-6-astra`, $10/$50 per 1M tokens, 1M context; cached input $1/M; Fast mode $20/$100; **Ultrafast mode** (Sep 2026): set `service_tier: "ultrafast"` in the Responses API to reduce inter-token latency. Compare via Exercise 04.
- **GPT-6.1 Sol** (Sep 29, 2026) — `gpt-6.1-sol`, $2/$10 per 1M tokens; cached read $0.10/M, cache write $2.50/M; 1,050,000-token context, 128K max output; same long-context surcharge as Astra (>272K: 2× in, 1.5× out); Multi-agent beta (Responses API): model can delegate to subagents in a single request.
- **GPT Image 2.5** (Sep 8, 2026) — Flare (speed) and Sunburst (detail), Sketch support, 50% latency reduction. Model IDs: `gpt-image-2.5-flare`, `gpt-image-2.5-sunburst`.
- **Assistants API sunset (Aug 26, 2026)** — `/v1/assistants`, `/v1/threads`, and `/v1/threads/runs` are shut down. This repo uses the Responses API throughout; migrate with Responses API + Conversations API.
- **Secure MCP Tunnel** (June 2026) — enterprise feature allowing ChatGPT, Codex, Responses API, and AgentKit to connect to private or on-prem MCP servers without public exposure
- **WebSocket mode for Responses API** (Feb 23)
- **Open Responses spec** (Jan 15) — open-source multi-provider interop
- **Agents SDK update** (Apr 15) — controlled sandboxes, inspectable harness, memory
- **Hosted Evals product** (`client.evals.*`)
- **Batch API** (50% pricing for async workloads)
- **Background mode** for long-running responses
- **Fine-tuning + distillation**
