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
| 04 | Model comparison (4.1 / 5.4 / 5.5 / 5.6) | Cost vs latency vs quality picker |

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

## Model lineup snapshot (verified September 10, 2026)

| Model | Input $/M | Output $/M | Context | When to reach for it |
|---|---|---|---|---|
| `gpt-4.1-nano` | 0.10 | 0.40 | 1M | Classification, routing, cheap extraction |
| `gpt-4.1-mini` | 0.40 | 1.60 | 1M | High-volume production where 5.x is overkill |
| `gpt-4.1` | 2.00 | 8.00 | 1M | 1M context without needing reasoning |
| `gpt-5.4-nano` | 0.20 | 1.25 | — | Budget reasoning. Compaction only (no tool search / computer) |
| `gpt-5.4-mini` | 0.75 | 4.50 | 400K | Default for agentic workloads where 5.6 is overkill |
| `gpt-5.4` | 2.50 | 15.00 | 1M | Cheaper per-token than 5.6 Sol; computer use, image gen |
| `gpt-5.4-pro` | — | — | 1M | March 5: computationally intensive problems |
| `gpt-5.5` | 5.00 | 30.00 | 1M | Apr 24 flagship; superseded by 5.6 Sol for new projects |
| `gpt-5.5-pro` | 30.00 | 180.00 | 1M | Hardest reasoning, unchanged from 5.4 Pro pricing |
| `gpt-5.6-luna` | 0.20 | 1.20 | 1M | Budget 5.6 tier (Jul 30 price cut); best cost/quality for simple tasks |
| `gpt-5.6-terra` | 2.00 | 12.00 | 1M | Mid-tier 5.6 (Jul 30 price cut); replaces 5.5 for cost-sensitive agentic work |
| `gpt-5.6-sol` | 4.00 (promo) | 20.00 (promo) | 1M | Flagship 5.6 (GA Jul 9). Promo pricing through Nov 21; list $5/$30 |
| `gpt-5.3-codex` | — | — | — | Feb 24: dedicated agentic coding model |
| `gpt-5.2-codex` | — | — | — | Jan 14: earlier codex generation |
| `o3` | 2.00 | 8.00 | — | Dedicated reasoning, complex proofs |
| `o4-mini` | 1.10 | 4.40 | — | Fast reasoning, math/code/visual |

### Caching gotchas
- Cached input reads are ~10% of standard input across the GPT families.
- Verify hits via `usage.input_tokens_details.cached_tokens` (Exercise 25).
- **GPT-5.5 only supports extended prompt caching — in-memory caching is unsupported.**
- **GPT-5.6+ introduces explicit cache breakpoints**: writes cost 1.25× input rate; reads cost 10% as before. 30-minute minimum cache lifetime.
- GPT-5.5 reasoning effort defaults to `medium`.

### Other 2026 API capabilities not yet covered

The following exist on the platform and are worth follow-up exercises:

- **GPT Image models** (covered by ex. 32) — gpt-image-1.5, gpt-image-1-mini also available; Batch 50% off. **`dall-e-2` and `dall-e-3` removed May 12, 2026.**
- **Sora 2 / sora-2-pro** (Mar 12) — video gen up to 20s, 1080p, video extensions, Batch
- **`gpt-audio-1.5`** (Feb 23) — Chat Completions audio model
- **GPT-5.6 explicit cache breakpoints** — GA July 9, 2026. Model is now in the lineup table (ex 04/18). New caching mechanic: writes billed at 1.25× input rate, reads at 10% (same as before); 30-min minimum cache lifetime; not yet covered by a dedicated exercise
- **Secure MCP Tunnel** (June 2026) — enterprise feature allowing ChatGPT, Codex, Responses API, and AgentKit to connect to private or on-prem MCP servers without public exposure
- **WebSocket mode for Responses API** (Feb 23)
- **Open Responses spec** (Jan 15) — open-source multi-provider interop
- **Agents SDK update** (Apr 15) — controlled sandboxes, inspectable harness, memory
- **Hosted Evals product** (`client.evals.*`)
- **Batch API** (50% pricing for async workloads)
- **Background mode** for long-running responses
- **Fine-tuning + distillation**
