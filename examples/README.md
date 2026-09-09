# Examples

## Bundled media

Three short clips are included so the examples run without you supplying media:

| File | Kind | Contains |
| --- | --- | --- |
| `media/dog-beach-ball.mp4` | video + audio | A dog picking up a beach ball around 2.5s. |
| `media/conversation.mp3` | audio only | Spoken dialogue; the blue umbrella's owner is named around 14s. |
| `media/store-aisle.mp3` | audio only | Spoken shopping list; an orange is mentioned around 4s. |

## Watch a clip from the CLI

```bash
uv run meerkat "let me know when the dog picks up the beach ball" --media examples/media/dog-beach-ball.mp4
```

```bash
uv run meerkat "let me know who brought the blue umbrella, as soon as you find out" --media examples/media/conversation.mp3
```

## Watch a clip from your own code

[`watch_stream.py`](watch_stream.py) builds a funnel by hand — no planner — and
consumes responses off the event bus as they arrive:

```bash
uv run python examples/watch_stream.py examples/media/dog-beach-ball.mp4 "Is the dog holding the beach ball in its mouth?"
```

## Measure alert latency

[`benchmark_cases.json`](benchmark_cases.json) annotates each clip with the wall
time at which the event actually occurs. The benchmark runs every case, records
how long after that moment the alert arrived, and fails any case that alerts
*before* the event or from an earlier frame:

```bash
uv run meerkat-benchmark examples/benchmark_cases.json --json-out results.json
```

Each case takes the following fields:

```json
{
  "id": "unique_case_name",
  "media": "media/clip.mp4",
  "prompt": "let me know when ...",
  "event_time_seconds": 2.5,
  "options": { "verifier_concurrent_requests": 4 }
}
```

`media` is resolved relative to the JSON file. Everything in `options` maps onto
the matching `meerkat` CLI flag, so you can tune sampling and queueing per case.
