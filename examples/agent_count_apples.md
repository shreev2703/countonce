# Agent example: "How many ripe apples are on these trees?"

Install and register the server (stdio):

```bash
pip install 'countonce[mcp,recon]'
claude mcp add countonce -- countonce-mcp
```

The user attaches four overlapping photos and asks:

> How many ripe apples are on these trees?

The agent calls the `count_objects` tool:

```json
{
  "images": ["/data/orchard/row3_a.jpg", "/data/orchard/row3_b.jpg",
             "/data/orchard/row3_c.jpg", "/data/orchard/row3_d.jpg"],
  "prompt": "ripe red apples, not leaves",
  "config": {"max_frames": 8}
}
```

The tool returns (placeholders, not real output):

```json
{
  "ok": true,
  "count": "<N>",
  "zero_count": false,
  "interval": {"low": "<L>", "high": "<H>", "level": 0.9, "method": "<method>"},
  "warnings": [{"code": "<code>", "message": "<text>", "fix": "<text>"}],
  "result_path": "/tmp/countonce-mcp-xxxx/result.json"
}
```

The agent answers with the count and the interval, repeats any warnings and says that each apple
is counted once across photos. For example: "About <N> ripe apples (likely between <L> and
<H>). Warning: <message>. Details are in `result_path`."

On bad input the tool returns `{"ok": false, "error": {"code", "message", "tip", "retryable"}}`.
The agent should follow `tip`, for example by asking the user for photos that exist, and retry
only when `retryable` is true.
