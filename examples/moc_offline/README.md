# Offline continuation example

These two independent campaign bundles contain the three confirmed initialization
measurements and one **synthetic** new result per arm. They are workflow fixtures,
not additional laboratory results or evidence of model accuracy.

The structured GP uses the source-derived sampler and full default draw counts.
The LLM arm uses deterministic mock completions and synthetic embeddings. Neither
arm makes provider calls. Each bundle resumes with four measured observations and
one reserved experiment. The matched manifest records initialization identity;
`verification.json` records the executed save/resume and replay checks.

Generate another isolated example from the repository root:

```powershell
.\.venv\Scripts\python.exe -m boicl.moc_cli demo --output .moc-demo/my-example
```

The `.json.gz` bundles are portable compressed JSON. Use the CLI import command
for compressed bundles, or decompress them before selecting the JSON bundle in
either browser view. Synthetic campaigns remain explicitly marked after import.

For generic integration examples, the adjacent `generic_max_fixture.csv` and
`generic_min_fixture.csv` files provide five candidate procedures with three
initial observations. Map `setting` as linear and `hold` as log; choose the
`response` objective (maximize, bounds −10 to 10) or `cost` objective (minimize,
bounds 0 to 50). Treat `esd`, `gof`, and `gap` as quality metadata, not features.
