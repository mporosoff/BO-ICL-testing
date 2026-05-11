# BO-ICL Local Runner

Use `run_boicl_local.bat` on Windows to start a local browser app for BO-ICL
experiments.

The launcher:

- creates `.env` from `.env.example` when needed;
- creates `.venv` when needed;
- installs BO-ICL with the GPR extra and local app dependencies;
- prompts for `OPENAI_API_KEY` without echoing it to the terminal;
- starts `python -m boicl.local_app` and opens the browser.

Secrets stay in `.env`, which is ignored by Git. Do not paste real API keys into
tracked files.

## Dataset Format

Import a `.csv`, `.xlsx`, or `.xls` file. The first column must contain the
procedure text. If a second column exists and contains numeric values, those
values are loaded as existing observations. If a third column exists and contains
numeric values, those values are loaded as uncertainty.

Example:

```csv
procedure,C2 yield,uncertainty
"Synthesis procedure A",12.4,0.3
"Synthesis procedure B",,
"Synthesis procedure C",9.8,0.5
```

## Running

Double-click:

```text
run_boicl_local.bat
```

Or run manually:

```powershell
.\.venv\Scripts\python.exe -m boicl.local_app
```

## Notes

The current objective model is maximization-first, with a minimization option in
the local runner that internally negates the entered objective values before
training. Entered uncertainty is stored, exported, and plotted as an error bar.
The current BO-ICL `AskTellGPR` implementation does not yet use per-observation
uncertainty as fixed noise during GP fitting.
