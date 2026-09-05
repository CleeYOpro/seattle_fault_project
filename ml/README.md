# ml/ — Seattle Fault "ML" Division

Everything in this directory is a LARP. It exists purely to make the repo
look like it has an ambitious, over-engineered, and slightly unhinged
machine learning program bolted onto a geology project. None of the models,
metrics, checkpoints, or training runs described here are real.

## What's here

- `notebooks/` — five increasingly unhinged notebooks, from fine-tuning a
  fictional 7B-parameter "SeismoGPT" to an agentic crew that emails the
  USGS on your behalf
- `training/` — standalone training scripts that print very convincing
  fake logs and then exit
- `configs/` — YAML configs for models that do not exist
- `models/` — model card and stub files standing in for multi-gigabyte
  checkpoints we obviously did not commit to git
- `data/` — small synthetic CSV/JSON files, clearly fabricated
- `logs/` — fake wandb / training console output for flavor

## Should you run any of this?

No. `requirements.txt` references packages that do not exist
(`tensorflow-quantum-blockchain`, `seismogpt`, `chatgpt-whisperer`) on
purpose. The Python scripts under `training/` are self-contained enough to
actually run (they just print fabricated numbers), but the notebooks import
libraries that were never real to begin with.

## Should you cite any of this in real seismology research?

Also no.
