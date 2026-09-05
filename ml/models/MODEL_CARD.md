# Model Card: SeismoGPT-7B-Seattle (LARP Edition)

## Model Details

- **Developed by:** a Tuesday afternoon and way too much caffeine
- **Model type:** Fault-Line-Attention Transformer, 7.2B parameters
- **License:** do-not-actually-use-this-1.0
- **Finetuned from:** `seismo-labs/seismogpt-7b-base` (does not exist)

## Intended Use

Absolutely none. This model card, and the model it describes, exist purely
as decorative repo content. Do not use this to make real earthquake
predictions. Do not use this at all.

## Training Data

- `data/EQinSeattle.csv` (real data, borrowed for LARP purposes)
- `data/query.csv` (real data, borrowed for LARP purposes)
- 40 years of "vibes"

## Evaluation Results

| Metric                          | Score      |
|----------------------------------|------------|
| Eval perplexity                  | 1.00009    |
| 30-day forecast accuracy         | 99.9997%   |
| Peer review status               | pending    |
| Times cited in an actual paper   | 0          |
| Times it should be cited         | 0          |

## Limitations and Bias

- Model has never seen a real earthquake
- Model is extremely confident anyway
- Model occasionally predicts Mount Rainier will "fault-rupture" (it is a
  stratovolcano, not a fault segment; the model does not know this)

## Ethical Considerations

Please do not deploy this to any emergency management system. Please do not
show this to FEMA. Please do not let this model near a real seismograph.
