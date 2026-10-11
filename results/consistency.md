<!-- python eval/consistency.py -->
### Spread over sentences (3 seeds, vs same-language reference clip)

| system | language | n | mean | std | min |
|---|---|---|---|---|---|
| E1 | en | 180 | 0.502 | 0.068 | 0.324 |
| E1 | hi | 36 | 0.643 | 0.057 | 0.500 |
| E1 | hinglish | 48 | 0.589 | 0.061 | 0.443 |
| Final | en | 180 | 0.730 | 0.034 | 0.618 |
| Final | hi | 36 | 0.706 | 0.037 | 0.607 |
| Final | hinglish | 48 | 0.738 | 0.035 | 0.665 |

### Same voice across languages (cosine between per-language average embeddings; seed-1234 audio: E1 single run, E3 run 1)

| system | en-hi | en-hinglish | hi-hinglish |
|---|---|---|---|
| E1 (n en 60, hi 12, hinglish 16) | 0.797 | 0.863 | 0.893 |
| Final (n en 60, hi 12, hinglish 16) | 0.962 | 0.964 | 0.990 |
| My real takes (n en 154, hi 97, hinglish 82) | 0.869 | 0.958 | 0.953 |
