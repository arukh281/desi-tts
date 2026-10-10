<!-- python eval/block15_summary.py --dir heldout -->

| setting | mean CER | name words missed | duration flags | guard retries |
|---|---|---|---|---|
| all | 0.039 | 36/90 = 0.40 | 2 | 0 |
| base | 0.037 | 31/90 = 0.34 | 1 | 0 |
| guard | 0.037 | 31/90 = 0.34 | 1 | 0 |
| respell | 0.046 | 41/90 = 0.46 | 0 | 0 |
| temp | 0.034 | 35/90 = 0.39 | 1 | 0 |

### CER: all vs base (seeds: 3 vs 3; 20 sentences)
| all | 20 | 0.037 | 0.039 | +0.003 | [-0.012, +0.017] | no |

### CER: guard vs base (seeds: 3 vs 3; 20 sentences)
| all | 20 | 0.037 | 0.037 | +0.000 | [+0.000, +0.000] | no |

### CER: respell vs base (seeds: 3 vs 3; 20 sentences)
| all | 20 | 0.037 | 0.046 | +0.009 | [-0.008, +0.026] | no |

### CER: temp vs base (seeds: 3 vs 3; 20 sentences)
| all | 20 | 0.037 | 0.034 | -0.003 | [-0.016, +0.010] | no |

