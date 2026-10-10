## E3 run 1, bench_norm_on
### Whisper word errors on results/runs/E3_run1/bench_norm_on (88 rows)

| type | wrong words | of all words of that type | rate |
|---|---|---|---|
| hinglish | 55 | 145 | 0.38 |
| hindi | 19 | 110 | 0.17 |
| number | 18 | 169 | 0.11 |
| acronym | 15 | 83 | 0.18 |
| name | 13 | 29 | 0.45 |
| english | 12 | 554 | 0.02 |

hinglish examples:
  b061: रिसीव -> रिसीव्ड
  b062: पांच -> पैंच सो रुपे
  b062: ड्यू -> डूए
  b063: ऑर्डर -> ओर्डर
  b063: बजे -> बार ये टाक्ट डिलिवर
  b063: तक -> बार ये टाक्ट डिलिवर

hindi examples:
  b077: भुगतान -> भुपतान
  b078: आपकी -> आपके
  b078: कल -> कर
  b078: साढे -> सारे
  b079: बताइए -> बताईए
  b080: डिलीवरी -> डिलिवरी

number examples:
  b032: six -> (missing)
  b062: आठ -> एनी आद
  b062: सौ -> पैंच सो रुपे
  b062: रुपये -> पैंच सो रुपे
  b066: आठ -> आथ
  b066: दो -> डो रेजिस्टर

### Listening: pairs where E3 was marked as having a wrong word, by category

- numbers/en: 2
- names/en: 2
- numbers/hinglish: 2
- dates_times/en: 2
- dates_times/hinglish: 1
- places/en: 1
- currency/en: 1
- questions/en: 1

## E3 run 1, bench_deva
### Whisper word errors on results/runs/E3_run1/bench_deva (88 rows)

| type | wrong words | of all words of that type | rate |
|---|---|---|---|
| hinglish | 21 | 145 | 0.14 |
| hindi | 19 | 110 | 0.17 |
| name | 13 | 29 | 0.45 |
| english | 12 | 554 | 0.02 |
| acronym | 9 | 83 | 0.11 |
| number | 9 | 169 | 0.05 |

hinglish examples:
  b061: सक्सेसफुली -> सक्सर्सफुली
  b062: कल -> कर
  b063: ऑर्डर -> ओर्डर
  b064: शेयर -> सेर
  b065: कैशबैक -> क्याश बैक
  b066: कंप्लेंट -> कम्प्लेंट

hindi examples:
  b077: भुगतान -> भुपतान
  b078: आपकी -> आपके
  b078: कल -> कर
  b078: साढे -> सारे
  b079: बताइए -> बताईए
  b080: डिलीवरी -> डिलिवरी

name examples:
  b009: iyer -> here
  b011: anjali -> and daddy ready
  b011: reddy -> and daddy ready
  b015: gurpreet -> gopreet
  b017: koramangala -> kuramana
  b019: guwahati -> buahtai

### Listening: pairs where E3 was marked as having a wrong word, by category

- numbers/en: 2
- names/en: 2
- numbers/hinglish: 2
- dates_times/en: 2
- dates_times/hinglish: 1
- places/en: 1
- currency/en: 1
- questions/en: 1

