# Spec: mapovanie rečníkov, surové metriky, dynamické skóre, bezpečná oprava prepisu

## Diagnóza (report 620752)

Všetky tri symptómy majú jednu príčinu: **mená rečníkov vznikajú iba ako vedľajší
produkt LLM opravy prepisu**. Keď oprava zlyhá, zvyšok pipeline beží nad
anonymnými štítkami, ale behaviorálny a fact-check agent píšu skutočné mená.

| Krok | Čo sa stalo v 620752 |
| --- | --- |
| `correct_transcript` (`src/agents.py:1028`) | Gemini vrátil celý prepis skrátený na 10 627 z 14 086 slov (výstup sa nezmestil do jednej odpovede). |
| Drift guard (`src/agents.py:1936`) | Správne odmietol opravu, `working = transcript` → prepis má len `Speaker A…K`. |
| Agenti | `behavioral_analysis.speakers` = `Erik Tomáš`, `Marián Viskupič`; `facts[].speaker` tiež mená. |
| `score_report` (`src/scoring.py:309`) | Kľúče štatistík `speaker h`, `speaker d` sa nezhodujú s menami, `_resolve_existing` pri `Speaker X` zámerne nerobí fuzzy match → `words = 0` → `_DEFAULT_WORDS = 1000`. |
| `apply_deterministic_moderator_metrics` | Počíta podiely z toho istého prepisu → `Speaker H 46,7 %`, `Speaker D 33,9 %`. |
| Úloha 5 | Citácie v reporte pochádzajú zo **surového ASR** (oprava bola zahodená): „ľudi“, „kpointe“, „napl nela“, „som rá“. |

Porovnanie: `620752.empty-facts.bak.json` (oprava vtedy prešla) má `Erik Tomáš 6278`,
`Marián Viskupič 4537` slov. Rovnaký vstup, iný výsledok → nedeterministická
závislosť skóre od jedného LLM volania.

Ďalšie zistenia pri oprave textu (diff raw vs corrected):

- 617000: korektor **zmazal** „Ako to vidíte vy,“ a „služby tu“; „Armádsky“ prepísal
  na „Pán Majerský“ (odhad mena, nie oprava preklepu). Globálny guard ±10 % slov
  lokálne zásahy nezachytí.
- 619465: 229 úprav, väčšina korektná („prezistentom“ → „prezidentom“), ale
  neexistuje žiadny záznam, čo sa zmenilo.
- `validate_facts` overuje, že citácia **niekde** v prepise je (fuzzy 0,6), ale
  neoveruje, **kto** ju povedal, ani či nezávisí od opraveného slova.
- `transcript_speaker_stats` počíta aj marker `(cez seba)` ako 2 slová.

Záver k Úlohe 5: máš pravdu. V 620752 nebola citácia opravená vôbec; v iných
behoch opravená bola, ale bez auditu a s reálnym rizikom zmeny významu.

## Ciele

1. Mapovanie `Speaker X → meno/rola` je samostatný, deterministicky aplikovaný
   krok s dôkazmi a stavom. Nezávisí od opravy textu.
2. Scoreboard nesie presné surové metriky (slová, podiel, prehovory) a celý
   „lievik“ tvrdení.
3. Manipulácie a fauly sa normalizujú skutočným počtom slov, vyhýbanie sa
   otázkam počtom podstatných priamych otázok. Žiadne fiktívne 1000.
   Keď dáta chýbajú, skóre disciplíny je `None` a víťaz sa nevyhlási.
4. Oprava prepisu je auditovateľný zoznam drobných úprav; obvinenie
   (False/Misleading) nikdy nesmie stáť na slove, ktoré zmenil korektor.

## Mimo rozsahu

- Zmena diarizačného modelu, ASR confidence skóre (Whisper logprob) — možné neskôr.
- Zmena váh disciplín.

## Nový tok pipeline

```
raw.txt (Speaker A..K)                        ← nemenné, zdroj pravdy
  │
  ├─ 1. map_speakers()        → SpeakerMap (+ debate_start)      [nový src/speakers.py]
  ├─ 2. apply_speaker_map()   → named.txt   (čisto v kóde)
  ├─ 3. propose_corrections() → edits[] po chunkoch (LLM)        [nový src/correction.py]
  ├─ 4. apply_corrections()   → corrected.txt + corrections.json (guardy v kóde)
  │
  ├─ analýza (briefing, extract, question audit, check, critic) nad corrected.txt
  ├─ validate_report(corrected, raw, alignment, speaker_map)
  └─ score_report(report, corrected, speaker_map, claim_funnel)
```

`correct_transcript` (celý prepis ako jeden LLM výstup) sa odstráni.

## Úloha 1: Speaker mapping (`src/speakers.py`)

### Dátový model (v `src/agents.py` k ostatným modelom)

```python
class SpeakerRole(str, Enum):
    MODERATOR = "moderator"
    GUEST = "guest"
    CLIP = "clip"        # zostrih, záznam, anketa
    UNKNOWN = "unknown"

class SpeakerMapEntry(BaseModel):
    label: str                    # "Speaker H"
    name: str                     # "Erik Tomáš" | "Moderátor" | "Záznam"
    role: SpeakerRole
    confidence: float             # 0-1
    evidence: list[str] = []      # "[03:11] Ja môžem za stranu HLAS"

class SpeakerMap(BaseModel):
    status: Literal["ok", "partial", "failed"]
    source: Literal["cli", "llm", "heuristic"]
    debate_start: str | None      # timestamp moderátorovho privítania hostí
    entries: list[SpeakerMapEntry]
    notes: list[str] = []
```

Viac štítkov smie mapovať na to isté meno (diarizácia rozdelí jedného človeka).

### Algoritmus

1. **Roster** (priorita): CLI `--guests "Erik Tomáš;Marián Viskupič"` a voliteľne
   `--moderator "…"` → inak mená z intra (LLM v kroku 3). Mapovač smie použiť
   iba mená z rostra.
2. **Štatistiky per štítok** (kód): slová, prehovory, podiel otázok (`?`),
   prvý výskyt.
3. **Heuristiky (kód)**:
   - `CLIP`: štítok < 1,5 % slov **alebo** všetky riadky pred `debate_start`.
   - `MODERATOR`: kandidát s najvyšším podielom otázok medzi štítkami > 10 % slov,
     ktorý hovorí ako prvý po intre.
   - `GUEST` kandidáti: zvyšné štítky > 10 % slov.
4. **LLM priradenie mien** (jedno malé volanie, nie celý prepis): pre každého
   kandidáta pošli prvých ~8 riadkov, riadky so sebaidentifikáciou
   (`ja ako minister`, `za stranu X`, `my v SaS`) a riadky moderátora tesne pred
   jeho prehovorom (oslovenia `pán X`). Pridaj stranu/funkciu z briefingu, ak
   existuje. Výstup: `SpeakerMap` s `evidence` citáciami.
5. **Validácia výstupu (kód)**:
   - každé meno je z rostra; každý hosť z rostra má aspoň jeden štítok;
   - žiadne meno hosťa nie je priradené dvom hlavným (> 10 %) štítkom, ak tie
     štítky hovoria striedavo (dialóg = dvaja ľudia);
   - každá `evidence` citácia sa nájde v riadku toho štítka (alebo v riadku
     moderátora tesne pred ním) cez `quote_grounded`;
   - `confidence < 0.7` alebo zlyhaná validácia → `status = "partial"`.
6. **Fallback**: LLM zlyhá → heuristika (moderátor + dvaja najväčší hostia,
   poradie mien podľa poradia predstavenia v intre). `status = "partial"`,
   `source = "heuristic"`.

### `apply_speaker_map(transcript, smap) -> str`

Čistá funkcia: nahradí prefix riadku menom z mapy. Text sa nemení ani o znak.
Test: `strip_labels(raw) == strip_labels(named)`.

### Rozdelenie zmiešaných riadkov

Prípad z 620752 [01:58]: moderátorov riadok obsahuje aj „Ďakujem za pozvanie
a všetkým prajem peknú nedeľu.“ (Tomáš). Rieši to krok 3 ako edit typu
`split_turn` (viď Úloha 5), nie mapovač.

### Spotrebitelia mapy

- Agenti dostávajú `corrected.txt` s menami → ich `speaker` polia sú konzistentné.
- `canonical_speaker(name, smap)` (nová funkcia, nahradí `_resolve_existing`):
  presná zhoda → zhoda priezviska v rámci rostra → `None`. Aplikuje sa na
  `facts[].speaker` a `behavioral_analysis.speakers[].speaker` pred skórovaním;
  nenamapované → poznámka v `method_notes`, nie tichý drop.
- `report.speaker_map` sa uloží do JSON.

## Úloha 2: Surové metriky

### Zmeny v `transcript_speaker_stats`

- Vstup: prepis + `SpeakerMap`; kľúč = kanonické meno, nie `speaker x`.
- Slová: `len(_strip_markers(text).split())`, kde `_strip_markers` odstráni
  `(cez seba)`, `(hovorenie cez seba)` a podobné anotácie v zátvorkách.
- Ignoruj riadky pred `debate_start` a riadky rolí `CLIP`.
- Nové polia v `SpeakerStats`:
  - `turns` — počet prehovorov (zmena rečníka; už existuje),
  - `substantive_turns` — prehovory s ≥ 15 slovami (bez krátkych vstupov
    „Ďakujem.“, „Áno.“),
  - `interjections` — prehovory s < 5 slovami,
  - `questions_received` — počet otázok moderátora adresovaných tomuto
    rečníkovi (kandidáti z kroku „Question audit“, všetky druhy),
  - `interruptions_caused` (už existuje).

### Nové polia `SpeakerScore`

```python
words: int                          # skutočný počet slov hosťa
word_share_percent: float           # podiel medzi hosťami (súčet hostí = 100)
transcript_share_percent: float     # podiel z celej debaty vrátane moderátora
turns: int
substantive_turns: int
interjections: int
questions_received: int
challenging_questions: int          # podstatné priame otázky (viď Responsiveness)
questions_dodged: int
questions_partial: int
interruptions_caused: int
```

Zmena sémantiky: `word_share_percent` bol doteraz podiel z celku vrátane
moderátora. Nové pole `transcript_share_percent` drží starý význam. `main.py`
a FB prompt zobrazujú `word_share_percent`.

### `moderator_audit.equal_time_distribution`

Počíta sa z tých istých štatistík (mená, nie štítky). Všetky `CLIP` štítky sa
zlúčia do jedného riadku `Záznamy/zostrihy`. Pridá sa `turns` do `TimeShare`.

## Úloha 3: Dynamické skóre

### Odstrániť

- `_DEFAULT_WORDS` a vetvu „assumed equal speaking time“.

### Pravidlá

```python
penalty_points = W_MANIP * n_manip + W_FALLACY * n_fallacy
normalization_words = max(words, _MIN_WORDS_FLOOR)
rate_per_1000 = penalty_points / normalization_words * 1000
score = max(0, 100 - rate_per_1000)
```

Platí pre `manipulation`. `responsiveness` má vlastný vzorec (nižšie),
nezávislý od počtu slov.

- `words == 0` pre hosťa (mapovanie zlyhalo): `manipulation` má
  `score = None`, detail „chýbajú štatistiky reči“.
  Celkové skóre sa renormalizuje cez dostupné disciplíny (existujúca logika).
- `_MIN_WORDS_FLOOR = 500` ostáva iba ako poistka proti degenerovanému prípadu;
  keď sa použije, `normalization_words != words` je vidno v JSON a pridá sa
  poznámka. Pri správnom mapovaní (hostia majú tisíce slov) sa neuplatní.
- Gate: `DebateVerdict.scoring_status = "ok" | "degraded"`. `degraded`, ak
  `speaker_map.status != "ok"` alebo niektorý hosť má `words == 0`.
  V stave `degraded` sa `winner` ani `discipline_winners` pre behaviorálne
  disciplíny nevyhlasujú.

### Responsiveness: vyhnutia / podstatné priame otázky

Vyhýbanie sa otázke má zmysel iba vzhľadom na položené otázky, a iba na
otázky, ktoré naozaj vyžadujú odpoveď. Normalizácia na 1000 slov sa ruší.

#### Question audit (nový krok vo fáze A, `src/questions.py`)

1. **Kandidáti (kód).** Každá veta končiaca `?` v prehovore moderátora.
   Adresát = oslovený rečník (`pán/pani X` v tom istom prehovore), inak
   nasledujúci hosť, ktorý prehovorí. Odpoveď = všetky prehovory adresáta až
   po ďalšiu otázku moderátora. Každý kandidát má `id`, `timestamp`, text
   otázky, adresáta a `answer_span` (riadky odpovede).
   Otázky medzi hosťami sa nepočítajú: v debate sú prevažne rečnícke a
   nikto od oponenta odpoveď nevyžaduje.
2. **Klasifikácia (LLM, jedno volanie nad zoznamom kandidátov, nie celým
   prepisom).** Pre každý kandidát:

   | `kind` | Význam | Počíta sa do menovateľa |
   | --- | --- | --- |
   | `challenging` | priama otázka k podstate: konkrétny postoj (áno/nie), číslo, zodpovednosť, termín, konfrontácia s rozporom alebo faktom | áno |
   | `open` | mäkká otvorená výzva („Ako to vidíte?“, „Čo na to poviete?“) | nie |
   | `procedural` | réžia debaty („Dokončíte?“, „Môžeme ísť ďalej?“) | nie |
   | `rhetorical` | otázka bez očakávanej odpovede | nie |

   Pre `challenging` aj `outcome`:

   | `outcome` | Význam | Váha v čitateli |
   | --- | --- | --- |
   | `answered` | odpoveď priamo reaguje na jadro otázky (aj keď nepríjemne alebo s výhradou) | 0 |
   | `partial` | odpovie na časť, jadro (číslo, áno/nie, zodpovednosť) vynechá | 0,5 |
   | `dodged` | odbočí na inú tému, protiútok, „to sa pýtajte inde“ bez vecnej odpovede | 1 |
   | `interrupted` | adresát nedostal priestor (< 15 slov odpovede pred ďalším vstupom) | vylúčená z menovateľa |

   Každé `partial`/`dodged` musí mať `evidence`: citáciu otázky a citáciu
   z `answer_span`, ktorá ukazuje odbočenie. Rozlišovať: tvrdá alebo nepohodlná
   odpoveď je `answered`, nie `dodged`.
3. **Validácia (kód).** Citácia otázky musí ležať v riadku moderátora s daným
   `timestamp`, citácia odpovede v `answer_span` adresáta. Neplatný dôkaz →
   `outcome` sa zmení na `answered` (v pochybnosti v prospech rečníka)
   a pridá sa poznámka. Odpoveď v riadku `(cez seba)` → `interrupted`.
4. `behavioral_analysis.speakers[].question_dodging` sa už negeneruje voľným
   textom; renderuje sa z auditu (`[MM:SS] otázka → dôvod`), aby existoval
   jeden zdroj pravdy.

#### Vzorec

```python
n = challenging - interrupted          # podstatné otázky, kde mal priestor
dodge_mass = dodged + 0.5 * partial
dodge_rate = (dodge_mass + _DODGE_PRIOR_MASS) / (n + _DODGE_PRIOR_QUESTIONS)
score = 100 * (1 - dodge_rate)          # None, ak n == 0
```

`_DODGE_PRIOR_QUESTIONS = 2`, `_DODGE_PRIOR_MASS = 0.3` (rovnaký princíp ako
truthfulness: 1 vyhnutie z 1 otázky nie je 0 bodov, 1 odpoveď z 1 nie je 100).

Príklad: 8 podstatných otázok, 1 prerušená, 2 vyhnutia, 1 čiastočná →
`n = 7`, `dodge_mass = 2,5`, `rate = 2,8 / 9 = 0,311`, `score = 68,9`.

`DisciplineResult` pre responsiveness: `inputs = {"challenging": 8,
"interrupted": 1, "dodged": 2, "partial": 1}`, `weights = {"dodged": 1.0,
"partial": 0.5}`, `rate = 0.311`, `penalty_points`/`normalization_words` sú
`None`. `evidence` = zoznam vyhnutí s časom a citáciami.

Odznak „Nx vyhýbanie sa otázke“ sa počíta z `questions_dodged` (dnes
z dĺžky voľného zoznamu `question_dodging`).

Moderátorský audit dostane bonus metriku: počet podstatných otázok na
každého hosťa. Výrazný nepomer (napr. 12 vs 4) ide do `moderator_audit.findings`.

### Transparentnosť disciplíny

`DisciplineResult` dostane štruktúrované polia namiesto len `detail` stringu:

```python
inputs: dict[str, int]          # {"manipulation": 3, "fallacies": 2}
weights: dict[str, float]       # {"manipulation": 8.0, "fallacies": 6.0}
penalty_points: float | None
normalization_words: int | None
rate_per_1000: float | None     # manipulation
rate: float | None              # responsiveness (podiel 0-1)
```

Z týchto polí musí ísť skóre prepočítať ručne.

## Úloha 4: Lievik tvrdení (claim funnel)

Dnes scoreboard vie iba o faktoch, ktoré prežili celý pipeline. Počty z
extrakcie a selekcie sú len v textových `critic_notes`.

### Model

```python
class ClaimFunnel(BaseModel):
    speaker: str
    extracted: int              # všetky z ExtractedClaims
    non_empirical: int          # opinion/prediction/definitional (selection.py)
    dropped_by_selection: int   # pod prahom consequence / cap
    selected_for_check: int
    removed_ungrounded: int     # validate_facts zahodil (citácia nenájdená)
    final_facts: int            # v report.facts
    checked: int                # True/False/Misleading po judge
    unverified: int
    contested: int
```

Invariant (test): `extracted = non_empirical + dropped_by_selection + selected_for_check`
a `selected_for_check = removed_ungrounded + final_facts`.

### Kde sa počíta

- `run_analysis`: po extrakcii a `select_claims` / `select_top_claims` zostav
  počty per kanonický rečník; `validate_facts` vráti aj zoznam odstránených
  faktov (nielen poznámky). Výsledok `report.claim_funnel: list[ClaimFunnel]`.
- `select_claims` a `select_top_claims` vrátia okrem `kept` aj štruktúrované
  `excluded` a `dropped` zoznamy (dnes len text).
- `score_report`: doplní `checked/unverified/contested` z finálnych faktov
  (po judge) a skopíruje do `SpeakerScore`:
  `claims_extracted`, `claims_selected`, `claims_checked` (= `checked_claims`).
- Truthfulness `detail`: „overené 11 z 15 vybraných (extrahovaných 18)“.

## Úloha 5: Bezpečná oprava prepisu (`src/correction.py`)

### Princíp

Raw ASR je nemenný zdroj pravdy. LLM nevracia prepis, vracia **zoznam úprav**.
Kód rozhoduje, ktoré úpravy prejdú. Každá úprava je zalogovaná.

### Edit model

```python
class EditType(str, Enum):
    SPELLING = "spelling"        # preklep, diakritika: „ľudi“ → „ľudí“
    WORD_BOUNDARY = "word_boundary"  # „kpointe“ → „k pointe“, „napl nela“ → „naplnila“
    PROPER_NOUN = "proper_noun"  # „HLas“ → „HLAS“, iba mená z rostra/briefingu
    PUNCTUATION = "punctuation"
    SPLIT_TURN = "split_turn"    # rozdelenie riadku medzi dvoch rečníkov

class TranscriptEdit(BaseModel):
    line_no: int
    type: EditType
    before: str                  # presný úsek z raw riadku
    after: str
    split_at_word: int | None = None
    new_speaker: str | None = None
    reason: str = ""
```

### Volanie LLM

- Chunky po ~40 riadkoch s 5 riadkami kontextu z oboch strán (kontext sa
  needituje). Rieši truncation, ktorá zabila 620752.
- Prompt: „Vráť iba úpravy. Neopravuj gramatiku ani štýl hovorenej reči,
  neparafrázuj, nedopĺňaj chýbajúce slová, nemaž opakovania ani výplňové slová.
  Ak si nie si istý, úpravu nenavrhuj.“ Povolené typy iba podľa `EditType`.

### Guardy v kóde (úprava sa zamietne, ak)

1. `before` sa nenachádza presne v danom raw riadku.
2. Čísla: množina číselných tokenov (`\d+`, slovné číslovky z malého slovníka)
   v `before` a `after` sa líši. Formát „4,9“ ↔ „4,9 %“ áno, hodnota nie.
3. Negácia: zmena prítomnosti `nie`, `ne-` prefixu slovesa, `nikdy`, `žiadn*`,
   `nič`, `ani`.
4. Podobnosť: `difflib` ratio `before`/`after` na znakoch < 0,6 pre
   `SPELLING`/`WORD_BOUNDARY` (zachytí „Armádsky“ → „Pán Majerský“).
5. Počet slov: rozdiel > 1 pre `SPELLING`/`WORD_BOUNDARY`/`PROPER_NOUN`.
6. `PROPER_NOUN.after` nie je z rostra, briefingu alebo glosára strán/inštitúcií.
7. `SPLIT_TURN`: spojenie častí sa nerovná pôvodnému textu, alebo
   `new_speaker` nie je v `SpeakerMap`.

Výstup: `corrected.txt`, `{ep}.corrections.json` (aplikované aj zamietnuté
s dôvodom) a alignment `corrected_line_no → raw_line_no`.
`report.transcript_quality = {applied, rejected_by_rule: {...}, log_path}`.

### Ochrana pred nespravodlivým obvinením

Nové v `src/validation.py`:

- `quote_attributed(quote, speaker, transcript) -> bool`: citácia musí ležať v
  riadku daného rečníka. Riadky s `(cez seba)` sa pre False/Misleading
  nepočítajú ako spoľahlivé priradenie.
- `quote_raw_support(quote, corrected, raw, edits) -> RawSupport`: nájde
  citáciu v corrected, cez alignment ju premietne do raw a vráti
  `similarity` a zoznam editov, ktoré úsek zasahujú.

Pravidlá pre fakty s verdiktom `False` / `Misleading`:

| Podmienka | Akcia |
| --- | --- |
| citácia nie je priradená danému rečníkovi | → `Unverified`, poznámka „nepotvrdené priradenie rečníka“ |
| raw similarity < 0,85 (dnes 0,6 pre všetko) | → `Unverified`, „citácia sa nezhoduje s pôvodným prepisom“ |
| úsek zasahuje edit typu `PROPER_NOUN` alebo `SPLIT_TURN` | → `Unverified`, „verdikt závisí od opravy prepisu“ |
| číslo alebo negácia v `claim` sa nenachádza v raw úseku | → `Unverified` |

Pre `True` / `Unverified` / `Contested` ostáva súčasný prah 0,6.

`VerifiedFact` dostane polia `quote_raw: str`, `timestamp: str`,
`transcript_edits: list[int]` (indexy do corrections log), aby čitateľ videl
pôvodný ASR text vedľa opraveného.

## Výstupný JSON (výťah)

```json
{
  "speaker_map": {"status": "ok", "source": "llm", "debate_start": "01:58",
    "entries": [{"label": "Speaker H", "name": "Erik Tomáš", "role": "guest",
                 "confidence": 0.93, "evidence": ["[03:11] Ja môžem za stranu HLAS"]}]},
  "transcript_quality": {"applied": 212, "rejected_by_rule": {"number": 3, "similarity": 5}},
  "claim_funnel": [{"speaker": "Erik Tomáš", "extracted": 18, "non_empirical": 2,
                    "dropped_by_selection": 1, "selected_for_check": 15,
                    "removed_ungrounded": 0, "final_facts": 15, "checked": 11,
                    "unverified": 4, "contested": 0}],
  "verdict": {"scoring_status": "ok",
    "scoreboard": [{"speaker": "Erik Tomáš", "words": 6278, "word_share_percent": 58.0,
      "transcript_share_percent": 44.6, "turns": 61, "substantive_turns": 38,
      "interjections": 14, "questions_received": 19, "challenging_questions": 8,
      "questions_dodged": 2, "questions_partial": 1, "interruptions_caused": 7,
      "claims_extracted": 18, "claims_selected": 15, "checked_claims": 11,
      "disciplines": [
        {"discipline": "manipulation", "inputs": {"manipulation": 3, "fallacies": 2},
         "weights": {"manipulation": 8.0, "fallacies": 6.0}, "penalty_points": 36.0,
         "normalization_words": 6278, "rate_per_1000": 5.7, "score": 94.3},
        {"discipline": "responsiveness",
         "inputs": {"challenging": 8, "interrupted": 1, "dodged": 2, "partial": 1},
         "weights": {"dodged": 1.0, "partial": 0.5}, "rate": 0.311, "score": 68.9}]}]},
  "question_audit": [{"id": 7, "timestamp": "12:40", "addressee": "Erik Tomáš",
    "question": "Zvýšite teda DPH, áno alebo nie?", "kind": "challenging",
    "outcome": "dodged", "evidence": ["[12:44] Pozrime sa, čo urobila vaša vláda…"]}]
}
```

## Dotknuté súbory

| Súbor | Zmena |
| --- | --- |
| `src/speakers.py` | nový: `map_speakers`, `apply_speaker_map`, `canonical_speaker` |
| `src/correction.py` | nový: `propose_corrections`, `apply_corrections`, guardy |
| `src/agents.py` | modely; `run_analysis` nový tok; odstrániť `correct_transcript`; funnel |
| `src/questions.py` | nový: `extract_question_candidates`, `classify_questions` (LLM), `validate_question_audit` |
| `src/selection.py` | štruktúrované `excluded` / `dropped` |
| `src/scoring.py` | stats podľa mapy, nové polia, bez `_DEFAULT_WORDS`, gate, štruktúrované disciplíny |
| `src/validation.py` | `quote_attributed`, `quote_raw_support`, prísnejšie pravidlá pre obvinenia; `validate_facts` vracia odstránené fakty |
| `main.py` | `--guests`, `--moderator`; uloženie `corrections.json`; výpis turns/claims |
| FB prompt | zobraziť slová/prehovory; pri `degraded` žiadny víťaz |

## Testy

- `tests/test_speakers.py`: fixture z prvých 30 riadkov 620752 → H = Tomáš,
  D = Viskupič, A = Moderátor, B/C/E/F/G = clip; `apply_speaker_map` nemení text;
  zlé meno mimo rostra → `partial`.
- `tests/test_correction.py`: zamietnutie čísla (40 → 140), negácie
  („podporili“ → „nepodporili“), nízkej podobnosti („Armádsky“ → „Pán Majerský“),
  mazania slov; prijatie „ľudi“ → „ľudí“, „kpointe“ → „k pointe“; `split_turn`
  zachová text.
- `tests/test_scoring.py`: hosť s 0 slovami → `None` disciplíny + `degraded`;
  `rate_per_1000` sa dá prepočítať z polí; `word_share_percent` hostí = 100;
  `(cez seba)` sa nepočíta do slov.
- `tests/test_validation.py`: False fakt s citáciou z riadku iného rečníka →
  `Unverified`; citácia závislá od `PROPER_NOUN` editu → `Unverified`.
- Funnel invarianty.
- `tests/test_questions.py`: extrakcia kandidátov (adresát z oslovenia aj
  z nasledujúceho hosťa, `answer_span` končí pri ďalšej otázke moderátora);
  `open`/`procedural`/`rhetorical` nejdú do menovateľa; `interrupted` sa
  vylúči; neplatná citácia odpovede → `answered`; vzorec na príklade
  8/1/2/1 → 68,9; `n == 0` → `None`.

## Overenie na dátach

Prebeh 620752 z existujúceho `data/transcripts/620752.txt` (bez ASR):
`words` hostí v pásme ±5 % od `620752.empty-facts.bak.json` (6278 / 4537),
`equal_time_distribution` s menami, `scoring_status = ok`,
`corrections.json` bez zamietnutí typu číslo/negácia na manuálnej kontrole 20 náhodných editov.
Question audit: ručná kontrola všetkých `dodged` v 620752 (každé má platnú
citáciu otázky aj odpovede, žiadne „tvrdá odpoveď = vyhnutie“).
