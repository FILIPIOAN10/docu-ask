# DocuAsk — jurnal de construcție

Serviciu RAG peste documente: FastAPI + LangChain + Postgres/pgvector. Construit pas cu pas după ghidul
`02-Proiect-1-DocuAsk-RAG.pdf` (44 de pași, 12 faze, A–L). Jurnalul e pentru mine, la interviu: pentru fiecare fază
am scris **ce am făcut, de ce, ce am verificat (cu rezultate reale)** și **cum o spun**.

Regula de onestitate: spun „construit și testat local”, nu „în producție”. Ce nu a rulat încă e marcat explicit.

---

## Faza A — Fundația (pașii 1–4) · 2026-10-17

### Ce am făcut
- **Pas 1 — unelte.** `uv` 0.12.18, `git` 2.55, Docker 28.3.2 (motorul pornit; Testcontainers și compose îl cer).
- **Pas 2 — repo și dependențe.** `git init -b main`, pachet `src/docuask`, `pyproject.toml`, `uv sync` →
  `.venv` cu Python 3.12.12 și `uv.lock` (84 de pachete rezolvate).
- **Pas 3 — `config.py`.** Toate setările într-o clasă `Settings` (pydantic-settings), citite din mediu / `.env`.
- **Pas 4 — fișiere de igienă.** `.env.example` (documentează setările), `.gitignore`, `.dockerignore`.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| `uv` + `uv.lock` commitat | Mediu reproductibil: aceleași versiuni pe laptop, în Docker și în CI |
| Layout `src/` + hatchling | Importul `docuask` merge doar din pachetul instalat, nu din folderul curent (prinde greșeli de import) |
| Grup `dev` separat | Teste, Testcontainers și ruff nu intră în imaginea de producție |
| Config din variabile de mediu (12-factor) | Aceeași aplicație rulează pe orice mașină; secretele nu stau în cod |
| Validare la pornire (`Literal`, tipuri) | **Fail fast**: o setare greșită oprește procesul imediat, nu la prima cerere |
| `SecretStr` pentru chei | Cheia nu apare în print sau log (`**********`) |
| `api_keys` = listă cu virgulă | Rotație fără oprire: adaug cheia nouă, mut clienții, scot cheia veche |
| `llm_provider = "fake"` implicit | Testele, CI-ul și prima demonstrație nu depind de cheie, rețea sau bani |
| `.env` în `.gitignore`, `.env.example` în git | Setările sunt documentate, secretele nu ajung în istoric |

### Ce am verificat (rezultate reale)
- `uv run python -c "import fastapi, langchain, sqlalchemy; print('ok')"` → `ok`
- `get_settings().llm_provider` → `fake`
- `print(s.api_keys, s.keys)` → `********** ['dev-key']`
- `LLM_PROVIDER=banana` → `ValidationError: Input should be 'fake', 'openai' or 'azure'` (fail fast, confirmat)
- `git status` nu arată `.venv/` sau `.env`

### Capcane
- **PowerShell 5.1 scrie UTF-16.** `echo 3.12 > .python-version` ar produce un fișier pe care `uv` îl poate citi greșit;
  am folosit `Set-Content -Encoding ascii`.
- **Folder în loc de fișier.** Am creat din greșeală un *folder* numit `pyproject.toml`. În `ls`, `Mode` începea cu `d`
  (la fișier e `-a----`). Remediu: `Remove-Item pyproject.toml`, apoi `code pyproject.toml` și salvare.

### Cum o spun la interviu
- „Configurația vine din variabile de mediu și e validată la pornire cu pydantic-settings: o valoare greșită oprește
  procesul imediat, nu după ce un utilizator primește o eroare.”
- „Secretele sunt `SecretStr`, deci nu ajung accidental în loguri. Cheile API sunt o listă, ca să pot face rotație fără
  oprire.”
- „Dependențele sunt blocate într-un `uv.lock` commitat, deci CI și imaginea Docker instalează exact aceleași versiuni.”
- „Am un furnizor `fake` ca valoare implicită, ca să pot testa logica mea fără cheie și fără rețea.”

### De ținut minte pentru fazele următoare
- `embedding_dim = 1536` trebuie să fie egal cu lățimea coloanei de vectori din Postgres (Faza B). Schimbarea modelului
  de embedding înseamnă migrare nouă și reindexare.
- `max_distance = 0.95` e provizoriu; îl calibrez pe cifre reale la Pasul 41.

---

## Faza B — Baza de date (pașii 5–8) · 2026-10-17

### Ce am făcut
- **Pas 5 — Postgres cu pgvector.** Container `docuask-db`, imaginea `pgvector/pgvector:pg16` (Postgres 16.15,
  pgvector 0.8.7), port `5434 → 5432`.
- **Pas 6 — `models.py`.** Tabelele `documents` și `chunks` ca modele SQLAlchemy 2.0 (`Mapped`, `mapped_column`).
- **Pas 7 — `db.py`.** Engine cu pool, factory de sesiuni, dependența `get_session()` pentru FastAPI.
- **Pas 8 — Alembic.** `alembic.ini`, `migrations/env.py`, migrarea `0001_init.py`, aplicată cu `alembic upgrade head`.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Postgres + pgvector, nu o bază vectorială separată | Relațional, full-text și vectori într-un singur sistem, cu tranzacții |
| Schema din migrare scrisă de mână, nu `create_all` / autogenerate | Autogenerarea nu știe de extensia `vector`, de coloana generată și de indexul HNSW |
| Index HNSW cu `vector_cosine_ops` (`m=16`, `ef_construction=64`) | Căutare rapidă după vectori; e aproximativ (poate rata rar vecinul perfect), în schimbul vitezei |
| Index GIN pe `tsv` | „Index de carte” pentru căutarea după cuvinte |
| `tsv` = coloană generată și stocată | Postgres o recalculează din `section + content`; nu poate ieși din sincron cu textul |
| `documents.id` UUID, `name` unic, `sha256` | UUID negenerabil din afară; `UNIQUE(name)` decide câștigătorul la încărcări simultane; hash-ul spune dacă fișierul s-a schimbat |
| `ON DELETE CASCADE` pe `chunks.document_id` | Ștergi documentul, dispar bucățile; nu rămân „fantome” în căutări |
| Sesiuni scurte, `pool_pre_ping`, `expire_on_commit=False` | Nu blochez o conexiune cât aștept modelul; repornirea bazei nu strică prima cerere; obiectele rămân citibile după commit |
| URL-ul bazei din `DATABASE_URL`, nu din `alembic.ini` | Aceleași migrări rulează pe laptop, în CI și în cluster |

### Ce am verificat (rezultate reale)
- `pg_isready` → `accepting connections`
- `sorted(Base.metadata.tables)` → `['chunks', 'documents']`
- `select 1` prin `get_session_factory()` → `1` (dovedește containerul, portul 5434 și `config.py` împreună)
- `alembic upgrade head` → `Running upgrade  -> 0001, ...`; `alembic current` → `0001 (head)`
- Tabele în bază: `alembic_version`, `chunks`, `documents`; extensia `vector 0.8.7`
- `\d chunks`: `embedding vector(1536)`, `tsv` generat `stored`, `ix_chunks_embedding hnsw (embedding vector_cosine_ops)
  WITH (m='16', ef_construction='64')`, `ix_chunks_tsv gin (tsv)`, cheia străină cu `ON DELETE CASCADE`

Neverificat încă: viteza indexului HNSW (nu am date pe care să o măsor); vine la evaluare, Faza I.

### Capcane
- **Port 5434 partajat.** Un container vechi, `ticketbox-postgres`, folosește tot 5434; nu pornesc ambele odată.
- **Docker Desktop trebuie pornit.** Motorul a căzut între doi pași; `docker version` trebuie să arate și „Server”.
- **Datele stau în container.** `docker rm` le șterge; migrarea se reaplică, iar documentele se reîncarcă la ingestie.
- **`vector_cosine_ops` ↔ `<=>`.** Operatorul din interogare trebuie să fie cel al clasei de index (cosinus), altfel
  Postgres ignoră indexul și face căutare exactă (Pasul 15).
- **Sursa de adevăr e migrarea.** `models.py` descrie tabelele pentru cod și diferă în detalii (`int` în model,
  `BigInteger` identity în migrare); schema reală vine din migrare.

### Cum o spun la interviu
- „Am ales Postgres cu pgvector ca să am date relaționale, full-text și vectori într-un sistem cu tranzacții, nu o
  bază vectorială separată de sincronizat.”
- „Schema e versionată cu Alembic și scrisă manual, fiindcă autogenerarea nu cunoaște extensia vector, coloana
  generată și indexul HNSW. Același set de migrări rulează local, în CI și în cluster.”
- „Căutarea după vectori folosește un index HNSW, care e aproximativ: schimb puțină acuratețe pe viteză. Clasa de
  operatori a indexului trebuie să fie aceeași cu operatorul din interogare, altfel indexul nu e folosit.”
- „Dimensiunea embedding-ului (1536) apare în config, model și migrare; dacă schimb modelul, fac migrare nouă și
  reindexez.”
- „Sesiunile de bază de date sunt scurte: nu țin o conexiune din pool cât aștept modelul de limbaj.”

---

## Faza C — Furnizorii de model, reali și falși (pașii 9–10) · 2026-10-17

### Ce am făcut
- **Pas 9 — `fake.py`.** `FakeEmbeddings` (hashed bag of words, 1536 de dimensiuni) și `FakeChatModel` (extractiv: copiază
  frazele cele mai relevante din pasajul cel mai potrivit și îl citează). Fără cheie, fără rețea.
- **Pas 10 — `providers.py`.** Singurul loc care construiește clienții de model: `fake`, `openai` sau `azure`, alese din
  `LLM_PROVIDER`. `get_embeddings()` și `get_chat_model()` cu `lru_cache`.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Un model și un embedding **false**, deterministe | Testele, CI-ul și demo-ul nu depind de cheie, rețea sau bani; testez logica mea, nu modelul furnizorului |
| Hash cu `blake2b`, nu `hash()` din Python | `hash()` dă alte valori la fiecare rulare; testele au nevoie de aceiași vectori mereu |
| Semn ±1 pe fiecare „coș” al vectorului | Răspândește coliziunile în jurul lui zero; apoi normalizez la lungime 1, deci cosinus = produs scalar |
| `FakeChatModel` respectă același contract ca unul real (`with_structured_output` → `raw/parsed/parsing_error`) | Restul codului nu știe că modelul e fals |
| `providers.py` = singurul loc cu clienții | Schimb furnizorul dintr-o variabilă de mediu, nu din cod |
| `_need(...)` ridică `ProviderConfigError` la construire | Fail fast: lipsa unei chei oprește procesul la pornire, nu la prima cerere |
| Azure: numele **deployment-ului**, nu al modelului | Așa funcționează Azure OpenAI (deployment-ul îl creez eu în portal) |
| `check_embedding_ctx_length=False` | Altfel biblioteca trimite liste de token-id și descarcă un vocabular de pe internet; trimit text simplu |
| Fără `temperature` | Unele modele noi o resping; stabilitatea vine din schema strictă de ieșire și din prompt |
| `timeout` și `max_retries` pe fiecare client | Nu aștept la nesfârșit un API extern; reîncerc puțin |

### Ce am verificat (rezultate reale)
- Embedding: `len=1536`; similaritate între „sick leave days” și „how many sick leave days” = **0.87**; cu „zebra giraffe
  safari” = **0.0**
- Model fals: răspuns cu citare (`... after 3 days. [1]`) pe întrebarea acoperită; `The documents do not cover this.`
  pe întrebarea fără legătură
- `get_embeddings().embed_query('sick leave')` → 1536 de numere
- `LLM_PROVIDER=openai` fără cheie → `ProviderConfigError: OPENAI_API_KEY is required when LLM_PROVIDER is set to this
  provider` (fail fast, confirmat)
- `ruff check` → `All checks passed!`; `ruff format --check` → fără diferențe

**Neverificat:** niciun apel către OpenAI sau Azure real (nu am rulat codul cu o cheie reală); clienții reali sunt
doar construiți corect. `with_structured_output` al modelului fals nu poate fi testat până la schema din Pasul 17.

### Capcane
- **Modelul fals e prost intenționat.** Întrebarea „What is the capital of France?” îl poate păcăli (găsește „capital
  cities” într-un document de cheltuieli). Nu e un bug de ascuns: e motivul pentru care există trei filtre „nu știu” și
  evaluare cu modele reale.
- **Embedding-ul fals e lexical.** „holiday” și „leave” nu se potrivesc; un model real ar face-o. Diferența o măsor la
  evaluare (Faza I), cu embedding real.
- **Formatul promptului e un contract ascuns.** `FakeChatModel` caută pasaje `[n] etichetă\ntext` și
  `<question>...</question>`; promptul din Pasul 18 trebuie să folosească exact acest format.

### Cum o spun la interviu
- „Am un furnizor fals, determinist, pentru embedding și pentru modelul de chat. Nu e trișare: testez logica mea
  (tăiere, căutare, filtre de refuz, citări), nu modelul furnizorului. Testele merg gratis, repetabil, fără rețea.”
- „Clienții de model se construiesc într-un singur modul. Trec de la fake la OpenAI sau Azure schimbând o variabilă de
  mediu, iar o setare lipsă oprește procesul la pornire.”
- „Modelul fals e lexical și intenționat slab: nu pretind că măsoară calitatea unui model real. Calitatea o măsor separat,
  cu un set de întrebări cu răspuns cunoscut, și cu embedding real când am cheie.”

---

## Faza D — Ingestia documentelor (pașii 11–14) · 2026-10-17

### Ce am făcut
- **Pas 11 — `chunking.py`.** Citire `.md` / `.txt` / `.pdf`, tăiere întâi după structură (titluri Markdown, pagini PDF),
  apoi după mărime (800 de caractere, suprapunere 100). Fiecare bucată știe de unde vine (secțiune, pagină).
- **Pas 12 — `ingest.py`.** Ingestie idempotentă: hash `sha256`, embedding în afara tranzacției, o tranzacție scurtă care
  înlocuiește bucățile documentului.
- **Pas 13 — documente de test.** Cinci fișiere în `sample-docs/` cu numere precise (25 de zile, 10 zile, 500 EUR).
- **Pas 14 — `cli.py`.** `python -m docuask.cli ingest <folder>` (și `ask`, activ după Faza G).

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Tai întâi după structură, apoi după mărime | O bucată rareori amestecă două subiecte; păstrează secțiunea/pagina, de unde se fac citările |
| `embed_text` = calea secțiunii + textul (doar pentru embedding) | „10 zile” nu spune nimic; „Sick leave > 10 zile” spune. Utilizatorul vede textul original |
| Suprapunere de 100 de caractere | O idee tăiată la graniță nu dispare |
| `_clean` scoate `\x00` | Postgres refuză caracterul NUL în coloane text; unele PDF-uri îl produc |
| `utf-8-sig` la decodare | Ignoră BOM-ul pe care Windows îl adaugă uneori |
| Excepții proprii (`UnsupportedDocument`, `EmptyDocument`, `IngestConflict`) | Stratul API le traduce în 415, 422 și 409 (Faza G) |
| `sha256` + `UNIQUE(name)` | Reîncărcarea aceluiași fișier nu costă nimic; două încărcări simultane nu strică nimic |
| **Embedding în afara tranzacției** | Apelul de rețea e lent; nu țin o conexiune și un lock cât aștept |
| `SELECT ... FOR UPDATE` pe document + `IntegrityError` → `IngestConflict` | Actualizările aceluiași document se serializează; la o cursă pe un nume nou, constrângerea unică alege câștigătorul |
| Verific lungimea vectorilor (1536) înainte de scriere | O dimensiune greșită oprește ingestia înainte de orice modificare în bază |
| Dependențe (`sessions`, `embeddings`, `settings`) ca parametri | Testele pot da un embedding fals și o bază de test |
| `cli.py`: importul `docuask.api` în interiorul `cmd_ask` | Vezi „Capcane”: ghidul avea o greșeală de ordine |

### Ce am verificat (rezultate reale)
- Chunking pe un `.md` mic: două bucăți, `'Leave Policy > Sick leave'` și `'Leave Policy > Annual leave'`, cu `embed_text`
  = secțiune + `\n` + text
- Fișier `.docx` → `UnsupportedDocument: Unsupported file type '.docx'. Use .md, .txt or .pdf`
- Text lung cu `size=120, overlap=30` → 4 bucăți, niciuna peste 120 de caractere
- Ingestie, prima rulare: 5 × `created`, fiecare `(5 chunks)`; a doua rulare: 5 × `unchanged` (idempotență confirmată)
- În bază: `documents` = 5, `chunks` = **25**; coloana `section` conține căi de tipul
  `Information Security > Passwords and accounts`
- `ruff check` și `ruff format --check` curate
- `python -m docuask.cli ask "x"` → `ModuleNotFoundError: No module named 'docuask.api'` (așteptat până la Faza G)

**Neverificat încă:** ingestia sub concurență (testul din Pasul 26, Faza H); citirea unui PDF real (`_load_pdf` nu a
rulat pe niciun fișier; testul din Faza H generează un PDF cu `fpdf2`); embedding real (acum e cel fals).

### Capcane
- **Greșeală de ordine în ghid.** `cli.py` din PDF începe cu `from docuask.api import get_rag_service`, dar `api.py` apare
  abia în Faza G; urmând ghidul exact, comanda de ingestie pică cu `ModuleNotFoundError`. Remediu: importul se mută în
  `cmd_ask`. Am verificat varianta într-o bază temporară înainte de a o folosi. Bonus de design: `ingest` nu încarcă
  stratul web.
- **800 de caractere nu e „corect”**, e un punct de pornire; îl aleg măsurând (Faza I).
- **PDF scanat** (doar imagini) → `EmptyDocument`: ar fi nevoie de OCR.
- **Embedding în interiorul tranzacției** ar ține o conexiune și un lock cât durează apelul către API; cu zece încărcări
  simultane se epuizează pool-ul.

### Cum o spun la interviu
- „Tai documentele întâi după structură, apoi după mărime, și păstrez secțiunea sau pagina pentru fiecare bucată: de
  acolo vin citările. La embedding pun titlul secțiunii în fața textului, ca vectorul să știe despre ce e bucata.”
- „Ingestia e idempotentă: hash de conținut, deci o reîncărcare nu costă nimic, și o constrângere unică în bază care
  decide câștigătorul când două cereri creează același document.”
- „Apelul de embedding se face în afara tranzacției: nu țin o conexiune și un lock pe rând cât aștept un API extern.”
- „Dimensiunea bucății (800) e un punct de pornire, nu o valoare demonstrată; o aleg măsurând, cu un set de
  întrebări cu răspuns cunoscut.”

---

## Faza E — Căutarea: vector, cuvinte, hibrid (pașii 15–16) · 2026-10-17

### Ce am făcut
- **Pas 15 — `retrieval.py`.** Trei moduri de căutare (`vector`, `keyword`, `hybrid`), fuziunea RRF și un
  `HybridRetriever` (interfață standard LangChain).
- **Pas 16 — încercare manuală.** Un script temporar `try_search.py` (șters după rulare, nu face parte din proiect) care
  rulează cele trei moduri pe aceeași întrebare, pe cele 25 de bucăți din bază.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Căutare **hibridă** (vector + cuvinte), nu doar vectorială | Vectorii prind sensul; cuvintele prind identificatori, numere și coduri („GDPR”, „SEC-”) pe care embedding-urile le estompează |
| Fuziune **RRF** pe poziții, nu pe scoruri | Distanța cosinus și rank-ul full-text au scări diferite; pozițiile se pot combina fără calibrare |
| `RRF_K = 60` (din lucrarea originală) | Aplatizează avantajul poziției 1 față de 2; nu-l „optimizez” fără o evaluare |
| Interogarea `tsquery` o construiesc eu din tokeni `\w+`, cu `OR` | O întrebare naturală trebuie să găsească bucăți cu *o parte* din cuvinte (`plainto_tsquery` le-ar cere pe toate); fără sintaxă injectabilă (`&`, `!`, `:*`) |
| `best_distance` = cea mai mică distanță dintre **toți** candidații vectoriali | O folosește poarta de refuz nr. 1 (Faza F), chiar dacă bucata nu intră în top-k |
| `candidate_k = 20` per ramură înainte de fuziune | Fiecare listă aduce destui candidați ca fuziunea să aibă ce combina |
| Egalitățile se rezolvă după `Chunk.id` | Rezultat determinist, deci testabil |
| `HybridRetriever` ca `BaseRetriever` | Același cod intră în orice lanț, agent sau unealtă de evaluare care așteaptă un retriever |
| Căutarea ca modul separat de generare | O măsor și o testez separat: dacă bucata corectă nu ajunge în prompt, modelul nu o mai salvează |

### Ce am verificat (rezultate reale)
Întrebarea „How long is paternity leave for fathers?”, `k=3`:
```text
vector   best_distance=0.5625 top3=[('Parental leave', 0.0), ('Annual leave', 0.0), ('Carry-over', 0.0)]
keyword  best_distance=None top3=[('Parental leave', 0.0), ('Annual leave', 0.0), ('Carry-over', 0.0)]
hybrid   best_distance=0.5625 top3=[('Parental leave', 0.0328), ('Annual leave', 0.0323), ('Carry-over', 0.0317)]
```
- Toate trei modurile pun „Parental leave” pe primul loc. Scorurile RRF se potrivesc cu formula: `0.0328 = 2/61`
  (poziția 1 în ambele liste), `0.0323 = 2/62`, `0.0317 = 2/63`.
- Alte întrebări, rulate pe baza locală (embedding fals): „SEC- ticket prefix” → *Reporting incidents*
  (`best_distance` 0.7226); „how much money for a desk and chair at home” → *Equipment* (0.7764).
- **Întrebarea fără răspuns în documente** „What is the capital of France?” → `best_distance` **0.9136**, iar căutarea
  întoarce totuși *Booking travel* („capital cities”). Față de 0.56–0.78 la întrebările cu răspuns, diferența e clară,
  deci poarta de refuz bazată pe distanță are pe ce se sprijini. Pragul din `config.py` (0.95) e prea larg pentru
  embedding-ul fals; îl calibrez la Pasul 41. (Sunt puține întrebări și embedding fals: o indicație, nu o dovadă.)
- `ruff check` și `ruff format --check` curate.

**Neverificat încă:** calitatea căutării măsurată cu hit@k și MRR (Faza I); `HybridRetriever` nu a rulat încă
(intră în joc în Faza F și I); folosirea efectivă a indecșilor HNSW și GIN (cu 25 de rânduri planificatorul probabil
alege oricum scanare secvențială; ar conta la volum mare); căutare cu embedding real.

### Capcane
- **Stemming englezesc.** Configurația `english` din full-text merge pentru documente în engleză; pentru română ar trebui
  alta, sau doar partea vectorială. Se alege per limbă.
- **Constanta 60 din RRF** nu se schimbă fără o evaluare care să arate că ajută.
- **Distanță vs similaritate.** Distanța cosinus: mic = bun (0 = identic). Similaritatea = `1 − distanță`: mare = bun.
  E ușor să le încurci când compari un prag.
- **Embedding-ul fals e lexical**, deci aici `vector` și `keyword` dau liste asemănătoare; diferența reală dintre
  moduri apare cu embedding real și la evaluare.

### Cum o spun la interviu
- „Căutarea e hibridă: vectorii prind sensul întrebării, iar full-text-ul prinde identificatori și numere exacte. Le
  combin cu Reciprocal Rank Fusion, pe poziții, ca să nu calibrez două scări diferite de scoruri.”
- „Din întrebare construiesc eu interogarea full-text, din cuvinte simple legate prin OR. Așa o întrebare naturală
  găsește rezultate, iar utilizatorul nu poate injecta operatori de interogare.”
- „Păstrez cea mai mică distanță dintre candidații vectoriali, chiar dacă bucata nu ajunge în top: pe ea se bazează
  primul filtru de refuz, înainte să plătesc un apel la model.”
- „Căutarea e un modul separat de generare, ca s-o pot testa și măsura separat: dacă bucata corectă nu ajunge în prompt,
  niciun model n-o mai salvează.”

---

## Faza F — Lanțul RAG (pașii 17–19) · 2026-10-17

### Ce am făcut
- **Pas 17 — `schemas.py`.** Contractul API: ce intră (`SearchIn`, `AskIn`) și ce iese (`AskOut`, `CitationOut`, `UsageOut`
  etc.), validat cu Pydantic. Rândurile din bază nu ies niciodată din stratul de servicii.
- **Pas 18 — `rag.py`.** Lanțul LCEL: căutare → poarta 1 → prompt cu pasaje numerotate → model cu ieșire JSON strictă →
  poarta 2 → poarta 3 → răspuns + citări + tokeni. Plus `RagService` (`retrieve`, `answer`, `ask`, `stream`).
- **Pas 19 — prima rulare.** Cu un script temporar `try_ask.py` (șters după rulare): o întrebare cu răspuns și două
  refuzuri, unul de la model și unul de la poarta de distanță.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Schemele Pydantic separate de entitățile din bază | Pot schimba baza fără să stric clienții; validare gratis (lungimi, tipuri) |
| Prompt cu reguli explicite: doar pasajele din `<context>`, „Not covered” e mai bun decât o presupunere, datele sunt date | Reduce halucinațiile și limitează efectul instrucțiunilor strecurate în întrebare sau în documente |
| Ieșire structurată: `Draft(covered, answer, source_ids)` | JSON strict în loc de text liber: pot verifica în cod, nu doar spera |
| **Trei filtre „nu știu” independente**, în ordinea costului | (1) distanța, gratis, înainte de model; (2) `covered`, costă un apel; (3) citările verificate în cod, gratis, după model |
| Citările se verifică în cod (`_citations`) | Un număr inventat de model dispare; un răspuns pe care nimeni nu-l poate verifica nu se afișează |
| `neutralize`: `<context>`/`<question>` din date devin `[tag]` | Un text nu poate „închide” secțiunea din prompt (protecție de bază împotriva prompt injection) |
| LCEL (`prompt \| model \| ...`) | Fiecare pas e un Runnable: lanțul se poate invoca, transmite în flux, trasa și testa cu un model fals |
| `retrieve` folosește o sesiune scurtă, închisă **înainte** de apelul la model | Nu țin o conexiune din pool pe durata unui apel de câteva secunde |
| `usage` din `usage_metadata` al mesajului brut | Văd tokenii consumați la fiecare întrebare (cost) |
| Streaming cu `done` final care poate spune „refuzat” | Limita onestă: tokenii sunt deja pe ecran când verific citările; clientul trebuie să înlocuiască textul |

### Ce am verificat (rezultate reale)
- „How many days is sick leave paid?” → răspuns cu citarea `[1]`, `citations[0]` = `leave-policy.md`, secțiunea
  `Leave Policy > Sick leave`; `usage` 522 tokeni de intrare / 53 de ieșire; `best_distance` 0.548.
- „zebra giraffe safari” → `refused: true`, `reason: not_covered`, `usage` 479 / 0, `best_distance` 0.906. Poarta 1 a lăsat
  cererea să treacă (0.906 < 0.95), iar modelul a refuzat (poarta 2).
- Același refuz cu `MAX_DISTANCE=0.8` → `reason: too_far`, `usage` **0 / 0**: modelul n-a fost apelat, refuzul n-a costat nimic.
- „What is the capital of France?” (rulat pe baza locală) → primește un **răspuns** despre „capital cities” din documentul de
  cheltuieli, cu citarea `[1]` (distanța 0.9136, sub pragul 0.95). Limita cunoscută a embedding-ului și a modelului false.
- `ruff check` și `ruff format --check` curate.

**Neverificat încă:** `stream()` (rulează în Faza G, prin SSE); prompt-ul cu un **model real** (`with_structured_output`
a fost folosit doar cu modelul fals); declanșarea manuală a `no_citation` și `no_passages` (acoperite de testele din
Faza H); protecția împotriva prompt injection e de bază și nu a fost testată adversarial.

### Capcane
- **A doua greșeală de ordine în ghid.** Pasul 19 rulează `python -m docuask.cli ask ...`, dar `ask` importă
  `docuask.api`, care apare abia în Faza G. Remediu: un script temporar care construiește direct `RagService`.
- **`with_structured_output(..., include_raw=True)` întoarce un dicționar** (`raw`, `parsed`, `parsing_error`), nu obiectul
  direct; uitat, dă `AttributeError: 'dict' object has no attribute 'covered'`.
- **Sesiune de bază de date ținută în timpul apelului la model** ar bloca conexiuni din pool; la zece cereri simultane
  și un pool de zece, restul așteaptă.
- **Modelul fals poate fi păcălit** („capital of France”): nu e un bug de ascuns, e motivul pentru care există trei
  filtre și pentru care calibrez pragul pe date reale (Faza I, Pasul 41).

### Cum o spun la interviu
- „Sistemul poate spune «nu știu» în trei feluri independente, în ordinea costului: un prag de distanță înainte de
  model (zero cost), flag-ul `covered` de la model și o verificare în cod că numerele citate există. Am verificat
  pe rând fiecare: la refuzul de la prag, modelul nu e apelat deloc.”
- „Modelul întoarce JSON strict (`covered`, `answer`, `source_ids`), iar eu verific citările în cod: un număr inventat
  dispare, iar un răspuns fără sursă verificabilă nu se afișează.”
- „Promptul are reguli explicite (doar sursele date, «not covered» în loc de presupuneri, textul din întrebare e date,
  nu instrucțiune), iar textul din documente nu poate închide secțiunea din prompt. E o protecție de bază, nu una completă.”
- „Lanțul e scris în LCEL, deci îl pot testa cu un model fals. Sesiunea de bază de date se închide înainte de apelul la
  model, ca să nu blochez conexiuni din pool.”
- „Cu modelul fals, sistemul răspunde greșit la o întrebare fără legătură („capital of France”); știu asta și o măsor
  la evaluare, nu o ascund.”

---

## Interludiu — prima rulare cu model real, pe 3 întrebări (între Faza F și G) · 2026-10-17

Ghidul face asta abia la Pasul 40; am făcut-o mai devreme, pe scurt, ca să am o comparație reală față de modelele false. Nu e
evaluarea completă (aceea rămâne la pașii 40–41, pe cele 28 de întrebări).

### Ce am făcut
- Cheia OpenAI în `.env` (ignorat de git), `CHAT_MODEL=gpt-4o-mini`, embedding implicit `text-embedding-3-small` (1536).
- O **bază separată**, `docuask_real`, cu schema aplicată prin Alembic și cele 5 documente reîncărcate cu embedding real.
- Furnizorul real îl aleg **per terminal**, prin variabile de mediu (`LLM_PROVIDER=openai`, `DATABASE_URL=...docuask_real`);
  în `.env` nu e `LLM_PROVIDER`, deci restul fazelor rămân deterministe și gratuite.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Bază separată pentru vectorii reali | `ingest` compară hash-ul și ar fi spus `unchanged` pe aceleași fișiere; întrebările reale s-ar fi căutat printre vectori falși |
| Cheia doar în `.env`; furnizorul ales per sesiune de terminal | Cheia nu ajunge în cod, comenzi sau git; nu pot apela din greșeală un API plătit |
| Verificarea cheii fără s-o afișez (doar valori booleene) și căutare în fișierele urmărite și în istoric | 0 potriviri cu un șir de forma `sk-...`; cheia nu a ieșit din `.env` |

### Rezultate reale
| Întrebare | `best_distance` fals → real | tokeni (intrare/ieșire) fals → real | rezultat cu modelul real |
|---|---|---|---|
| „How many vacation days do I get each year?” | 0.756 → **0.514** | 500/47 → 574/29 | răspuns corect, citare *Annual leave* |
| „How long is paternity leave for fathers?” | 0.563 → **0.298** | 524/46 → 578/38 | răspuns corect, citare *Parental leave* |
| „What is the capital of France?” (fără răspuns în documente) | 0.914 → **0.798** | 483/34 → 559/11 | **refuz** `not_covered` (modelul fals răspundea greșit) |

- **Vectori:** în `docuask` (fals) un vector are 32 de componente nenule din 1536; în `docuask_real`, 1536 din 1536 (dens).
- **Separare:** cu embedding real, întrebările cu răspuns au distanțe 0.30–0.51, cea fără răspuns 0.80; cu cel fals,
  0.56–0.76 față de 0.91. Marja e mai mare cu embedding real (aprox. 0.28 față de 0.16), dar pe doar trei întrebări.
- **Pragul 0.95 e prea larg:** întrebarea fără răspuns (0.80) a trecut de poarta 1, iar refuzul l-a făcut modelul (poarta 2),
  după ce a consumat 559 de tokeni de intrare. Un prag între ~0.5 și ~0.8 ar fi refuzat-o gratis la poarta 1. **Nu l-am
  schimbat:** îl calibrez la Pasul 41, pe setul întreg.
- Răspunsurile modelului real sunt mai scurte și mai precise (o propoziție) decât extracția modelului fals.
- Ieșirea structurată (`json_schema`, implicit în `langchain-openai` 1.7.2) a mers cu `gpt-4o-mini`: `covered` și `source_ids`
  au fost parsate, iar citările au trecut verificarea.

### Neverificat încă
Cifrele pe setul de 28 de întrebări (hit@k, MRR, distanțe) cu embedding real; latențele (nu le-am măsurat); costul în bani
(am văzut doar tokeni); `stream()` cu model real; comportamentul sub încărcare.

### Cum o spun la interviu
- „Am rulat DocuAsk cu OpenAI (`text-embedding-3-small` și `gpt-4o-mini`) schimbând doar variabile de mediu: clienții se
  construiesc într-o singură fabrică. Pe trei întrebări, embedding-ul real a dat distanțe mai mici la întrebările cu
  răspuns și a separat mai bine întrebarea fără răspuns; evaluarea completă o fac pe un set de 28 de întrebări.”
- „Pragul de distanță implicit era prea larg pentru embedding-ul real: refuzul l-a făcut modelul, nu poarta de distanță,
  deci am plătit tokeni. De aceea îl calibrez pe date, nu îl ghicesc.”
- „Am folosit o bază separată pentru vectorii reali, fiindcă ingestia idempotentă ar fi sărit peste re-embedding și aș fi
  amestecat vectori falși cu reali.”

---

## Faza G — API-ul HTTP (pașii 20–23) · 2026-10-17

### Ce am făcut
- **Pas 20 — `security.py`.** Autentificare cu chei API, ca dependență FastAPI.
- **Pas 21 — `api.py`.** Rutele: documente (încărcare, listă, ștergere), `/v1/search`, `/v1/ask`, `/v1/ask/stream` (SSE) și
  `/health/live` + `/health/ready`.
- **Pas 22 — `main.py`.** Aplicația: verificări la pornire, `X-Request-ID` în loguri, handler global de erori.
- **Pas 23 — server și `curl`.** `uvicorn docuask.main:app --port 8000`, apoi toate rutele încercate cu `curl.exe`.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Cheie API în antetul `X-API-Key`, comparată cu `hmac.compare_digest`, **toate** cheile verificate | Timpul răspunsului nu scapă informații despre cheie; lista de chei (`API_KEYS=a,b`) permite rotație fără oprire |
| Autentificarea e o dependență pe routerul `/v1` (`protected`); `/health` e separat, fără cheie | Nicio rută protejată nu uită cheia; proba de sănătate vine de la orchestrator |
| Strat HTTP subțire: validează, autentifică, apelează serviciul, traduce excepțiile în coduri | Logica stă în `ingest.py` și `rag.py`, nu în rute (ca un controller subțire în Spring) |
| Excepții de domeniu → 415 / 422 / 409; limită de dimensiune → 413; document lipsă → 404 | Clientul primește coduri clare, nu 500 |
| Fișierul se citește cu **un octet peste limită** | Dacă îl primesc, e prea mare (413), iar în memorie nu țin niciodată mai mult decât limita |
| Rute sincrone (`def`), rulate de FastAPI într-un thread pool | Munca e I/O blocant (bază, API de model); cod simplu, scalat cu replici |
| `/v1/ask/stream` face căutarea **înainte** de flux | Sesiunea de bază se închide înainte de apelul lent la model |
| `/health/live` nu atinge baza; `/health/ready` verifică baza, extensia `vector` și schema migrată | O pană de bază nu repornește inutil pod-uri sănătoase; un pod care pornește înainte de migrare nu primește trafic |
| `lifespan`: verific `EMBEDDING_DIM` și construiesc clienții la pornire | **Fail fast**: o setare sau o cheie greșită oprește procesul acum, nu la prima cerere |
| `X-Request-ID` (al clientului sau generat), păstrat într-un `ContextVar`, în răspuns și în loguri | Pot urmări o cerere prin loguri |
| Handler global: `500 Internal error`, fără detalii către client | Detaliile (și orice secret din ele) rămân în loguri |
| `/v1/search` ca endpoint de depanare | Când un răspuns e greșit, mă uit întâi la ce a găsit căutarea |

### Ce am verificat (rezultate reale)
- Fără cheie → `401`; `/health/ready` → `{"status":"ready"}`.
- Încărcare `leave-policy.md` (deja ingerat din CLI) → `"status":"unchanged"`: idempotența funcționează și prin HTTP.
- `/v1/ask` „How many days is sick leave paid?” → răspuns cu `[1]`, citare `leave-policy.md > Leave Policy > Sick leave`, `usage` 522/53.
- `/v1/ask` „zebra giraffe safari” → `refused: true`, `reason: not_covered`, `best_distance` 0.906.
- Document nou `notes.md` → `201`, `"status":"created"`, `chunk_count: 1`; apoi `DELETE` → `204`, a doua oară → `404`
  `{"detail":"No such document"}`; lista a revenit la cele 5 documente.
- Erori: `.exe` → `415 Unsupported file type '.exe'. Use .md, .txt or .pdf`; fișier gol → `422 The document has no readable text`;
  fișier de 5.000.001 de octeți → `413 Maximum size is 5000000 bytes`.
- `X-Request-ID: my-trace-1` trimis de mine apare în răspuns.
- `/v1/search` hibrid „paternity leave” → primul rezultat *Parental leave*, `vector_rank` 1 și `keyword_rank` 1, scor
  `0.0328` (= 2/61); al doilea *Annual leave*, `0.0323`.
- `/v1/ask/stream` → o serie de evenimente `token`, apoi `done` cu citările.
- Un `id` invalid într-un URL (`<ID>` literal, din greșeală) → `422`: validarea UUID din FastAPI.
- Rulat de mine pe un server de test, pe o bază temporară: `/docs` → 200; întrebare prea scurtă (`"x"`) → 422; ID generat de
  12 caractere când clientul nu trimite unul.

**Neverificat încă:** autentificarea cu mai multe chei (rotația) și comportamentul sub concurență (testele din Faza H);
`/health/ready` cu `503` (nu am provocat baza goală); streaming cu model real; sarcină și latențe.

### Capcane
- **`curl` în PowerShell** e un alias pentru `Invoke-WebRequest`: folosesc `curl.exe`.
- **JSON în `-d '{"..."}'` se strică în PowerShell 5.1**: ghilimelele dispar și `curl` dă eroarea 3. Soluție: JSON-ul se trimite
  prin stdin, `'{"question":"..."}' | curl.exe ... -d "@-"`.
- **Un placeholder lăsat în comandă** (`<ID>`) n-a șters nimic (`422`), iar documentul de test a rămas în bază (6 documente
  în loc de 5). Trebuie verificat că starea reală e cea așteptată, nu doar că „n-a dat eroare”.
- **`[000]` la `curl`** înseamnă că fișierul de încărcat nu există (comanda care îl crea nu rulase), nu o eroare a serverului.
- **Linia de rezumat din middleware are `[-]`** în loc de ID: `request_id_var` e resetat în `finally` înainte de `log.info`.
  Liniile din timpul cererii (ex. `ask ...`) au ID-ul corect, la fel și antetul din răspuns. Limită cunoscută a codului din
  ghid; remedierea ar fi logarea înainte de resetare.
- **Streaming:** evenimentul `done` are `usage` 0/0 (calea de flux nu numără tokenii), iar ultimul `token` are text gol.
- **Copierea din terminal poate tăia spații** la capătul de rând al unui rând lung (am văzut „daysof”, „months,unpaid” în ce
  am lipit); în bază și în API spațiile sunt corecte, am verificat.

### Cum o spun la interviu
- „API-ul e subțire: autentifică, validează și traduce excepțiile în coduri clare (415, 422, 409, 413, 404). Logica stă în
  servicii, nu în rute.”
- „Cheile API se compară în timp constant și toate cheile se verifică, deci pot face rotație fără oprire. Autentificarea e o
  dependență pe tot routerul `/v1`, deci nicio rută nu o uită.”
- „Separ liveness de readiness: liveness nu atinge baza, ca o pană de bază să nu repornească pod-uri sănătoase; readiness
  verifică și schema migrată, deci un pod care pornește înainte de migrare nu primește trafic.”
- „Aplicația pică la pornire dacă o setare e greșită (dimensiunea embedding-ului, cheia lipsă), nu la prima cerere. Fiecare
  cerere are un ID în loguri, iar o excepție neașteptată devine un 500 generic, fără detalii către client.”
- „Limite pe care le recunosc: o singură categorie de chei (fără izolare pe client), cheie simplă în loc de token semnat,
  streaming-ul nu poate retrage tokenii deja trimiși și nu raportează consumul.”

---

## Faza H — Teste (pașii 24–31) · 2026-10-17

### Ce am făcut
Opt fișiere de teste, **58 de teste** în total, pe Postgres real cu pgvector:
| Fișier | Teste | Ce acoperă |
|---|---|---|
| `conftest.py` | — | baza de test, migrarea, fixture-uri (`sessions`, `settings`, `loaded`, `client`) |
| `test_fake.py` | 6 | embedding-ul și modelul fals |
| `test_chunking.py` | 7 | citire și tăiere (md, txt, pdf), NUL, fișiere invalide |
| `test_ingest.py` | 6 | ingestie, idempotență, dimensiune greșită, **concurență** |
| `test_retrieval.py` | 13 | RRF, cele trei moduri, text ostil, retriever LangChain |
| `test_rag.py` | 11 | cele trei filtre, citări inventate, ieșire malformată, streaming |
| `test_api.py` | 9 | rutele, codurile de stare, cheile, SSE, `X-Request-ID` |
| `fake_openai.py` + `test_providers.py` | 6 | contract cu clasele reale OpenAI/Azure, contra unui server local fals |

### Decizii și de ce
| Decizie | De ce |
|---|---|
| **Postgres real cu pgvector** (Testcontainers), nu SQLite sau mock-uri | Comportamentul pgvector, full-text, `FOR UPDATE` și `UNIQUE` nu se poate simula; testez pe același tip de bază ca în producție |
| Containerul pornește o dată pe sesiune; în CI, `TEST_DATABASE_URL` | Rapid local; în CI folosesc serviciul existent |
| `alembic upgrade head` rulat în teste | Testez și migrarea: dacă ar fi greșită, toate testele ar pica |
| `TRUNCATE chunks, documents RESTART IDENTITY CASCADE` înainte de fiecare test | Fiecare test pornește dintr-o bază goală; testele nu se influențează |
| `Settings(_env_file=None, llm_provider="fake")` în teste | `.env` (cu cheia OpenAI) nu poate ajunge în teste; nu se poate apela din greșeală un API plătit |
| `TestClient` + `dependency_overrides` | Testez aplicația fără server, înlocuind baza, setările și modelele |
| Dubluri de test definite inline (`Exploding`, `WrongWidth`, `Synced`, `Spy`, `Liar`, `Broken`) | Fiecare regulă se dovedește izolat: nicio chemare la embedding la fișier neschimbat, vectori de lățime greșită, model care minte sau care e apelat |
| **Test de concurență cu `threading.Barrier(2)`** | Fără barieră, două fire rareori se întrec exact; așa ambele termină embedding-ul și scriu împreună. Rezultat acceptat: un document, 2 bucăți (nu 4) |
| Text ostil în căutare (`'; DROP TABLE chunks; --`, `a & b \| !c :*`, ...) | Dovedește că textul utilizatorului nu poate strica interogarea full-text și că tabela rămâne |
| Server OpenAI fals (FastAPI) + clasele **reale** `ChatOpenAI`/`OpenAIEmbeddings` | Prind greșeli de cerere (ex. embedding trimis ca liste de token-id), fără cheie, rețea sau cost |
| PDF de test generat cu `fpdf2` | Nu țin fișiere binare în repo |
| Numele testelor = specificația | Se citesc ca o listă de reguli ale sistemului |

### Ce am verificat (rezultate reale)
- `uv run pytest --cov=docuask --cov-report=term-missing` → **58 passed** în 18,7 s; acoperire **TOTAL 92 %**.
  100 %: `chunking`, `config`, `ingest`, `models`, `providers`, `retrieval`, `schemas`, `security`; 91–95 %: `api`, `fake`, `main`, `rag`;
  75 %: `db`; **0 %: `cli`**.
- `ruff check` → `All checks passed!`; `ruff format --check` → 33 de fișiere formatate.
- Rezultat identic cu rularea mea de control pe același cod (58 de teste, 92 %).

**Neverificat / limite, cinstit:**
- `cli.py` nu are teste (0 %); `db.py` 75 % (`get_engine`, `get_session_factory` și `get_session` nu rulează în teste, fiindcă
  `conftest.py` le înlocuiește prin `dependency_overrides`).
- Ramuri de eroare nescrise: `/health/ready` cu `503`, handler-ul global `500`, verificarea `EMBEDDING_DIM` din `lifespan`,
  tratarea erorilor la streaming (`rag.py` 209–212), calea `409 IngestConflict` prin HTTP.
- Serverul OpenAI fals dovedește **forma** cererilor și răspunsurilor, nu calitatea unui model real; calitatea o măsor la
  evaluare (Faza I) și la pașii 40–41.
- Testul de concurență folosește două fire, nu o sarcină reală.
- `test_evaluate.py` (testul pentru evaluare) vine în Faza I, deci numărul final de teste va crește.

### Capcane
- **Docker trebuie pornit**; prima rulare e lentă, fiindcă pornește un container nou (și descarcă imaginea, dacă lipsește).
- **Terminal curat.** `lifespan`-ul din `main.py` apelează `get_settings()` direct, nu prin `Depends`, deci override-ul din
  teste nu-l atinge; variabilele `LLM_PROVIDER`/`DATABASE_URL` rămase din alte experimente pot schimba comportamentul.
- **Baza de test e a ei**, pe un port aleatoriu, nu `docuask`-ul de dezvoltare; la final, containerul dispare.
- **Cod mort în teste:** clasa `MustNotBeCalled` din `test_rag.py` e definită dar nefolosită (e în codul verificat din ghid și
  `ruff` n-o prinde); la o curățenie ulterioară se șterge.
- `.coverage` și `.pytest_cache/` sunt în `.gitignore`, nu se commitează.

### Cum o spun la interviu
- „Testele rulează pe un Postgres real cu pgvector, pornit de Testcontainers, nu pe mock-uri; migrarea se aplică în teste,
  deci o verific și pe ea. Fiecare test pornește dintr-o bază goală, iar setările de test ignoră `.env`, deci cheia reală nu
  poate ajunge acolo.”
- „Concurența: două fire care ingerează același nume nou, sincronizate cu o barieră ca să se întreacă efectiv. Rezultatul:
  un singur document și două bucăți, niciodată duplicate. Asta dovedește constrângerea unică și blocarea rândului.”
- „Verific fiecare filtru de refuz separat: un spion dovedește că modelul nu e apelat când poarta de distanță refuză, un
  «model mincinos» care citează un pasaj inexistent e respins, iar o ieșire malformată e eroare, nu răspuns inventat.”
- „Pentru OpenAI am un server HTTP local care vorbește protocolul, iar clasele reale ale clientului îi vorbesc. Așa prind
  greșeli de cerere fără cheie și fără cost. Nu dovedește calitatea modelului; pe aceea o măsor separat.”
- „Am 58 de teste și 92 % acoperire. Nu sunt acoperite CLI-ul și câteva ramuri de eroare, și știu care.”

---

## Faza I — Evaluarea: măsori, nu ghicești (pașii 32–34) · 2026-10-17

### Ce am făcut
- **Pas 32 — `eval/golden.jsonl`.** 28 de întrebări: 22 **cu răspuns**, formulate altfel decât textul („vacation days” pentru „annual leave”,
  „doctor's note” pentru „medical certificate”), și 6 **fără răspuns** (2 banale, 4 „perfide”, apropiate de domeniu).
- **Pas 33 — `evaluate.py` + `tests/test_evaluate.py`.** Măsoară hit@1, hit@3 și MRR pentru cele trei moduri de căutare și
  verifică poarta de refuz (distanța). Opțiunea `--min-hit3` dă cod de ieșire 1 dacă hibridul scade sub prag.
- **Pas 34 — rularea.** Tabelul de cifre pe cele 5 documente din bază.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Set de evaluare propriu („golden”), cu întrebări parafrazate | Testează și sensul, nu doar potrivirea de cuvinte; îmi spune dacă o schimbare ajută sau strică |
| Șase întrebări fără răspuns, dintre care patru apropiate de domeniu | Un sistem prost ar inventa răspunsuri tocmai la întrebări care *par* acoperite („CEO's salary”) |
| Metrici: hit@1, hit@3, MRR | hit@1 = secțiunea corectă e prima; hit@3 = e printre primele trei; MRR răsplătește poziția (1,0 = mereu prima) |
| Evaluare prin `HybridRetriever` | Același retriever standard LangChain pe care îl poate folosi orice lanț sau agent |
| Poarta de refuz măsurată separat: `positives_max` față de `negatives_min` | Văd dacă există un prag care separă întrebările cu răspuns de cele fără |
| `--min-hit3` cu cod de ieșire 1 | Poartă pentru CI: un build pică dacă cineva strică căutarea |
| Evaluarea e și un test obișnuit în suită | O regresie de căutare pică la `pytest`, nu rămâne neobservată |
| Experimentele cu altă configurație rulează pe **baze separate** | Ingestia e idempotentă (hash), deci o reingestie pe aceeași bază răspunde `unchanged` și nu mai re-taie nimic |

### Ce am verificat (rezultate reale, embedding fals)
```text
| search | hit@1 | hit@3 | MRR |
| vector | 77% | 100% | 0.88 |
| keyword | 86% | 100% | 0.93 |
| hybrid | 91% | 100% | 0.95 |
MAX_DISTANCE = 0.95; farthest answerable: 0.918; nearest unanswerable: 0.702
answerable let through: 100%; unanswerable refused: 33%
```
- **Hibridul bate ambele variante simple** la hit@1 (91 % față de 77 % și 86 %), pe cele 22 de întrebări.
- **Poarta de distanță nu separă cu embedding-ul fals:** întrebarea cu răspuns cea mai îndepărtată (0,918) e mai departe decât
  cea fără răspuns cea mai apropiată (0,702); la 0,95 doar 33 % din cele fără răspuns sunt refuzate de poartă.
- `--min-hit3 0.9` → cod de ieșire 0; un prag imposibil (`1.01`) → `FAIL: hybrid hit@3 100% is below 101%` și cod 1 (rulat de mine).
- `pytest` → **60 de teste trec**, acoperire **91 %**; `ruff` curat. `evaluate.py` are ~73 % (`main` și `--report` nu au teste).
- **Experiment cu dimensiunea bucății** (rulat de mine, pe baze temporare, șterse): `CHUNK_SIZE=300` → 26 de bucăți, cifre
  neschimbate; `CHUNK_SIZE=150` → 81 de bucăți, vector 73 %, keyword 91 %, **hibrid 95 %** (MRR 0,98).

**Neverificat încă:** evaluarea cu **embedding real** (rămâne opțională aici, apoi la pașii 40–41); `MAX_DISTANCE` nu e calibrat;
se evaluează doar căutarea, nu calitatea răspunsurilor generate și nici refuzul de la un capăt la altul.

### Capcane
- **Sugestia din ghid** („schimbi `chunk_size` în `.env` și reingestezi”) **nu funcționează pe aceeași bază**: ingestia vede
  același hash și răspunde `unchanged`. Experimentul cere o bază separată.
- **Setul l-am scris eu, pe documente pe care le cunosc**: e bun ca *semnal de regresie*, nu ca dovadă de calitate; poate fi optimist.
- **Concluzii cu rezervă din experimentul cu bucăți:** 22 de întrebări și embedding fals; „hibridul a ieșit mai bun cu bucăți
  mici, pe acest set mic” nu înseamnă „schimb valoarea implicită”.
- `evaluate` se rulează din rădăcina repo-ului (calea implicită `eval/golden.jsonl` e relativă).

### Cum o spun la interviu
- „Nu am judecat căutarea «după ochi»: am un set de 28 de întrebări, 22 cu secțiunea corectă cunoscută și 6 fără răspuns, și un
  script care măsoară hit@1, hit@3 și MRR pentru căutarea vectorială, pe cuvinte și hibridă. Pe acest set, hibridul a dat 91 %
  hit@1, față de 77 % (vector) și 86 % (cuvinte).”
- „Cifrele sunt cu un embedding fals, lexical, pe un set mic scris de mine: le folosesc ca semnal de regresie, nu ca dovadă de
  calitate. Evaluarea rulează și ca test, iar cu `--min-hit3` poate opri un build în CI.”
- „Am măsurat și poarta de refuz: cu embedding fals, distanța nu separă întrebările cu răspuns de cele fără (0,918 față de
  0,702), deci avem trei filtre, nu unul. Pragul îl calibrez pe date, nu îl ghicesc.”
- „Când am vrut să văd efectul dimensiunii bucății, am observat că ingestia idempotentă ignoră re-rularea cu altă setare;
  am folosit o bază separată pentru experiment.”

---

## Faza J — Docker (pașii 35–37) · 2026-10-17

### Ce am făcut
- **Pas 35 — `Dockerfile`.** Imagine multi-stage, non-root, cu `HEALTHCHECK`.
- **Pas 36 — `docker-compose.yml`.** Trei servicii: `db` (Postgres cu pgvector), `migrate` (job unic) și `api`.
- **Pas 37 — rularea.** `docker compose up -d --build`, încărcarea documentelor și o întrebare prin `curl`, pe containere.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| **Multi-stage**: `uv` și uneltele de build rămân în prima etapă | Imaginea finală e mică (295 MB) și are mai puțin de atacat |
| Versiunea `uv` fixată (`ghcr.io/astral-sh/uv:0.12`) și `uv sync --frozen` | Construcție repetabilă: se instalează exact `uv.lock`, iar o nepotrivire cu `pyproject.toml` oprește build-ul |
| Dependențele **înaintea codului** (două `uv sync`) | Stratul de dependențe se reface doar când se schimbă `pyproject.toml`/`uv.lock`; o rebuild cu cache a durat ~18 s, față de ~2,5 min prima dată |
| **Non-root**, UID numeric 10001 | Un proces compromis nu poate scrie în imagine; UID-ul numeric poate fi verificat de Kubernetes (`runAsNonRoot`) |
| `HEALTHCHECK` cu Python, pe `/health/live` | Imaginea `slim` nu are `curl`; liveness nu atinge baza |
| `CMD` în formă exec | uvicorn e PID 1 și primește `SIGTERM` direct, deci se oprește frumos |
| `.dockerignore` (fără `.venv`, `.env`, `tests`, `eval`, `sample-docs`, `.git`) | Imaginea nu conține secrete și nici fișiere de dezvoltare |
| `migrate` ca **serviciu separat**, one-shot, iar `api` pornește la `service_completed_successfully` | Dacă migrarea pică, API-ul nu pornește; același tipar ca un Job în Kubernetes; o replică nouă de API nu rulează migrări |
| `db` cu `healthcheck` (`pg_isready`) și `migrate` pornit după `service_healthy` | Nicio migrare pe o bază care încă nu e gata |
| `DATABASE_URL` cu gazda `db` în `environment` (are prioritate față de `env_file`) | În rețeaua Compose, serviciile se găsesc după nume, iar `.env` nu poate muta API-ul pe altă bază |
| Volum numit `pgdata` | Datele bazei supraviețuiesc unui `docker compose down`; dispar doar cu `-v` |
| `docker stop docuask-db`, **nu** `docker rm -f` | Containerul de dezvoltare n-are volum și conține și `docuask_real` (vectori reali); `rm` ar fi șters datele |

### Ce am verificat (rezultate reale)
- `docker compose up -d --build` → `api` **healthy**, `db` **healthy**, `migrate` **Exited (0)**.
- Încărcarea celor 5 documente prin `curl` → 5 × `created` (a doua rulare → `unchanged`, verificat de mine într-un proiect temporar).
- „How much is the per diem for meals on a business trip?” → răspuns cu `per diem of 35 EUR`, citare
  `expenses.md > Travel and Expenses > Daily allowance`, `usage` 476/54, `best_distance` 0,645.
- `docker compose exec api id` → `uid=10001(app) gid=999(app) groups=999(app)`.
- Imaginile `docuask-api` și `docuask-migrate`: **295 MB** fiecare (aceeași imagine, construită de două servicii).
- Curățenie: `docker compose down -v` a șters containerele, rețeaua și volumul; `docuask-db` repornit, cu bazele `docuask` și
  `docuask_real` intacte (25 de bucăți în fiecare).
- Build-ul cu `uv.lock`-ul meu a trecut și într-un proiect Compose temporar, rulat de mine (prima construcție ~2 min 36 s).

**Neverificat:** scanarea imaginii pentru vulnerabilități; sistem de fișiere `read-only`; limite de CPU/memorie; imaginea de bază
fixată după digest; build multi-arhitectură; oprirea frumoasă la `SIGTERM` sub trafic; containerul rulat cu model real; build-ul
în CI (Faza K).

### Capcane
- **Ghidul spune `docker rm -f docuask-db`** pentru a elibera portul 5434. Containerul din Pasul 5 n-are volum, deci `rm` șterge
  toate datele din el. Am folosit `docker stop` (și `docker start` după).
- **Porturile 5434 și 8000 trebuie libere:** containerul de dezvoltare și serverul `uvicorn` local le țin ocupate.
- **`.env` intră în mediul containerului `api`** (`env_file`), deci și cheia OpenAI: nu e în imagine, dar `docker compose config` sau
  `docker inspect` ar afișa-o. Providerul rămâne `fake`, deci nu se folosește.
- **`docker compose ps` ascunde serviciile terminate**; `migrate` se vede doar cu `ps -a`.
- **Bucla `for` din ghid e pentru bash**; în PowerShell am folosit `Get-ChildItem ... | ForEach-Object` cu `curl.exe`.
- **`docker compose down -v` șterge volumul Compose** (`pgdata`); fără `-v`, datele rămân între porniri.

### Cum o spun la interviu
- „Imaginea e multi-stage: `uv` instalează dependențele din `uv.lock` (`--frozen`), iar imaginea finală copiază doar rezultatul.
  Dependențele sunt într-un strat separat de cod, deci o modificare de cod reface o rebuild în câteva secunde. Rulează non-root, cu
  UID numeric, are `HEALTHCHECK`, iar uvicorn pornește în formă exec ca să primească `SIGTERM`.”
- „Migrarea e un serviciu separat, care rulează o dată și iese; API-ul pornește doar dacă migrarea a reușit. E același model ca un
  Job de Kubernetes, nu o migrare făcută de aplicație la pornire.”
- „Ce nu am făcut și știu că lipsește: sistem de fișiere read-only, limite de resurse, imagine de bază fixată după digest și
  scanare de vulnerabilități. Scanarea o pun în CI.”

---

## Faza K — CI cu GitHub Actions (pașii 38–39) · 2026-10-17

### Ce am făcut
- **Pas 38 — `.github/workflows/ci.yml`.** Două joburi: `test` (lint, teste cu baza reală, acoperire, poarta de calitate a
  căutării) și `image` (construiește imaginea și îi verifică health check-ul).
- **Pas 39 — repo pe GitHub.** Repo-ul `FILIPIOAN10/docu-ask` exista deja, cu un commit pe fază. Rămâne **regula de protecție pe
  `main`**, de pus după prima rulare verde.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Rulează la **push pe `main`** și la **pull request** | Fiecare schimbare e verificată înainte să conteze |
| `permissions: contents: read` | Cel mai mic privilegiu pentru tokenul automat al workflow-ului |
| `concurrency` cu `cancel-in-progress` | Un push nou anulează rularea veche pe aceeași ramură |
| Postgres ca **service container** (`pgvector/pgvector:pg16`) cu `health-cmd`, iar `TEST_DATABASE_URL` îl indică | Testele rulează pe aceeași bază ca în producție și nu pornesc un al doilea container |
| `LLM_PROVIDER: fake`, fără secrete în CI | Nicio cheie nu poate ajunge într-un log sau într-o rulare a unui fork |
| `uv sync --frozen` | CI instalează exact `uv.lock`, ca pe laptop și în Docker |
| `--cov-fail-under=80` | O scădere a acoperirii sub 80 % oprește build-ul |
| **Poarta de calitate a căutării:** `evaluate --min-hit3 0.9` | Dacă un commit strică căutarea (hybrid hit@3 sub 90 %), build-ul pică; tabelul apare și în rezumatul rulării (`$GITHUB_STEP_SUMMARY`) |
| Jobul `image` pornește după `test` (`needs`), construiește fără push și verifică `/health/live` | Prind o imagine care nu pornește, înainte de orice publicare |
| Acțiunile fixate pe versiuni exacte | Nicio schimbare surpriză la o etichetă mobilă; ideal, în producție, pe hash de commit |
| Regula de protecție după prima rulare verde, cu bypass pentru proprietar | Jobul `test` apare ca verificare necesară doar după ce a rulat o dată; o regulă prea strictă ar bloca push-urile directe |

### Ce am verificat înainte de push (emulare locală, nu rularea de pe GitHub)
- Toți pașii jobului `test`, într-o copie temporară și pe o bază temporară, cu aceleași variabile ca în workflow:
  `uv.lock` sincron cu `pyproject.toml` (84 de pachete), `ruff` curat, **60 de teste trec, acoperire 90,4 %** (poarta e 80 %),
  `alembic upgrade head` + ingestie + `evaluate --min-hit3 0.9` → cod de ieșire 0, cu raportul generat.
- Jobul `image`, local: imaginea se construiește, pornește, iar `/health/live` răspunde după ~5 secunde.
- Cele patru acțiuni cu versiunile din fișier există pe GitHub (`checkout@v7.0.1`, `setup-uv@v10.3.0`,
  `setup-buildx-action@v4.4.1`, `build-push-action@v7.4.0`).
- `actionlint`: un singur avertisment de stil (`SC2034`: variabila `i` din bucla `for` a pasului de health check nu e folosită);
  workflow-ul e valid. Fișierul din repo e identic cu referința, fără tab-uri.

### Prima rulare reală pe GitHub (run #1, commit `4c86b85`)
**Verde din prima**, în 2 min 3 s în total (citit din interfața GitHub și din API-ul public al repo-ului):
| Job | Rezultat | Durată | Pașii mai lungi |
|---|---|---|---|
| `test` | success | 56 s | pornirea containerelor 16 s, **Tests 27 s**, **Retrieval quality gate 4 s** |
| `image` | success | 61 s | **build 41 s**, health check al imaginii 4 s |

- Pagina rulării afișează în **Summary** tabelul de evaluare (hit@1: vector 77 %, keyword 86 %, hybrid 91 %; hit@3 100 %; poarta de
  refuz 0,918 față de 0,702), deci `$GITHUB_STEP_SUMMARY` funcționează.
- Rezultatul pe GitHub coincide cu emulația locală (aceleași cifre de evaluare); nu a fost nevoie de niciun commit de reparare.
- Rularea a produs și un artefact (înregistrarea build-ului Docker).

**Neverificat încă:** efectul cache-ului `gha` (prima rulare a pornit fără cache; o a doua rulare ar trebui să fie mai rapidă),
comportamentul pe un pull request și regula de protecție pe `main`.

### Capcane
- **Primul push e testul real.** Tot ce am verificat e o emulare; un pas poate pica pe GitHub din motive pe care local nu le văd.
- **Comentariul din pasul de calitate** zice că testele golesc tabelele; în rularea locală, `ingest` a răspuns totuși `unchanged`
  pentru toate cinci, fiindcă ultimul test lăsase deja documentele în bază. Rezultatul e același, poarta merge oricum.
- **Etichetele acțiunilor pot fi mutate**; fixarea pe hash de commit e mai sigură (nu am făcut-o).
- **Regula de protecție poate bloca push-urile directe** pe `main`; jobul `test` apare în lista verificărilor necesare abia după
  prima rulare.

### Cum o spun la interviu
- „Pipeline-ul rulează la fiecare push și PR: lint, teste pe un Postgres real cu pgvector (service container), poartă de
  acoperire la 80 %, apoi construiește imaginea și îi verifică health check-ul. Nu folosește niciun secret: modelul e fals în CI.”
- „Am o poartă de calitate pentru căutare: după teste, rulez evaluarea pe un set de întrebări cu răspuns cunoscut, iar build-ul
  pică dacă hit@3 al căutării hibride scade sub 90 %. Tabelul apare în rezumatul rulării.”
- „Workflow-ul are permisiuni minime, anulează rulările depășite și își fixează acțiunile pe versiuni exacte; următorul pas ar fi
  fixarea pe hash.”
- „Ce nu are încă: scanare de secrete și de vulnerabilități (gitleaks, Trivy, CodeQL), publicarea imaginii într-un registry,
  deploy automat și o matrice de versiuni de Python.”
- „Înainte de primul push am emulat local fiecare pas al pipeline-ului (lint, teste, poarta de evaluare, imaginea), iar prima rulare
  reală pe GitHub a fost verde, în aproximativ două minute. Dovada e în istoricul de Actions.”

---

## Faza L — Modelul real, calibrarea, README (pașii 40–43) · 2026-10-17

### Ce am făcut
- **Pas 40 — model real.** Evaluarea completă și întrebări cu `text-embedding-3-small` și `gpt-4o-mini`, pe baza separată `docuask_real`.
- **Pas 41 — calibrare.** `MAX_DISTANCE` ales din distribuția tuturor celor 28 de distanțe: **0,72**.
- **Pas 42 — Azure.** Sărit la început (n-aveam cont Azure); reluat după README, vezi secțiunea de mai jos când e completată.
- **Pas 43 — README.** Cu badge de CI, cele două seturi de cifre (embedding fals și real) și o secțiune „ce am verificat și ce nu”.

### Decizii și de ce
| Decizie | De ce |
|---|---|
| Prag **0,72**, setat ca **variabilă de mediu** în terminalul cu modelul real, nu în `.env` | `.env` e folosit și de modelul fals, la care întrebările cu răspuns ajung la 0,918; 0,72 l-ar face să refuze întrebări bune |
| Pragul stă puțin peste cea mai îndepărtată întrebare cu răspuns (0,683, marjă 0,037) și sub 0,739 | Refuzul unei întrebări bune e mai grav decât un apel de model în plus; **nu rotunjesc pragul în jos** (un prag de 0,65 ar refuza întrebarea cu cod „SEC prefix”) |
| README corectat față de ghid: „hibridul bate ambele” e adevărat doar cu embedding fals | Cu embedding real, `vector` singur a ieșit mai bine pe setul meu; afirmația trebuie să fie ce am măsurat |
| Nu schimb codul pentru egalitatea din RRF | E o descoperire de documentat și de măsurat separat, nu o modificare făcută din impuls |

### Rezultate reale
| Căutare | embedding fals: hit@1 / MRR | `text-embedding-3-small`: hit@1 / MRR |
|---|---|---|
| vector | 77 % / 0,88 | **100 % / 1,00** |
| keyword | 86 % / 0,93 | 86 % / 0,93 (identic, ca verificare: nu folosește embedding) |
| hybrid | 91 % / 0,95 | 95 % / 0,98 |

hit@3 e 100 % peste tot.

- **Distanțe reale (cea mai apropiată bucată):** 21 de întrebări cu răspuns între 0,252 și 0,612; **„SEC prefix” (întrebare cu cod) la 0,683**; fără
  răspuns: 0,671 (angajați), 0,739 (salariul CEO), 0,770 (limbajul backend), 0,798 (Franța), 0,843 (bursa), 0,922 (carbonara). Se suprapun cu 0,012.
- **Pragul pe aceste date:** 0,65 → 21/22 cu răspuns trec, 6/6 refuzate; **0,72 → 22/22 trec, 5/6 refuzate**; 0,75 → 22/22, 4/6; 0,95 → 22/22, 0/6.
- **Verificat cu `MAX_DISTANCE=0.72`:** `evaluate` → 100 % trec, 83 % refuzate; „capital of France” → `too_far`, `usage` **0/0** (înainte 559 de
  tokeni); „SEC prefix” → răspuns corect, cu citare; „How many employees work at the company?” (0,671) → trece de poartă, **modelul** refuză
  (`not_covered`, 554/11 tokeni). Cele trei filtre au lucrat în ordinea proiectată.
- **Descoperire — egalitate în RRF.** Hibridul are 95 %, nu 100 %, din cauza unei întrebări (carry-over) cu rangurile 1 (vector) și 2 (cuvinte): o bucată
  pe pozițiile (1, 2) are scorul `1/61 + 1/62`, **identic** cu una pe (2, 1), iar egalitatea se rezolvă după `chunk_id`, nu după căutarea vectorială.
  Am reprodus-o cu `fuse`. Un tie-break după rangul vectorial ar fi de încercat și de măsurat; nu l-am implementat.
- **Nedeterminism.** Aceeași întrebare („SEC prefix”) a produs două texte diferite (49 față de 57 de tokeni de ieșire), cu aceeași citare: nu trimit
  `temperature`, iar stabilitatea vine din schema strictă și din verificarea citărilor, nu din text identic.
- Testul `hybrid ≥ max(vector, keyword)` din `test_evaluate.py` e o proprietate a setării false (cu embedding real ar pica); testele rulează cu modelul fals.

### Neverificat încă
Azure OpenAI pe un endpoint real; latențe și comportament sub încărcare; calitatea pe alte documente decât cele cinci; efectul unui tie-break după
rangul vectorial; calitatea răspunsurilor generate (măsurăm doar căutarea și refuzul).

### Capcane
- **`.env` e comun** modelului fals și celui real, deci orice valoare calibrată pentru cel real (pragul) se setează în terminal, nu în `.env`.
- **Poarta de distanță nu poate separa perfect** (suprapunere de 0,012): e un filtru ieftin, nu o garanție; `covered` și citările rămân necesare.
- **Setul e mic și scris de mine**: concluziile despre „vector vs hibrid” sunt valabile pe acest set, cu acest model de embedding.

### Cum o spun la interviu
- „Am rulat DocuAsk cu OpenAI și am comparat embedding-ul fals cu cel real pe aceleași 28 de întrebări: vectorii reali au dus hit@1 de la 77 % la
  100 %, iar hibridul a ajuns la 95 %. Cu un model real, căutarea vectorială singură a ieșit mai bine decât hibridul pe setul meu mic, deci n-am
  considerat hibridul un câștig automat.”
- „Am investigat de ce hibridul pierde o întrebare și am găsit o egalitate de scor în Reciprocal Rank Fusion, rezolvată după un id arbitrar. Știu
  remediul probabil, un tie-break după rangul vectorial, dar îl voi implementa abia după ce îl măsor.”
- „Pragul de refuz l-am calibrat pe toate cele 28 de distanțe: 0,72 lasă să treacă toate întrebările cu răspuns și refuză 5 din 6 fără răspuns înainte de
  model; a șasea e refuzată de model. O întrebare străină nu mai costă tokeni.”
- „Răspunsurile unui LLM nu sunt identice la fiecare rulare; de aceea testele mele cu modele reale verifică structura și citările, nu textul.”

---

## Extra (după ghid) — pagina cu butoane · 2026-10-17

Nu e în ghid: am cerut-o eu, fiindcă un API fără interfață e greu de arătat și de explicat. Rămâne un singur fișier, fără framework și fără pas de build.

### Ce am făcut
- **`src/docuask/static/index.html`:** o pagină cu butoane pentru cheia API, **Upload**, lista documentelor cu **Delete**, **Ask**, **Show passages**
  (căutarea singură, cu distanțe și scoruri) și cinci întrebări-exemplu pe care dai click.
- **`GET /` în `main.py`:** servește pagina, cu antete de securitate.
- **Un test nou** în `tests/test_api.py` (în total **61 de teste**).

### Decizii și de ce
| Decizie | De ce |
|---|---|
| O singură pagină HTML + JavaScript, fără bibliotecă și fără build | Nimic de instalat și de întreținut; se vede ușor ce face |
| Servită chiar de API (același origin) | Nu e nevoie de CORS și nu există un al doilea server de rulat |
| Tot textul venit de la server se pune cu `textContent`, niciodată ca HTML | Un document cu etichete rău intenționate nu poate injecta cod în pagină |
| `Content-Security-Policy` (`default-src 'none'`, `connect-src 'self'`, `frame-ancestors 'none'`), `nosniff`, `Referrer-Policy` | Chiar dacă ar apărea o breșă, pagina nu poate vorbi cu alt site și nu poate fi încadrată în alt site |
| Cheia API în `sessionStorage`, nu în `localStorage` | Dispare când închizi tab-ul; nu rămâne pe disc |
| Test de gardă: `innerHTML` nu are voie să apară în pagină | Dacă cineva îl introduce, testul pică |
| „Show passages” ca buton | Când un răspuns pare greșit, te uiți mai întâi la ce a găsit căutarea (același principiu ca `/v1/search`) |
| Fără streaming în pagină | Rămâne simplă; endpoint-ul de streaming există pentru alți clienți |

### Ce am verificat
- **Într-un browser** (copie a proiectului, model fals, bază temporară): lista, întrebare cu răspuns, refuz, „Show passages”, întrebare prea scurtă,
  cheie greșită, încărcare, ștergere, aspect pe ecran de 375 px fără derulare orizontală.
- **Test de siguranță:** un document cu `<img onerror=...>` și `<script>` în nume și în text apare ca text inert în listă, în răspuns și în pasaje;
  niciun element injectat, niciun cod executat.
- **Imaginea Docker** conține pagina și o servește (`GET /` → 200, cu antetele).
- **Pe laptopul meu:** pagina rulează; o frază care nu are legătură cu documentele (am scris în caseta de întrebare „pagina merge”) a fost
  **refuzată de poartă** (`too_far`, distanța 1,000, 0 tokeni), exact comportamentul proiectat. `pytest`: 61 de teste, 91 %; `ruff` curat.

**Neverificat:** răspunsuri cu modelul real în pagină; fereastra reală de alegere a fișierelor (am simulat selecția din cod); alte browsere;
accesibilitate cu cititor de ecran; autentificare reală (cheia se tastează).

### Capcane
- **Butonul Delete șterge pe bune.** La o încercare am șters două documente din baza de dezvoltare (rămăseseră 3); se readuc prin reîncărcare
  (butonul Upload sau `cli ingest sample-docs`). Confirmarea din browser nu e o plasă de siguranță.
- **Cheia în pagină nu e un login:** oricine are cheia poate face orice poate face API-ul.
- **Testul de gardă a picat din prima** pentru că un *comentariu* din pagină conținea cuvântul `innerHTML`; l-am reformulat. Un test care caută un șir
  prinde și comentarii.
- **Adăugarea de text cu un script într-un fișier cu sfârșituri de rând CRLF** poate amesteca LF și CRLF; `ruff format` o semnalează. Editorul tău păstrează
  sfârșitul de rând al fișierului, deci nu apare în fluxul normal.

### Cum o spun la interviu
- „Am adăugat o pagină mică, un singur fișier fără framework, servită de API, ca să pot arăta proiectul. Tot ce vine de la server se afișează cu
  `textContent`, am verificat cu un document care conține HTML rău intenționat, iar pagina are o Content-Security-Policy strictă. Am și un test care
  pică dacă apare `innerHTML` în ea.”
- „Butonul «Show passages» arată ce a găsit căutarea, cu distanțe și scoruri, fără să apeleze modelul: e primul loc unde mă uit când un răspuns pare greșit.”
- „Limite pe care le știu: nu are login (cheia se tastează) și nu folosește streaming-ul.”

---

## Încheiere — verificarea finală (Pasul 44) · 2026-10-17

### Ce am verificat
- **Secrete:** `.env` e ignorat de git și nu e urmărit; **0** potriviri cu șiruri de tip cheie (OpenAI `sk-`, GitHub, AWS, chei private, chei Azure cu valoare) în fișierele
  urmărite și în **tot istoricul**; `gitleaks` pe cele 15 commit-uri existente: **no leaks found**. În repo rămân doar valorile de dezvoltare: `dev-key` și parola `docuask` a bazei locale.
- **Date personale:** niciun nume, e-mail sau cale locală în fișierele urmărite. Adresa de e-mail a autorului apare în metadatele commit-urilor, deci e publică pe GitHub.
- **CI:** rularea #4, pe commit-ul cu pagina (`232aa21`), verde: `test` și `image`.
- **Stare finală:** 61 de teste, acoperire ~91 %, `ruff` curat, imagine Docker ~295 MB non-root, README cu cifrele reale.

### Modificări de încheiere
- `.env.example`: comentariul de la `MAX_DISTANCE` cu valorile măsurate (0,72) și o notă că `dev-key` e doar pentru dezvoltare locală.
- `README.md`: în „Known limits”, cheia implicită `dev-key` (nimic n-o blochează la pornire).

### Neverificat la final
Azure OpenAI pe un endpoint real; latențe și comportament sub încărcare; PDF-uri reale complexe sau scanate; alte limbi; scanare de vulnerabilități a imaginii; OCR pentru poze.

### Rămase opționale
Tie-break după rangul vectorial în RRF (de implementat și remăsurat); rulare pe Azure; OCR.

---

## Extra 2 — server MCP (FastMCP) · 2026-10-25

### De ce
Serverele MCP peste un RAG sunt o cerință tot mai des întâlnită în rolurile de AI engineering. DocuAsk avea deja un RAG măsurat; îi lipsea ușa
pentru agenți. MCP (Model Context Protocol) e protocolul prin care un client AI (Claude, Copilot, un agent) **descoperă** tool-urile unui
server și le apelează singur: REST e pentru programe, MCP pentru modele.

### Ce am făcut
- `security.py`: regula de verificare a cheii scoasă în `key_is_valid`, folosită și de REST (`X-API-Key`) și de MCP (bearer token): o singură regulă,
  nu două copii.
- `mcp_server.py` (nou): `create_mcp_server(rag, sessions, auth)` cu două tool-uri, `ask_documents` și `search_documents`, și un resource,
  `docuask://documents`. Tool-urile cheamă **același** `RagService` ca REST-ul. Sunt marcate read-only / idempotent, argumentele au limite (`Field`
  devine JSON Schema), întorc output structurat, iar erorile neprevăzute sunt mascate (`mask_error_details`). `ApiKeyVerifier` acceptă cheile din
  `API_KEYS` ca bearer token.
- `pyproject.toml`: dependența `fastmcp>=4.1.3,<5` și comanda `docuask-mcp` (transportul stdio).
- `main.py`: serverul MCP montat la `/mcp/`, cu lifespan-ul lui pornit în lifespan-ul aplicației.
- `tests/test_mcp.py`: 10 teste: în memorie (descoperire, scheme, răspuns cu citări, refuz, search, validare, eroare mascată, resource), pe HTTP
  real cu uvicorn într-un thread (cheie bună → răspuns, fără cheie sau cu cheie greșită → 401) și montarea din `main.py`.

### Ce am verificat
- `ruff check` + `ruff format --check` curate; **71 de teste** (61 + 10), acoperire **91 %**.
- **MCP Inspector 2.12, stdio:** conectat (protocol `2025-11-25`), cele 2 tool-uri listate; o întrebare acoperită de `01-northwind-benefits.pdf`
  (pensia) → `refused: false`, citare pagina 2, `best_distance` 0,64; o întrebare despre concediul medical, cu doar acel PDF în bază →
  `not_covered`, corect.
- Înainte, pe o copie de referință: drum stdio complet cu un client MCP real (răspuns, refuz, search, resource), HTTP pe aplicația reală
  (cheie bună → răspuns, cheie greșită → 401), imagine Docker **361 MB** (de la 295 MB).

### Neverificat
Clienții Claude Desktop / Claude Code; HTTP în Inspector; modelul real (totul a rulat cu `LLM_PROVIDER=fake`); OAuth / Entra ID în loc de chei
statice.

### Capcane
- **Fișierul pus în rădăcina repo-ului** (`mcp_server.py` lângă `pyproject.toml`) → `ModuleNotFoundError: docuask.mcp_server`. Pachetul e doar ce stă
  în `src/docuask/`; mutat cu `git mv`.
- **Importuri adăugate peste cele vechi** în `main.py` → ruff F811 (nume importat de două ori). Blocul se înlocuiește, nu se completează.
- **URL-ul HTTP e `/mcp/` cu slash:** `POST /mcp` primește 307. Clientul FastMCP urmează redirect-ul; alți clienți s-ar putea să nu-l urmeze.
- **Pe stdio, stdout e canalul protocolului:** un `print()` strică clientul; log-urile merg pe stderr.
- **MCP SDK v2 a trecut câmpurile pe snake_case** (`read_only_hint`, `input_schema`); variantele camelCase merg încă, dar cu avertismente.
- Un resource care întoarce modele Pydantic pică („not JSON serializable”): îl serializăm noi, cu `TypeAdapter(...).dump_json`.
- Conectarea stdio durează ~8 s prima dată (`uv run` + importurile LangChain); apelurile de după sunt rapide.

### Cum o spun la interviu
- „DocuAsk are două uși către același serviciu: REST pentru programe și MCP pentru agenți. Claude sau Copilot descoperă singuri `ask_documents` și
  `search_documents` și le apelează; răspunsul e identic, cu aceleași citări și aceleași refuzuri.”
- „Tool-urile sunt read-only, cu argumente validate prin JSON Schema, cu erorile mascate și cu aceleași chei ca API-ul, trimise ca bearer token. Pentru
  producție aș înlocui verificatorul de chei cu Entra ID, fără să ating tool-urile.”
- „Tool-urile sunt sincrone, pentru că stratul de servicii e sincron, iar FastMCP le rulează într-un thread, deci un apel lent la model nu blochează
  event loop-ul.”
