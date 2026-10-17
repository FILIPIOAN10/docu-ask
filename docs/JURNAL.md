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
