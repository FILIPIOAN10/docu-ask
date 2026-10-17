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
