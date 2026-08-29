# Deployment runbook

Everything needed to take Medical Embeddings Search from a laptop to Azure.

> **Status: none of this has been deployed.** The configuration here is written
> and reviewed but never applied — no Azure subscription was available during
> development. Treat it as a reviewed plan, not a verified deployment, and
> expect to correct details on first run. See `Memory.md`.

---

## Contents

```
deploy/
├── docker/
│   ├── Dockerfile          multi-stage, non-root, slim
│   └── compose.yaml        local run under a 2 GB limit
└── azure/
    ├── data-factory/
    │   ├── linked-services/   managed identity, no sasUri literals
    │   ├── pipelines/         train-embeddings
    │   └── triggers/          on-new-corpus (BlobEventsTrigger)
    ├── databricks/
    │   ├── run_training.py    Spark Python task
    │   └── run_indexing.py    Spark Python task
    └── app-service/
        └── site-config.json
```

---

## 0. Before anything else: revoke the legacy credentials

The predecessor project committed live Azure SAS tokens into
`reference/legacy/azure-pipeline/src/read_data.py` and `top_n.py`, carrying
`sp=racwdymeop` — read, add, create, **write**, **delete**. The storage account
name and subscription id were committed alongside them.

The tokens expired on 2021-12-31. **Expiry is not revocation.** If the storage
account still exists:

```bash
# Rotate the account keys the SAS was signed with -- this invalidates
# every SAS derived from them, expired or not.
az storage account keys renew --account-name <ACCOUNT> --key primary
az storage account keys renew --account-name <ACCOUNT> --key secondary
```

Then confirm no stored access policy still grants those rights:

```bash
az storage container policy list --container-name <CONTAINER> --account-name <ACCOUNT>
```

Nothing in this repository contains a credential. CI enforces that with a
secret scan on every push, and a pre-commit hook blocks the same patterns
locally.

---

## 1. Local container

```bash
docker compose -f deploy/docker/compose.yaml up --build
open http://localhost:8501
```

The compose file caps the container at **2 GB** deliberately: it reproduces the
App Service tier and proves the serving path stays inside the 1.2 GB budget in
[Architecture.md §9](../Architecture.md#9-resource-budget), rather than only
fitting on an unconstrained machine.

Artefacts are **mounted, not baked**. The image carries code and NLTK corpora
only, so a retrain rolls out without an image rebuild.

Build on the dev laptop with care — `C:` had 34.7 GB free at last check.
`docker system prune` between builds.

---

## 2. Storage layout

One container. The layout is **not** free-form -- it is what
`Settings.paths` resolves to, so the training job and the app agree without
either translating:

```
medsearch/                                 <- mounted read-only at /mnt/artefacts
├── data/
│   ├── raw/       dimension-covid.csv     <- drop new corpora here; triggers retraining
│   ├── interim/   token caches
│   └── processed/ skipgram-abstract/, ... <- vectors.npy + manifest.json
└── models/        skipgram/, fasttext/    <- model.kv + metadata.json
```

`run_training.py` builds `Settings(data_dir=f"{mount}/data",
model_dir=f"{mount}/models")`; App Service sets `MEDSEARCH_DATA_DIR` and
`MEDSEARCH_MODEL_DIR` to the same two paths under its blob mount. Change one
and you must change the other -- there is no indirection between them.

> **This was wrong until the pre-deployment audit.** This section documented a
> top-level `index/` prefix that nothing writes and nothing reads, and the App
> Service config declared no mount at all, so a deployment would have started
> cleanly, passed its health check, and returned zero results for every query.
> Neither error was reachable without deploying, which is the argument for
> doing so before calling Sprint 10 done.

Apply a lifecycle policy to `interim/` — token caches are reproducible and need
no retention.

---

## 3. Identity and access

Auth is managed identity end to end. **No component holds a credential.**

| Principal | Role | Scope |
|-----------|------|-------|
| Data Factory (system-assigned) | Storage Blob Data Reader | the container |
| Databricks (system-assigned) | Storage Blob Data Contributor | the container |
| App Service (system-assigned) | Storage Blob Data Reader | `models/`, `index/` |

```bash
az role assignment create \
  --assignee-object-id "$(az webapp identity show -g <RG> -n medsearch-app --query principalId -o tsv)" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Reader" \
  --scope "/subscriptions/<SUB>/resourceGroups/<RG>/providers/Microsoft.Storage/storageAccounts/<ACCOUNT>"
```

---

## 4. Databricks

Training moves off the laptop here. This is the intended path for full-corpus
runs — the 10,666-document run needs ~2 GB free, which an 8 GB dev machine
cannot reliably provide with a browser open.

```bash
python -m build --wheel
databricks fs cp dist/medsearch-*.whl dbfs:/FileStore/medsearch/wheels/
databricks fs cp deploy/azure/databricks/run_training.py dbfs:/FileStore/medsearch/jobs/
databricks fs cp deploy/azure/databricks/run_indexing.py dbfs:/FileStore/medsearch/jobs/
```

Mount the container once at the workspace level as `/mnt/medsearch`, using the
cluster's managed identity.

`run_training.py` replaces the legacy `main.py`, whose first line was
`%run ./training_model` — referencing a module **that does not exist anywhere
in the predecessor repository**. That notebook could never have run. Training
logic now lives in the installed wheel, so Databricks, the CLI, and the test
suite all execute the same code path.

Set the cluster's Spark env vars to pin BLAS threads (already in
`linked-services/databricks.json`).

---

## 5. Data Factory

Import in dependency order:

1. `linked-services/blob-storage.json`
2. `linked-services/databricks.json`
3. `pipelines/train-embeddings.json`
4. `triggers/on-new-corpus.json`

Replace every `REPLACE_ME` with real resource identifiers — they are
non-secret (workspace URL, resource id, storage account resource id) and belong
in Data Factory's own configuration, not in this repository.

The trigger starts **stopped**. Start it only after a manual pipeline run
succeeds:

```bash
az datafactory trigger start --resource-group <RG> --factory-name <ADF> --name on-new-corpus
```

The pipeline runs training and indexing as **two separate Spark tasks** on
purpose: the two stages have different peak-memory profiles, and separate
processes let the cluster reclaim training memory before indexing begins.

---

## 6. App Service

```bash
az webapp create -g <RG> -p <PLAN> -n medsearch-app \
  --deployment-container-image-name ghcr.io/<OWNER>/clinical-trial-search:latest
az webapp identity assign -g <RG> -n medsearch-app
az webapp config set -g <RG> -n medsearch-app --generic-configurations @deploy/azure/app-service/site-config.json
```

Use **B2 (3.5 GB)**, not B1. B1's 1.75 GB is uncomfortably close to the 1.2 GB
serving budget once both models are resident.

`WEBSITES_CONTAINER_START_TIME_LIMIT` is raised to 300 s because the container
pulls artefacts from Blob at cold start.

---

## 7. Verify

```bash
curl -f https://medsearch-app.azurewebsites.net/_stcore/health   # -> ok
```

Then drop a CSV into `raw/` and confirm the trigger fires, the Databricks job
completes, and new artefacts appear under `models/` and `index/`.

---

## 8. Rollback

Artefacts are versioned by fingerprint in `metadata.json`, and an index refuses
to load against a model it was not built from — so a bad retrain fails loudly
rather than silently serving wrong results.

To roll back: repoint the App Service image tag to the previous release and
restore the previous `models/` + `index/` prefixes from a blob snapshot.

---

## Open items

| # | Item | Blocker |
|---|------|---------|
| 1 | Revoke legacy SAS tokens (§0) | Needs account access |
| 2 | Apply and verify every step here | Needs a subscription |
| 3 | Replace `REPLACE_ME` placeholders | Needs real resource ids |
| 4 | ~~Confirm the image builds and serves~~ | **Done 2026-08-28.** Both targets build; both serve under `--memory=2g`. Building it found a `PermissionError` on `data/interim` that no test could reach — see Phases.md Sprint 9. The image was 941 MB against a < 800 MB DoD; trimming the venv on 2026-08-29 brought it to **721 MB** (`standalone` 829 MB) — see Phases.md Sprint 9 |

---

## 6. Deploying on the Azure free tier

The configuration in `site-config.json` targets **B2 Basic**, which is not
free. `site-config-free.json` targets **F1 Free**. One structural difference
drives everything else.

### The blocker, and the fix

F1 gives you no storage account, so there is nothing to mount artefacts from —
and the mount is the only reason the container needs anything external. So on
free tier, **bake them in**:

```bash
docker build --target standalone -t medsearch:standalone -f deploy/docker/Dockerfile .
```

That copies `models/`, `data/raw/`, `data/interim/` and `data/processed/` into
the image — **110 MB**, measured. The trade is that a retrain now needs an
image rebuild. On a corpus that changes rarely, that is a better deal than a
storage account you have to pay for.

All four paths are required. `data/interim` is the non-obvious one: without the
token cache, the TF-IDF half of the union re-preprocesses all 10,666 documents
on the first query — ~46 s, on **every cold start**, out of a 60-minute daily
CPU budget.

### Does it actually fit?

| F1 limit | Our measurement | Verdict |
|---|---|---|
| 1,024 MB memory per plan | **438 MB** serving RSS (fasttext + union) | fits, ~58% headroom |
| 1 GB app storage | 110 MB artefacts | fits |
| 60 CPU-minutes/day | ~10 CPU-s per cold start, ~0.13 s per query | fine for demo traffic |
| No Always On | ~10 s cold start after idle | tolerable; see below |
| 1 shared instance | single-user demo | fine |

Free tier **works** for this app. It is not a production posture: there is no
Always On, so the app unloads when idle and the next visitor waits ~10 seconds
while the model loads and the TF-IDF matrix rebuilds. Do not point a load test
or a crawler at it — the daily CPU quota is real, and exceeding it stops the
app until the quota resets.

### Deploy

```bash
az group create --name medsearch-rg --location eastus
az appservice plan create --name medsearch-plan --resource-group medsearch-rg \
  --is-linux --sku F1
az webapp create --resource-group medsearch-rg --plan medsearch-plan \
  --name <globally-unique-name> \
  --deployment-container-image-name ghcr.io/OWNER/clinical-trial-search:standalone
az webapp config appsettings set --resource-group medsearch-rg \
  --name <globally-unique-name> \
  --settings WEBSITES_PORT=8501 MEDSEARCH_LOG_JSON=true \
             MEDSEARCH_DATA_DIR=/home/app/data MEDSEARCH_MODEL_DIR=/home/app/models
```

`alwaysOn` must stay **false** — setting it true is rejected on F1.

### What is *not* free

Only the web app is. The rest of the Sprint 10 pipeline is not, and no
configuration makes it so:

| Component | Free? |
|---|---|
| App Service F1 | **yes**, indefinitely |
| Azure Databricks (training) | **no** — trial only, then billed per DBU |
| Azure Data Factory (orchestration) | **no** beyond a small monthly activity allowance |
| Blob Storage | 5 GB free for 12 months on a new account, then billed |

The free-tier path therefore serves a **pre-trained** image: train locally with
`make train` (2 min 22 s on a 4-core laptop), bake, push, deploy. Databricks
and Data Factory only earn their place once retraining is frequent enough to
need automating.
