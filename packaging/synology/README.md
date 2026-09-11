# Unslothed on a Synology NAS

Runs as a **Container Manager project**, not an SPK. The SPK route is blocked by
DSM ordering: Container Manager's docker-project resource worker runs *before* a
package's `postinst`, so a `postinst`-written `.env` never lands in time and any
`${VAR}` in the compose file expands to an empty string. `compose.yaml` beside
this file therefore uses **literal values only**.

Image: `xbxhkz/unslothed:cpu` (public on Docker Hub, ~8.2 GB).

---

## 1. Create the data directory first

DSM's Container Manager does **not** auto-create bind-mount source directories
the way plain `docker run` does, and the container drops to UID/GID 1000. Both
of these have to be right before the first start or it fails in ways that do not
name the cause.

SSH into the NAS and run:

```sh
sudo mkdir -p /volume1/docker/unslothed
sudo chown -R 1000:1000 /volume1/docker/unslothed
```

| Skipped step | What you get instead |
|---|---|
| directory missing | `Bind mount failed: /volume1/docker/unslothed does not exist` |
| wrong owner | `sqlite3.OperationalError: unable to open database file` |

If `/volume1` is not your volume, change it in **both** the command above and
`compose.yaml`. DSM shared-folder ACLs can override POSIX ownership — if `chown`
looks correct and it still cannot write, check the shared folder's permissions
in DSM rather than re-running `chown`.

## 2. Create the project

1. **Container Manager → Project → Create**
2. Project name: `unslothed`
3. Path: `/volume1/docker/unslothed`
4. Source: **Upload docker-compose.yml**, and upload `compose.yaml`
5. Skip the web-portal step (the compose file already publishes the port)
6. **Done** — it pulls ~8.2 GB on first run, so give it time

## 3. First login

Open `http://<nas-ip>:8800`.

With no password set, Studio generates one on first run and prints it to the
log — **Container Manager → Project → unslothed → Log**, near the top of the
first start. To set your own instead, uncomment the `command:` line in
`compose.yaml` before creating the project:

```yaml
    command: ["--password", "your-password-here"]
```

Then remove the line and recreate the project once you have logged in, so the
password is not sitting in the compose file.

---

## What this NAS is good at, and what it is not

Your NAS (dual Xeon Gold 6130, 32 cores / 64 threads, 320 GB RAM) has **no
usable GPU for this**: DSM has no NVIDIA Container Toolkit, so `--gpus` binds
nothing, and the Quadro P1000's 4 GB would be marginal even if it did. Everything
below runs on CPU.

| Workload | Verdict |
|---|---|
| **LLM inference (GGUF / llama.cpp)** | **Best use of this box.** 32 cores is real throughput, and 320 GB fits models no consumer GPU can hold. |
| Chat, RAG, code tools | Fine — barely compute-bound. |
| Embeddings / vector search | Fine. |
| Image generation | Works, but minutes per image. |
| LoRA / fine-tuning | Possible, very slow. Prefer a GPU machine. |
| **Video generation (Wan2.2)** | **Don't.** Measured at 76 s/step *on a GPU*; CPU is 50–100× that, i.e. days per clip. |

The NAS is the better machine for large-model **text** inference. A GPU machine
stays better for anything diffusion.

`UNSLOTH_CPU_THREADS: "32"` in the compose file pins one pool per physical core.
Left unset, the BLAS/OpenMP pools size to all 64 threads and thrash. Lower it if
other containers share the NAS.

---

## Troubleshooting

**Container restart-loops immediately.** Almost always the bind mount. Check the
directory exists and is `1000:1000` (step 1).

**UI loads but login fails.** The data root was created root-owned on first
start. Stop the project, `chown -R 1000:1000 /volume1/docker/unslothed`, start.

**Port 8800 already in use.** Change the host side of `"8800:8000"` only — the
container side must stay 8000, which is what the image's healthcheck probes.

**Health shows unhealthy for the first few minutes.** Expected. The image allows
a 180 s start period; first run initialises the data root.

**Logs.** Container Manager's log viewer shows stdout. Deeper Studio logs live
inside the volume at `/volume1/docker/unslothed/logs/server/`.

**A ttyd web terminal on the NAS is itself a container** and cannot see DSM's
own logs. Use real SSH for anything under `/var/log/`.

---

## Updating

```sh
# on the NAS, or via Container Manager's "Reset" on the project
docker pull xbxhkz/unslothed:cpu
```

then **Project → Action → Reset** (or Stop/Start) to recreate the container.
Everything under `/data` persists across updates; the image carries no state.

`xbxhkz/unslothed:cpu-<sha>` tags pin an exact build if you need to roll back.
