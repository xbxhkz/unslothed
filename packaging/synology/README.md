# Unslothed on a Synology NAS

Runs as a **Container Manager project**, not an SPK. The SPK route is blocked by
DSM ordering: Container Manager's docker-project resource worker runs *before* a
package's `postinst`, so a `postinst`-written `.env` never lands in time and any
`${VAR}` in the compose file expands to an empty string. `compose.yaml` beside
this file therefore uses **literal values only**.

Image: `xbxhkz/unslothed:cpu-c39f46e` (public on Docker Hub, ~1.9 GB compressed).

`compose.yaml` pins an exact tag rather than floating `:cpu`. Container Manager
reuses a cached `:cpu` it has already pulled, so after a Docker Hub update a
Reset can silently keep the OLD image; a tag it has never seen cannot come from
cache. `cpu-c39f46e` is the first build containing a llama.cpp runtime -- before
it, every GGUF failed with *"no executable llama.cpp runtime (llama-server) is
available"*.

---

## 1. Create the data directory first

DSM's Container Manager does **not** auto-create bind-mount source directories
the way plain `docker run` does. SSH into the NAS and run:

```sh
sudo mkdir -p /volume1/docker/unslothed
```

Skip it and the container dies with
`Bind mount failed: /volume1/docker/unslothed does not exist`.

If `/volume1` is not your volume, change it in **both** the command above and
`compose.yaml`.

**No `chown` is needed for this image**, unlike `xbxhkz/assist`. This one has no
`USER` directive and its entrypoint drops no privileges — verified by running
it: `id` reports `uid=0(root)`, and it creates `/data/auth` as
`drwx------ root root`. So the "unable to open database file" failure that the
Assist deployment hit does not apply here.

The consequence worth knowing: everything under that path ends up **root-owned**,
so browsing or deleting it in File Station needs an admin account. That is also
the trade-off of the container running as root — acceptable on a private LAN
NAS, worth a second thought if this port is ever exposed beyond it.

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
directory exists (step 1). Ownership is not the issue for this image -- it runs
as root.

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

Edit the `image:` line in `compose.yaml` to the new `cpu-<sha>` tag, then
**Project → Action → Build** (or Stop → Build → Start) so Container Manager
re-reads the file and pulls the new tag.

```sh
# optional: pre-pull on the NAS so the project start is not waiting on a download
docker pull xbxhkz/unslothed:cpu-<sha>
```

Everything under `/data` persists across updates; the image carries no state, so
rolling back is just pinning the previous `cpu-<sha>` tag again.

Avoid switching the pin back to floating `:cpu` -- that is precisely the case
where Container Manager can serve a stale cached image and leave you debugging a
bug that was already fixed.
