# Password-protected public deployment

This deployment keeps both native services on host loopback. Caddy is the only
public HTTP entry point and provides automatic HTTPS plus a temporary Basic Auth
gate. The application adds signed anonymous-session isolation, per-session
queue limits, request throttling, exact case/result TTL cleanup, and a hard MPI
watchdog. If a native call exceeds its deadline, the app exits with code 70;
Compose restarts it and startup recovery marks the interrupted job failed while
preserving queued jobs.

Use `36-103-234-95.sslip.io` only for a short password-protected acceptance
stage. It resolves automatically to the embedded IP and is not a domain owned
by this project. A formal public launch on a mainland-China server requires the
applicable ICP filing even when visitors use only the IP address.

On the server, create a root-owned deployment directory and copy this directory
there. Create `public.env` from the example, generate a long Basic Auth password
and its Caddy hash, and create a stable session secret. Keep the bcrypt hash in
single quotes in `public.env` so Compose treats its dollar signs literally:

```bash
umask 077
head -c 48 /dev/urandom > /data/surrogate-newton/secrets/demo-session-secret
docker run --rm caddy:2.11.4-alpine caddy hash-password --plaintext 'choose-a-long-password'
```

Set the host paths and immutable image tag outside the env file, validate the
rendered Compose model, then start it:

```bash
export RUNTIME_IMAGE=surrogate-newton-cfd-runtime:release-tag
export MODEL_DIR=/data/surrogate-newton/model-release
export RUNTIME_DIR=/data/surrogate-newton/runtime/current
export UIUC_DIR=/data/surrogate-newton/demo-assets/uiuc
export SESSION_SECRET_FILE=/data/surrogate-newton/secrets/demo-session-secret
export CADDY_DATA_DIR=/data/surrogate-newton/caddy/data
export CADDY_CONFIG_DIR=/data/surrogate-newton/caddy/config
docker compose -f compose.public.yaml config --quiet
docker compose -f compose.public.yaml up -d --no-build
```

Before opening ports, take an image backup or publish the image to a private
registry. Permit inbound TCP 80/443 and SSH only; do not expose 18082, 65432,
8429, 9100, or 9835. Keep the provider security group and the host firewall in
agreement. Verify `/api/health/ready`, a complete mesh/predict/recover workflow,
restart recovery, case expiry, log rotation, disk free space, and a 12-client
queue test before sharing the URL.

Rollback is an image-tag change: restore the prior `RUNTIME_IMAGE`, run
`docker compose ... up -d --no-build`, and verify readiness. The persistent
runtime directory must be backed up before a schema-changing release.
