# Password-protected public deployment

This deployment keeps both native services on the server's host loopback. Nginx
is the only public HTTP entry point and provides HTTPS plus a temporary Basic Auth
gate. The application adds signed anonymous-session isolation, per-session
queue limits, request throttling, exact case/result TTL cleanup, and a hard MPI
watchdog. If a native call exceeds its deadline, the app exits with code 70;
Compose restarts it and startup recovery marks the interrupted job failed while
preserving queued jobs.

## Provider-limited HTTP preview

If the hosting provider blocks inbound 80/443/8080, use
`nginx-http-preview.conf` on TCP 8888 and set
`DEMO_PUBLIC_ORIGIN=http://36.103.234.95:8888`. The Compose template explicitly
sets `DEMO_ALLOW_INSECURE_PUBLIC_HTTP=1`, which removes only the `Secure` flag
from the signed anonymous-session cookie so job ownership still works over
HTTP. `HttpOnly` and `SameSite=Strict` remain enabled. Do not configure Basic
Auth on this plaintext listener and do not accept confidential, proprietary,
personal, or otherwise sensitive uploads. This is a preview fallback, not an
equivalent replacement for HTTPS.

The paper project page remains on GitHub Pages at
`https://leimingcheng.github.io/surrogate-newton-cfd/`. Only the interactive
demo and API use the compute server: the demo URL is
`http://36.103.234.95:8888/demo`, and requests to the server root or
`/index.html` redirect to GitHub Pages. The same separation is preserved by the
future HTTPS configuration.

The deployment is addressed directly as `https://36.103.234.95`; it does not
depend on a purchased domain, an external wildcard-DNS service, an SSH tunnel,
or the development Mac. Let's Encrypt IP-address certificates use the
`shortlived` profile and expire after roughly six days, so a working server-side
renewal timer and reload hook are release requirements.

Install Nginx, `apache2-utils`, and Certbot 5.4 or newer on the server. Generate
the password file on the server, keeping the plaintext password outside Git:

```bash
sudo apt-get install -y nginx apache2-utils
sudo snap install certbot --classic
sudo install -d -o root -g root -m 0755 /data/surrogate-newton/acme-webroot
sudo install -d -o root -g root -m 0700 /data/surrogate-newton/secrets
sudo htpasswd -cB /data/surrogate-newton/secrets/demo.htpasswd demo
```

Copy `nginx-http-bootstrap.conf` to `/etc/nginx/sites-available/surrogate-newton`,
enable it, and remove the default site. Permit inbound TCP 80 at both the cloud
security group and UFW before requesting the first certificate:

```bash
sudo ln -s /etc/nginx/sites-available/surrogate-newton /etc/nginx/sites-enabled/surrogate-newton
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
sudo certbot certonly --non-interactive --agree-tos \
  --email leimc22@mails.tsinghua.edu.cn \
  --preferred-profile shortlived --webroot \
  --webroot-path /data/surrogate-newton/acme-webroot \
  --ip-address 36.103.234.95
```

Replace the bootstrap site with `nginx-public.conf`, validate it, and enable
server-side automatic renewal. The deploy hook reloads Nginx after each renewal:

```bash
sudo nginx -t && sudo systemctl reload nginx
sudo certbot reconfigure --cert-name 36.103.234.95 \
  --preferred-profile shortlived --webroot \
  --webroot-path /data/surrogate-newton/acme-webroot \
  --deploy-hook '/usr/bin/systemctl reload nginx'
sudo certbot renew --dry-run
systemctl list-timers 'snap.certbot.renew*'
```

On the server, create a root-owned deployment directory and copy this directory
there. Create `public.env` from the example and generate both a long Basic Auth
password and a stable application session secret. Keep both secrets out of Git:

```bash
umask 077
head -c 48 /dev/urandom > /data/surrogate-newton/secrets/demo-session-secret
openssl rand -base64 24 > /data/surrogate-newton/secrets/demo-basic-auth-password
sudo sh -c 'htpasswd -niB demo < /data/surrogate-newton/secrets/demo-basic-auth-password > /data/surrogate-newton/secrets/demo.htpasswd'
```

The runtime image uses fixed UID/GID `10001:10001`. Create the persistent
runtime directory before first launch and give that exact account ownership;
otherwise prewarm will fail before the HTTP listener starts:

```bash
sudo install -d -o 10001 -g 10001 -m 0775 /data/surrogate-newton/runtime/current
```

Set the host paths and immutable image tag outside the env file, validate the
rendered Compose model, then start it:

```bash
export RUNTIME_IMAGE=surrogate-newton-cfd-runtime:release-tag
export MODEL_DIR=/data/surrogate-newton/model-release
export RUNTIME_DIR=/data/surrogate-newton/runtime/current
export UIUC_DIR=/data/surrogate-newton/demo-assets/uiuc
export OOD_DIR=/data/surrogate-newton/demo-assets/ood
export SESSION_SECRET_FILE=/data/surrogate-newton/secrets/demo-session-secret
docker compose -f compose.public.yaml config --quiet
docker compose -f compose.public.yaml up -d --no-build
```

Before replacing a release, keep its immutable image locally or publish it to a
private registry. Permit inbound TCP 80/443 and SSH only; do not expose 18082, 65432,
8429, 9100, or 9835. Keep the provider security group and the host firewall in
agreement. Verify `/api/health/ready`, a complete mesh/predict/recover workflow,
restart recovery, case expiry, log rotation, disk free space, and a 12-client
queue test before sharing the URL.

Rollback is an image-tag change: restore the prior `RUNTIME_IMAGE`, run
`docker compose ... up -d --no-build`, and verify readiness. The persistent
runtime directory must be backed up before a schema-changing release.
