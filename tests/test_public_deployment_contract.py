"""Static contracts for the direct-public-IP deployment."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPO_ROOT / "deployment/public"


class PublicDeploymentContractTests(unittest.TestCase):
    def test_public_env_uses_direct_ip(self) -> None:
        env = (PUBLIC_ROOT / "public.env.example").read_text(encoding="utf-8")
        self.assertIn("DEMO_PUBLIC_HOST=36.103.234.95", env)
        self.assertIn("DEMO_PUBLIC_ORIGIN=https://36.103.234.95", env)
        self.assertNotIn("sslip.io", env)

    def test_nginx_bootstrap_only_serves_acme_over_http(self) -> None:
        config = (PUBLIC_ROOT / "nginx-http-bootstrap.conf").read_text(encoding="utf-8")
        self.assertIn("listen 80", config)
        self.assertIn("/.well-known/acme-challenge/", config)
        self.assertIn("root /data/surrogate-newton/acme-webroot", config)
        self.assertIn('return 503 "HTTPS certificate provisioning is in progress.', config)
        self.assertNotIn("proxy_pass", config)

    def test_nginx_public_proxy_contract(self) -> None:
        config = (PUBLIC_ROOT / "nginx-public.conf").read_text(encoding="utf-8")
        self.assertIn("listen 443 ssl", config)
        self.assertIn("/etc/letsencrypt/live/36.103.234.95/fullchain.pem", config)
        self.assertIn("auth_basic_user_file", config)
        self.assertIn("limit_req zone=demo_per_ip", config)
        self.assertIn("client_max_body_size 2m", config)
        self.assertIn("server 127.0.0.1:18082", config)
        self.assertIn("proxy_set_header X-Forwarded-Proto https", config)

    def test_compose_keeps_native_ports_private_and_mounts_all_assets(self) -> None:
        compose = yaml.safe_load(
            (PUBLIC_ROOT / "compose.public.yaml").read_text(encoding="utf-8")
        )
        services = compose["services"]
        demo = services["demo"]
        self.assertEqual(demo["network_mode"], "host")
        self.assertEqual(demo["environment"]["DEMO_WEB_HOST"], "127.0.0.1")
        self.assertEqual(demo["environment"]["DEMO_SURROGATE_HOST"], "127.0.0.1")
        self.assertEqual(demo["environment"]["DEMO_HEAVY_JOB_CONCURRENCY"], "1")
        self.assertEqual(demo["environment"]["DEMO_MPI_RANKS"], "8")
        self.assertTrue(any("OOD_DIR" in mount for mount in demo["volumes"]))
        self.assertNotIn("ports", demo)
        self.assertNotIn("caddy", services)


if __name__ == "__main__":
    unittest.main()
