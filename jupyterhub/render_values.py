#!/usr/bin/env python3
"""Render and validate the event Helm values from environment variables."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "values-event.yaml.tmpl"
DEFAULT_OUTPUT = ROOT / ".generated" / "values-event.yaml"

REQUIRED = {
    "ENABLE_SIGNUP",
    "EVENT_HOSTNAME",
    "HUB_IMAGE_REFERENCE",
    "INGRESS_IP_NAME",
    "LAB_IMAGE_REFERENCE",
    "KUBERNETES_SERVICE_ACCOUNT",
    "STORAGE_CLASS_NAME",
    "TEMPO_DATA_URI",
    "TLS_CERTIFICATE_NAME",
    "WORKSHOP_ADMIN_USERNAME",
    "WORKSHOP_ALLOWED_USERS",
    "WORKSHOP_RELEASE",
}

DIGEST_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
GCP_RESOURCE_NAME = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def valid_dns_name(value: str) -> bool:
    return len(value) <= 253 and "." in value and all(
        DNS_LABEL.fullmatch(label) for label in value.split(".")
    )


def validate(values: dict[str, str]) -> None:
    missing = sorted(name for name in REQUIRED if not values.get(name, "").strip())
    if missing:
        raise SystemExit("Missing required environment variables: " + ", ".join(missing))

    if values["ENABLE_SIGNUP"].lower() not in {"true", "false"}:
        raise SystemExit("ENABLE_SIGNUP must be true or false")
    values["ENABLE_SIGNUP"] = values["ENABLE_SIGNUP"].lower()

    for name in ("HUB_IMAGE_REFERENCE", "LAB_IMAGE_REFERENCE"):
        if not DIGEST_REFERENCE.fullmatch(values[name]):
            raise SystemExit(f"{name} must be a full image reference ending in @sha256:<64 hex>")
        image_name, image_digest = values[name].rsplit(":", 1)
        prefix = name.removesuffix("_REFERENCE")
        values[f"{prefix}_NAME"] = image_name
        values[f"{prefix}_DIGEST"] = image_digest

    if not USERNAME.fullmatch(values["WORKSHOP_ADMIN_USERNAME"]):
        raise SystemExit("WORKSHOP_ADMIN_USERNAME is not a safe JupyterHub username")

    users = {item.strip() for item in values["WORKSHOP_ALLOWED_USERS"].split(",") if item.strip()}
    if values["WORKSHOP_ADMIN_USERNAME"] not in users:
        raise SystemExit("WORKSHOP_ALLOWED_USERS must include WORKSHOP_ADMIN_USERNAME")
    invalid_users = sorted(user for user in users if not USERNAME.fullmatch(user))
    if invalid_users:
        raise SystemExit("Invalid usernames in WORKSHOP_ALLOWED_USERS: " + ", ".join(invalid_users))

    hostname = values["EVENT_HOSTNAME"]
    if not valid_dns_name(hostname):
        raise SystemExit("EVENT_HOSTNAME must be a lowercase fully qualified DNS name")

    for name in ("INGRESS_IP_NAME", "TLS_CERTIFICATE_NAME"):
        if not GCP_RESOURCE_NAME.fullmatch(values[name]):
            raise SystemExit(f"{name} must be a lowercase Google Cloud resource name")

    for name in ("KUBERNETES_SERVICE_ACCOUNT", "STORAGE_CLASS_NAME"):
        if len(values[name]) > 253 or not all(
            DNS_LABEL.fullmatch(label) for label in values[name].split(".")
        ):
            raise SystemExit(f"{name} must be a lowercase Kubernetes DNS name")

    if not RELEASE.fullmatch(values["WORKSHOP_RELEASE"]):
        raise SystemExit("WORKSHOP_RELEASE contains unsafe characters")

    tempo_uri = values["TEMPO_DATA_URI"]
    if (
        not tempo_uri.startswith("gs://")
        or "/" not in tempo_uri.removeprefix("gs://")
        or any(character.isspace() or character in "\"'\\" for character in tempo_uri)
    ):
        raise SystemExit("TEMPO_DATA_URI must be a safe gs://bucket/path URI")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    values = {name: os.environ.get(name, "") for name in REQUIRED}
    validate(values)
    rendered = Template(args.template.read_text()).substitute(values)
    leftovers = sorted(set(re.findall(r"\$\{[A-Z0-9_]+\}", rendered)))
    if leftovers:
        raise SystemExit("Unresolved template variables: " + ", ".join(leftovers))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
