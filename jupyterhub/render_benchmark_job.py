#!/usr/bin/env python
"""Render the one-shot L4 candidate benchmark Job from validated inputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from string import Template

import yaml

DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="fully qualified image@sha256 digest")
    parser.add_argument("--namespace", default="jupyterhub")
    parser.add_argument("--service-account", default="earth2-attendee")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("model-benchmark-job.yaml.tmpl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / ".generated/model-benchmark-job.yaml",
    )
    return parser.parse_args()


def _dns_label(name: str, field: str) -> str:
    if len(name) > 63 or not DNS_LABEL_RE.fullmatch(name):
        raise ValueError(f"{field} must be a valid DNS label")
    return name


def main() -> int:
    args = parse_args()
    if not DIGEST_RE.fullmatch(args.image):
        raise ValueError("--image must be a fully qualified image@sha256 digest")

    rendered = Template(args.template.read_text()).substitute(
        CANDIDATE_IMAGE_DIGEST=args.image,
        NAMESPACE=_dns_label(args.namespace, "namespace"),
        KUBERNETES_SERVICE_ACCOUNT=_dns_label(
            args.service_account, "service account"
        ),
    )
    document = yaml.safe_load(rendered)
    if document.get("kind") != "Job":
        raise ValueError("Rendered document is not a Kubernetes Job")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
