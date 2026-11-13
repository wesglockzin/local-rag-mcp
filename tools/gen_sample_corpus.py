#!/usr/bin/env python3
"""Generate the sample corpus — six fictional platform-team documents that
exercise every ingest behavior: whole-file embedding, heading-split chunking
(`rag_chunk: headings`), exclusion (`rag: false`), and the superseded
lifecycle filter. Committed so the quickstart works offline of this script;
CI checks the committed files match this generator."""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "sample-corpus")

DOCS = {
    "reference_oncall_escalation.md": """---
name: reference_oncall_escalation
description: "On-call escalation order and the paging threshold"
metadata:
  type: reference
---

Sev-1 pages the primary on-call immediately; if unacknowledged in 15 minutes
it escalates to the secondary, then the team lead. Sev-2 waits for business
hours. The paging threshold for the API error-rate monitor is 5% over five
minutes — below that it only alerts the channel.
""",
    "feedback_deploy_in_phases.md": """---
name: feedback_deploy_in_phases
description: "Ship risky changes in phases with checkpoints, never as one batch"
metadata:
  type: feedback
---

Rolling out the payment-service config change to all regions at once caused
the March incident. Rule: non-trivial live changes ship in phases — canary
region first, verify dashboards, then the rest. A bad batch is a full
outage; a bad canary is a blip.
""",
    "reference_ca_bundle.md": """---
name: reference_ca_bundle
description: "A TLS-inspecting proxy means tools need a custom CA bundle"
metadata:
  type: reference
---

A TLS-inspecting proxy re-signs traffic. Python tools must set
REQUESTS_CA_BUNDLE to the custom bundle or requests fail with
certificate errors that look like outages. Node tools need
NODE_EXTRA_CA_CERTS pointed at the same file.
""",
    "reference_retired_worker_pool.md": """---
name: reference_retired_worker_pool
description: "RETIRED: the legacy worker pool was decommissioned"
lifecycle: superseded
metadata:
  type: reference
---

The legacy cron worker pool is gone. All background jobs run on the queue
service with per-job retry policies. This file is kept as the historical
record of the cutover — the lifecycle marker keeps it out of search results.
""",
    "notes_scratch_do_not_index.md": """---
name: notes_scratch_do_not_index
description: "Scratch notes — excluded from the index"
rag: false
metadata:
  type: reference
---

Raw meeting scratch. The `rag: false` frontmatter keeps this file out of the
index entirely; if you see it in search results, exclusion is broken.
""",
    "runbook_incident_response.md": """---
name: runbook_incident_response
description: "Incident response runbook — heading-split into chunks"
rag_chunk: headings
metadata:
  type: reference
---

# Incident response runbook

## Declare

Declare an incident when user-facing error rate exceeds the paging threshold
or when two independent alerts fire for the same subsystem within ten
minutes. Name an incident commander immediately; the IC coordinates and does
not debug.

## Communicate

Open the incident channel, post the impact statement within ten minutes, and
update every thirty. External status-page updates go out only through the
communications lead, never directly from responders.

## Mitigate

Prefer the reversible mitigation first: feature-flag off, traffic shift,
rollback. Root-causing happens after user impact stops. Any mitigation that
touches data requires a second responder's ack before it runs.

## Review

Every incident gets a blameless review within five working days: timeline,
contributing factors, and at most three action items with owners. An action
item without an owner is a wish, not an action.
""",
}


def main() -> None:
    os.makedirs(CORPUS, exist_ok=True)
    # Remove any .md not owned by this generator, so the CI drift check also
    # catches EXTRA committed files, not just modified ones.
    for existing in os.listdir(CORPUS):
        if existing.endswith(".md") and existing not in DOCS:
            os.remove(os.path.join(CORPUS, existing))
    for name, body in DOCS.items():
        with open(os.path.join(CORPUS, name), "w", encoding="utf-8") as f:
            f.write(body)
    logging.getLogger(__name__).info("wrote %d documents -> %s", len(DOCS), CORPUS)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    main()
