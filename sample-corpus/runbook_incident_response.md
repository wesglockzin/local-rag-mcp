---
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
