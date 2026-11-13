---
name: reference_ca_bundle
description: "A TLS-inspecting proxy means tools need a custom CA bundle"
metadata:
  type: reference
---

A TLS-inspecting proxy re-signs traffic. Python tools must set
REQUESTS_CA_BUNDLE to the custom bundle or requests fail with
certificate errors that look like outages. Node tools need
NODE_EXTRA_CA_CERTS pointed at the same file.
