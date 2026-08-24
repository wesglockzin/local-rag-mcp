---
name: reference_retired_worker_pool
description: "RETIRED: the legacy worker pool was decommissioned"
lifecycle: superseded
metadata:
  type: reference
---

The legacy cron worker pool is gone. All background jobs run on the queue
service with per-job retry policies. This file is kept as the historical
record of the cutover — the lifecycle marker keeps it out of search results.
