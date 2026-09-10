# Contributing

Keep patches narrow, versioned, and backed by a behavior or numerical check. Preserve failed checks and explain their limits. Record the exact source/image identity and workload for performance results.

Run local configuration checks with `python3 -m unittest discover -s tests -v`. Ordinary CI does not have DGX Spark hardware. Attach separate GPU and multi-node results when a change affects those paths.

The supported recipe covers eight Sparks with RAM-resident Engram and vision. Disk offload, different node counts, larger context limits and new cache formats require separate validation.
