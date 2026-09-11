# Unchanged base-stack validation

These files were recorded on 10 September 2026 for base image `sha256:4514e1e972669763b1ceb94b6f15f70e66e6ee8f381a502246217f165b63a342`. The current image retains its source pins, stable extension, attention fixes, model snapshot and package stack, adding one MXFP8 routing file.

`kernel-validation.json` contains the native/MHC, sparse-attention and prefill checks. Package, source and checkpoint records establish provenance. The GPU-state probe and clock samples describe their original measurement windows; they are not fresh measurements of the current serving run.

The added source change and full current deployment are checked separately in [2026-09-11](../2026-09-11/). A separate complete cold rebuild of the combined Dockerfile has not been performed.
