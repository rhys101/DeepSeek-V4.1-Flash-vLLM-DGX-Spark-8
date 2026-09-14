# SG18 native prefill — promoted measurements and evidence

**Promoted after the successful capacity repeat.** The cold confirmation
reduced geometric-mean time on the primary long-prompt cells by 21.40–28.05%
against its two controls; the unchanged README suite passed both comparisons.
After Spark1 rebooted, eight distinct 997,100-token documents passed cold and
cached retrieval with all eight contexts active together. The requested
memory guard was 1 GiB; the minimum observed was 3.555 GiB.

- [Current result tables and method](../../docs/sg18-prefill-results.md)
- [Promotion decision, exact headlines and evidence hashes](promotion.json)
- [Capacity repeat measurements](capacity-retest.json)
- [Capacity raw evidence and offline validators](capacity-retest-evidence.zip)
- [Final publication health and source audit](promotion-health.json)
- [Historical evaluation before the successful repeat](evaluation.md)
- [Historical timing trials, comparisons and first capacity failure](measurements.json)
- [Original execution plan](plan.md)

The first attempt failed its 2 GiB memory gate on Spark1 before completing a
retrieval. Its report, measurements, archived decision and rollback evidence
are retained unchanged as historical records. The promotion decision above
includes the later successful repeat; it does not rewrite the failed attempt.
The retest JSON and ZIP also retain their original pre-publication wording.

## Offline reproduction of recorded results

From the repository root, using standard Python 3 without `-O`:

```bash
python3 sglang/experiments/sg18-prefill-tp-split/verify-source.py
python3 sglang/experiments/sg18-prefill-tp-split/verify-results.py
python3 sglang/experiments/sg18-prefill-tp-split/verify-promotion.py
```

The first command verifies the exact production overlays, both patches,
inherited B12x source and 16,404 partition cases. The second verifies every
archive member's hash, creates a temporary copy of the historical directory
layout and replays the original offline validators. It uses the 35 pinned
reference files already in this repository. It runs no benchmark client,
network request, model or GPU program, and it leaves the checkout unchanged.

The README receipts were recorded with Python 3.9. Newer Python versions can
round `statistics.stdev` differently in the last binary digit. The replay
verifies the current calculation and independently reproduces the original
two-pass variance and square root from the identical trial values before
comparing receipts exactly. Raw values, means, thresholds and stored evidence
remain unchanged; no comparison tolerance is widened.

- [recorded-evidence.tar.gz](recorded-evidence.tar.gz) contains complete cold
  and README blocks, the interrupted attempt, frozen prompts, SSE records,
  original clients, raw memory/health samples, comparisons and failure records.
  Original SG18 controls predate the scratch fix and are preserved separately
  from the corrected control blocks used for the performance gates.
- [component-evidence.tar.gz](component-evidence.tar.gz) contains the original
  eight-rank component results and test source, the initial all-zero top-k
  tie failure, its diagnostic, and subsequent qualifications. Binary diagnostic
  tensors are preserved but never loaded by the offline verifier.
- [evidence-manifest.json](evidence-manifest.json) records the archive and member
  hashes, run labels, inclusion status and scope. Source identity is recorded
  separately in the [source snapshot](../../experiments/sg18-prefill-tp-split/).

Full profiler traces remain in the laboratory archive; published dispatch
summaries identify their hashes and observations. CPU replay checks the saved
records and arithmetic, not fresh GPU execution. The pinned prose runner did
not retain absolute first/last aggregate-window endpoints, so its exact
aggregate window cannot be independently reconstructed from those records.
Host addresses and absolute paths inside raw receipts are historical deployment
metadata, not portable deployment instructions.


`verify-promotion.py` checks the promoted source and current headline values
against the preserved confirmation records, verifies the fresh publication
audit, then extracts the capacity ZIP into a temporary directory and replays
its two independent validators. This checks every frozen prompt, response,
usage record, residency sample and recorded all-rank memory sample, and
reproduces the saved validation JSON exactly. The ZIP includes the original
prompt-freeze reference from the failed first attempt.
