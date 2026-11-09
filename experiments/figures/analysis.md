# Measured experiment summary

Input: `final.summary.json`

SHA256: `9aaab61011223981488a0eb885fd9506336760f68e8bb52169c2adfc4ad8203e`

Each cell is the median of separately prepared repetition medians. Error bars show population
standard deviation of those run medians, not confidence intervals.

| Backend | N | Runs | Host ms | GPU command buffer ms |
| --- | ---: | ---: | ---: | ---: |
| accelerate | 64 | 3 | 0.000958 | unavailable |
| accelerate | 128 | 3 | 0.00475 | unavailable |
| accelerate | 256 | 3 | 0.02875 | unavailable |
| accelerate | 512 | 3 | 0.121313 | unavailable |
| cpu | 64 | 3 | 0.145645 | unavailable |
| cpu | 128 | 3 | 1.54402 | unavailable |
| cpu | 256 | 3 | 15.0855 | unavailable |
| cpu | 512 | 3 | 139.276 | unavailable |
| metal | 64 | 3 | 0.212813 | 0.046125 |
| metal | 128 | 3 | 0.250229 | 0.088125 |
| metal | 256 | 3 | 0.496812 | 0.317292 |
| metal | 512 | 3 | 0.952688 | 0.726646 |
| metal-tiled | 64 | 3 | 0.17375 | 0.018875 |
| metal-tiled | 128 | 3 | 0.208146 | 0.035375 |
| metal-tiled | 256 | 3 | 0.348354 | 0.184542 |
| metal-tiled | 512 | 3 | 0.624333 | 0.429 |
| mps | 64 | 3 | 0.161834 | 0.0245833 |
| mps | 128 | 3 | 0.185896 | 0.0371042 |
| mps | 256 | 3 | 0.180875 | 0.026625 |
| mps | 512 | 3 | 0.249542 | 0.0652708 |

## Naive and tiled comparison

- N=64: naive/tiled host latency ratio 1.225x.
  Naive/tiled GPU command-buffer ratio 2.444x.
- N=128: naive/tiled host latency ratio 1.202x.
  Naive/tiled GPU command-buffer ratio 2.491x.
- N=256: naive/tiled host latency ratio 1.426x.
  Naive/tiled GPU command-buffer ratio 1.719x.
- N=512: naive/tiled host latency ratio 1.526x.
  Naive/tiled GPU command-buffer ratio 1.694x.

Ratios above one mean the tiled variant had lower latency in this experiment.
The host/GPU difference does not isolate exact dispatch overhead. Workloads use
fixed uploaded inputs, output readback, warmup, and sequential backend runs.
Thermals and background activity were not controlled; this is one machine and
one experiment, not a universal speedup or an accelerator power measurement.

Repetitions execute within one process and are not assumed statistically independent.
