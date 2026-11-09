# Measured experiment summary

Input: `large.summary.json`

SHA256: `95ab1c69acdd0b2b98facc22f38ae2fcf77773f692a2385249a9cce430cccc3c`

Each cell is the median of independent run medians. Error bars show population
standard deviation of those run medians, not confidence intervals.

| Backend | N | Runs | Host ms | GPU command buffer ms |
| --- | ---: | ---: | ---: | ---: |
| accelerate | 1024 | 3 | 0.794063 | unavailable |
| accelerate | 2048 | 3 | 7.52429 | unavailable |
| accelerate | 4096 | 3 | 63.2468 | unavailable |
| metal | 1024 | 3 | 6.4299 | 6.12006 |
| metal | 2048 | 3 | 45.0493 | 44.3444 |
| metal | 4096 | 3 | 395.822 | 393.845 |
| metal-tiled | 1024 | 3 | 3.648 | 3.35388 |
| metal-tiled | 2048 | 3 | 27.6794 | 26.9755 |
| metal-tiled | 4096 | 3 | 220.454 | 218.142 |
| mps | 1024 | 3 | 0.717563 | 0.47375 |
| mps | 2048 | 3 | 4.31071 | 3.77217 |
| mps | 4096 | 3 | 32.9989 | 31.0029 |

## Naive and tiled comparison

- N=1024: naive/tiled host latency ratio 1.763x.
  Naive/tiled GPU command-buffer ratio 1.825x.
- N=2048: naive/tiled host latency ratio 1.628x.
  Naive/tiled GPU command-buffer ratio 1.644x.
- N=4096: naive/tiled host latency ratio 1.795x.
  Naive/tiled GPU command-buffer ratio 1.805x.

Ratios above one mean the tiled variant had lower latency in this experiment.
The host/GPU difference does not isolate exact dispatch overhead. Workloads use
fixed uploaded inputs, output readback, warmup, and sequential backend runs.
Thermals and background activity were not controlled; this is one machine and
one experiment, not a universal speedup or an accelerator power measurement.
