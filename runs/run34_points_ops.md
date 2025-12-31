```bash
Timer precision: 20 ns
concurrent_masstree24                           fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_concurrent_writes_disjoint                              │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      4.681 ms      │ 7.999 ms      │ 5.231 ms      │ 5.517 ms      │ 100     │ 100
│     ├─ 2                                      6.003 ms      │ 10.87 ms      │ 9.237 ms      │ 8.515 ms      │ 100     │ 100
│     ├─ 3                                      7.119 ms      │ 14.04 ms      │ 11.04 ms      │ 11.23 ms      │ 100     │ 100
│     ├─ 4                                      9.947 ms      │ 16.07 ms      │ 13.14 ms      │ 13.45 ms      │ 100     │ 100
│     ├─ 5                                      10.74 ms      │ 18.4 ms       │ 15.17 ms      │ 15.24 ms      │ 100     │ 100
│     ╰─ 6                                      13.08 ms      │ 23.94 ms      │ 17 ms         │ 17.42 ms      │ 100     │ 100
├─ 02_concurrent_writes_contention                            │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      1.227 ms      │ 2.782 ms      │ 1.277 ms      │ 1.475 ms      │ 100     │ 100
│     ├─ 2                                      2.017 ms      │ 4.973 ms      │ 3.647 ms      │ 3.47 ms       │ 100     │ 100
│     ├─ 3                                      2.651 ms      │ 6.561 ms      │ 4.746 ms      │ 4.759 ms      │ 100     │ 100
│     ├─ 4                                      3.574 ms      │ 6.957 ms      │ 5.633 ms      │ 5.643 ms      │ 100     │ 100
│     ├─ 5                                      4.238 ms      │ 8.453 ms      │ 6.53 ms       │ 6.55 ms       │ 100     │ 100
│     ╰─ 6                                      5.13 ms       │ 10.81 ms      │ 7.693 ms      │ 7.806 ms      │ 100     │ 100
├─ 03_single_threaded_insert                                  │               │               │               │         │
│  ╰─ masstree24                                9.156 ms      │ 10.74 ms      │ 9.466 ms      │ 9.528 ms      │ 100     │ 100
├─ 04_read_after_write                                        │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      3.037 ms      │ 5.632 ms      │ 3.614 ms      │ 4.117 ms      │ 100     │ 100
│     ├─ 2                                      1.587 ms      │ 3.961 ms      │ 2.542 ms      │ 2.558 ms      │ 100     │ 100
│     ├─ 3                                      1.279 ms      │ 2.574 ms      │ 1.802 ms      │ 1.872 ms      │ 100     │ 100
│     ├─ 4                                      1.051 ms      │ 2.348 ms      │ 1.437 ms      │ 1.561 ms      │ 100     │ 100
│     ├─ 5                                      1.074 ms      │ 2.214 ms      │ 1.242 ms      │ 1.322 ms      │ 100     │ 100
│     ╰─ 6                                      890.8 µs      │ 1.89 ms       │ 1.264 ms      │ 1.208 ms      │ 100     │ 100
├─ 05_get_by_key_size                                         │               │               │               │         │
│  ├─ masstree24_8B                             60.69 µs      │ 86.74 µs      │ 63.19 µs      │ 65.96 µs      │ 100     │ 100
│  ├─ masstree24_16B                            71.99 µs      │ 87.07 µs      │ 73.65 µs      │ 74.6 µs       │ 100     │ 100
│  ├─ masstree24_24B                            76.81 µs      │ 118.6 µs      │ 78.05 µs      │ 87.58 µs      │ 100     │ 100
│  ╰─ masstree24_32B                            71.09 µs      │ 86.85 µs      │ 72.92 µs      │ 73.58 µs      │ 100     │ 100
├─ 06_insert_by_key_size                                      │               │               │               │         │
│  ├─ masstree24_8B                             69.92 µs      │ 102.6 µs      │ 70.77 µs      │ 73.04 µs      │ 100     │ 100
│  ├─ masstree24_16B                            83.68 µs      │ 117.2 µs      │ 84.03 µs      │ 84.99 µs      │ 100     │ 100
│  ├─ masstree24_24B                            95.37 µs      │ 107.7 µs      │ 95.58 µs      │ 96.22 µs      │ 100     │ 100
│  ╰─ masstree24_32B                            90.47 µs      │ 116.8 µs      │ 90.94 µs      │ 91.85 µs      │ 100     │ 100
├─ 07_concurrent_reads_scaling                                │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      21.49 ms      │ 26.03 ms      │ 21.95 ms      │ 22.53 ms      │ 100     │ 100
│     ├─ 2                                      21.72 ms      │ 27.62 ms      │ 22.79 ms      │ 22.95 ms      │ 100     │ 100
│     ├─ 3                                      22.36 ms      │ 29.49 ms      │ 25.55 ms      │ 25.48 ms      │ 100     │ 100
│     ├─ 4                                      23.34 ms      │ 31.76 ms      │ 26.22 ms      │ 26.35 ms      │ 100     │ 100
│     ├─ 5                                      23.99 ms      │ 31.99 ms      │ 26.53 ms      │ 26.71 ms      │ 100     │ 100
│     ╰─ 6                                      24.83 ms      │ 31.6 ms       │ 26.39 ms      │ 26.96 ms      │ 100     │ 100
├─ 08_concurrent_reads_long_keys                              │               │               │               │         │
│  ╰─ masstree24_32b                                          │               │               │               │         │
│     ├─ 1                                      24.98 ms      │ 30.5 ms       │ 25.75 ms      │ 26.05 ms      │ 100     │ 100
│     ├─ 2                                      25.54 ms      │ 27.91 ms      │ 26.57 ms      │ 26.55 ms      │ 100     │ 100
│     ├─ 3                                      26.44 ms      │ 33.64 ms      │ 29.34 ms      │ 29.26 ms      │ 100     │ 100
│     ├─ 4                                      27.34 ms      │ 35.37 ms      │ 30.56 ms      │ 30.34 ms      │ 100     │ 100
│     ├─ 5                                      28 ms         │ 49.14 ms      │ 31.41 ms      │ 31.48 ms      │ 100     │ 100
│     ╰─ 6                                      29.18 ms      │ 37.95 ms      │ 32.52 ms      │ 32.95 ms      │ 100     │ 100
├─ 09_mixed_uniform                                           │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      1.704 ms      │ 3.755 ms      │ 2.913 ms      │ 2.827 ms      │ 100     │ 100
│     ├─ 2                                      2.07 ms       │ 3.683 ms      │ 2.934 ms      │ 2.956 ms      │ 100     │ 100
│     ├─ 3                                      2.037 ms      │ 3.83 ms       │ 3.004 ms      │ 2.995 ms      │ 100     │ 100
│     ├─ 4                                      2.25 ms       │ 3.976 ms      │ 3.183 ms      │ 3.17 ms       │ 100     │ 100
│     ├─ 5                                      2.154 ms      │ 4.497 ms      │ 3.107 ms      │ 3.149 ms      │ 100     │ 100
│     ╰─ 6                                      2.359 ms      │ 4.135 ms      │ 3.236 ms      │ 3.241 ms      │ 100     │ 100
├─ 10a_read_scaling_8B                                        │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      3.382 ms      │ 5.726 ms      │ 3.52 ms       │ 3.784 ms      │ 100     │ 100
│     │                                         14.78 Mitem/s │ 8.731 Mitem/s │ 14.2 Mitem/s  │ 13.21 Mitem/s │         │
│     ├─ 2                                      3.43 ms       │ 7.42 ms       │ 4.58 ms       │ 4.804 ms      │ 100     │ 100
│     │                                         29.14 Mitem/s │ 13.47 Mitem/s │ 21.83 Mitem/s │ 20.81 Mitem/s │         │
│     ├─ 3                                      3.574 ms      │ 8.152 ms      │ 5.446 ms      │ 5.598 ms      │ 100     │ 100
│     │                                         41.96 Mitem/s │ 18.4 Mitem/s  │ 27.53 Mitem/s │ 26.79 Mitem/s │         │
│     ├─ 4                                      3.597 ms      │ 9.474 ms      │ 5.31 ms       │ 5.483 ms      │ 100     │ 100
│     │                                         55.58 Mitem/s │ 21.1 Mitem/s  │ 37.66 Mitem/s │ 36.47 Mitem/s │         │
│     ├─ 5                                      3.595 ms      │ 10.36 ms      │ 6.321 ms      │ 5.867 ms      │ 100     │ 100
│     │                                         69.52 Mitem/s │ 24.12 Mitem/s │ 39.54 Mitem/s │ 42.6 Mitem/s  │         │
│     ╰─ 6                                      3.646 ms      │ 9.44 ms       │ 5.246 ms      │ 5.036 ms      │ 100     │ 100
│                                               82.26 Mitem/s │ 31.77 Mitem/s │ 57.18 Mitem/s │ 59.56 Mitem/s │         │
├─ 10b_read_scaling_32B                                       │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      3.641 ms      │ 7.682 ms      │ 4.243 ms      │ 4.548 ms      │ 100     │ 100
│     │                                         13.72 Mitem/s │ 6.508 Mitem/s │ 11.78 Mitem/s │ 10.99 Mitem/s │         │
│     ├─ 2                                      3.757 ms      │ 8.604 ms      │ 4.544 ms      │ 4.911 ms      │ 100     │ 100
│     │                                         26.61 Mitem/s │ 11.62 Mitem/s │ 22 Mitem/s    │ 20.36 Mitem/s │         │
│     ├─ 3                                      3.864 ms      │ 9.352 ms      │ 7.464 ms      │ 6.714 ms      │ 100     │ 100
│     │                                         38.81 Mitem/s │ 16.03 Mitem/s │ 20.09 Mitem/s │ 22.33 Mitem/s │         │
│     ├─ 4                                      3.998 ms      │ 9.521 ms      │ 5.905 ms      │ 6.2 ms        │ 100     │ 100
│     │                                         50.01 Mitem/s │ 21 Mitem/s    │ 33.86 Mitem/s │ 32.25 Mitem/s │         │
│     ├─ 5                                      4.234 ms      │ 10.86 ms      │ 6.188 ms      │ 6.289 ms      │ 100     │ 100
│     │                                         59.04 Mitem/s │ 23 Mitem/s    │ 40.39 Mitem/s │ 39.74 Mitem/s │         │
│     ╰─ 6                                      4.297 ms      │ 10.19 ms      │ 6.218 ms      │ 6.423 ms      │ 100     │ 100
│                                               69.81 Mitem/s │ 29.41 Mitem/s │ 48.24 Mitem/s │ 46.69 Mitem/s │         │
├─ 10c_write_scaling_32B                                      │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      2.544 ms      │ 4.37 ms       │ 2.912 ms      │ 3.172 ms      │ 100     │ 100
│     │                                         3.93 Mitem/s  │ 2.288 Mitem/s │ 3.432 Mitem/s │ 3.152 Mitem/s │         │
│     ├─ 2                                      3.83 ms       │ 7.998 ms      │ 5.642 ms      │ 5.644 ms      │ 100     │ 100
│     │                                         5.221 Mitem/s │ 2.5 Mitem/s   │ 3.544 Mitem/s │ 3.543 Mitem/s │         │
│     ├─ 3                                      4.82 ms       │ 8.442 ms      │ 7.016 ms      │ 7.013 ms      │ 100     │ 100
│     │                                         6.223 Mitem/s │ 3.553 Mitem/s │ 4.275 Mitem/s │ 4.277 Mitem/s │         │
│     ├─ 4                                      5.942 ms      │ 11.05 ms      │ 8.838 ms      │ 8.864 ms      │ 100     │ 100
│     │                                         6.731 Mitem/s │ 3.619 Mitem/s │ 4.525 Mitem/s │ 4.512 Mitem/s │         │
│     ├─ 5                                      8.223 ms      │ 14.73 ms      │ 11.43 ms      │ 11.55 ms      │ 100     │ 100
│     │                                         6.08 Mitem/s  │ 3.394 Mitem/s │ 4.372 Mitem/s │ 4.325 Mitem/s │         │
│     ╰─ 6                                      7.561 ms      │ 15.83 ms      │ 13.26 ms      │ 13.13 ms      │ 100     │ 100
│                                               7.934 Mitem/s │ 3.788 Mitem/s │ 4.523 Mitem/s │ 4.568 Mitem/s │         │
├─ 11_single_hot_key                                          │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 2                                      954 µs        │ 2.755 ms      │ 1.983 ms      │ 1.924 ms      │ 100     │ 100
│     ├─ 4                                      1.724 ms      │ 3.989 ms      │ 3.325 ms      │ 3.267 ms      │ 100     │ 100
│     ├─ 8                                      3.602 ms      │ 7.637 ms      │ 6.318 ms      │ 6.348 ms      │ 100     │ 100
│     ├─ 16                                     11.91 ms      │ 20.67 ms      │ 13.84 ms      │ 14.16 ms      │ 100     │ 100
│     ╰─ 32                                     24.14 ms      │ 55.04 ms      │ 29.93 ms      │ 31.67 ms      │ 100     │ 100
├─ 11a_random_read_8B                                         │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      30.32 ms      │ 36.69 ms      │ 31.5 ms       │ 31.71 ms      │ 100     │ 100
│     │                                         3.297 Mitem/s │ 2.725 Mitem/s │ 3.174 Mitem/s │ 3.153 Mitem/s │         │
│     ├─ 2                                      30.75 ms      │ 41.52 ms      │ 32.16 ms      │ 32.51 ms      │ 100     │ 100
│     │                                         6.503 Mitem/s │ 4.815 Mitem/s │ 6.217 Mitem/s │ 6.15 Mitem/s  │         │
│     ├─ 3                                      31.59 ms      │ 42.2 ms       │ 33.68 ms      │ 34.7 ms       │ 100     │ 100
│     │                                         9.494 Mitem/s │ 7.108 Mitem/s │ 8.907 Mitem/s │ 8.643 Mitem/s │         │
│     ├─ 4                                      32.17 ms      │ 45.01 ms      │ 35.6 ms       │ 36.05 ms      │ 100     │ 100
│     │                                         12.43 Mitem/s │ 8.885 Mitem/s │ 11.23 Mitem/s │ 11.09 Mitem/s │         │
│     ├─ 5                                      32.62 ms      │ 43.62 ms      │ 35.89 ms      │ 36.09 ms      │ 100     │ 100
│     │                                         15.32 Mitem/s │ 11.46 Mitem/s │ 13.92 Mitem/s │ 13.85 Mitem/s │         │
│     ╰─ 6                                      32.58 ms      │ 47.03 ms      │ 36.19 ms      │ 37.14 ms      │ 100     │ 100
│                                               18.41 Mitem/s │ 12.75 Mitem/s │ 16.57 Mitem/s │ 16.15 Mitem/s │         │
├─ 11b_random_read_32B                                        │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      37.02 ms      │ 48.47 ms      │ 38.03 ms      │ 38.33 ms      │ 100     │ 100
│     │                                         2.701 Mitem/s │ 2.062 Mitem/s │ 2.629 Mitem/s │ 2.608 Mitem/s │         │
│     ├─ 2                                      38.44 ms      │ 51.65 ms      │ 40.03 ms      │ 40.32 ms      │ 100     │ 100
│     │                                         5.202 Mitem/s │ 3.872 Mitem/s │ 4.995 Mitem/s │ 4.959 Mitem/s │         │
│     ├─ 3                                      38.93 ms      │ 52.99 ms      │ 41.2 ms       │ 43.05 ms      │ 100     │ 100
│     │                                         7.704 Mitem/s │ 5.661 Mitem/s │ 7.28 Mitem/s  │ 6.967 Mitem/s │         │
│     ├─ 4                                      40.11 ms      │ 49.96 ms      │ 42.55 ms      │ 43.44 ms      │ 100     │ 100
│     │                                         9.972 Mitem/s │ 8.005 Mitem/s │ 9.398 Mitem/s │ 9.207 Mitem/s │         │
│     ├─ 5                                      40.45 ms      │ 57.41 ms      │ 46.19 ms      │ 45.49 ms      │ 100     │ 100
│     │                                         12.36 Mitem/s │ 8.709 Mitem/s │ 10.82 Mitem/s │ 10.99 Mitem/s │         │
│     ╰─ 6                                      41.45 ms      │ 59.75 ms      │ 47.87 ms      │ 48.11 ms      │ 100     │ 100
│                                               14.47 Mitem/s │ 10.04 Mitem/s │ 12.53 Mitem/s │ 12.46 Mitem/s │         │
├─ 12_get_by_key_size_shared_prefix                           │               │               │               │         │
│  ├─ masstree24_16B                            156.7 µs      │ 269.3 µs      │ 218.5 µs      │ 203.6 µs      │ 100     │ 100
│  ├─ masstree24_24B                            122.6 µs      │ 191.4 µs      │ 124 µs        │ 133.7 µs      │ 100     │ 100
│  ╰─ masstree24_32B                            130.1 µs      │ 147.9 µs      │ 131.5 µs      │ 132.9 µs      │ 100     │ 100
├─ 12a_string_values_read                                     │               │               │               │         │
│  ╰─ masstree24_string                                       │               │               │               │         │
│     ├─ 1                                      14.19 ms      │ 17.6 ms       │ 14.88 ms      │ 15.12 ms      │ 100     │ 100
│     │                                         3.521 Mitem/s │ 2.839 Mitem/s │ 3.36 Mitem/s  │ 3.305 Mitem/s │         │
│     ├─ 2                                      14.42 ms      │ 20.52 ms      │ 15.35 ms      │ 15.57 ms      │ 100     │ 100
│     │                                         6.932 Mitem/s │ 4.872 Mitem/s │ 6.513 Mitem/s │ 6.42 Mitem/s  │         │
│     ├─ 3                                      14.84 ms      │ 22.29 ms      │ 17.53 ms      │ 17.36 ms      │ 100     │ 100
│     │                                         10.1 Mitem/s  │ 6.728 Mitem/s │ 8.554 Mitem/s │ 8.64 Mitem/s  │         │
│     ├─ 4                                      15.23 ms      │ 20.96 ms      │ 17.96 ms      │ 17.81 ms      │ 100     │ 100
│     │                                         13.13 Mitem/s │ 9.54 Mitem/s  │ 11.13 Mitem/s │ 11.22 Mitem/s │         │
│     ├─ 5                                      15.71 ms      │ 22.38 ms      │ 18.25 ms      │ 18.61 ms      │ 100     │ 100
│     │                                         15.9 Mitem/s  │ 11.16 Mitem/s │ 13.69 Mitem/s │ 13.42 Mitem/s │         │
│     ╰─ 6                                      15.8 ms       │ 21.36 ms      │ 18.27 ms      │ 18.68 ms      │ 100     │ 100
│                                               18.97 Mitem/s │ 14.03 Mitem/s │ 16.41 Mitem/s │ 16.05 Mitem/s │         │
├─ 12b_string_values_write                                    │               │               │               │         │
│  ╰─ masstree24_string                                       │               │               │               │         │
│     ├─ 1                                      2.296 ms      │ 4.124 ms      │ 3.1 ms        │ 2.971 ms      │ 100     │ 100
│     │                                         4.353 Mitem/s │ 2.424 Mitem/s │ 3.225 Mitem/s │ 3.365 Mitem/s │         │
│     ├─ 2                                      3.114 ms      │ 5.323 ms      │ 4.361 ms      │ 4.273 ms      │ 100     │ 100
│     │                                         6.422 Mitem/s │ 3.757 Mitem/s │ 4.585 Mitem/s │ 4.679 Mitem/s │         │
│     ├─ 3                                      3.62 ms       │ 6.921 ms      │ 5.269 ms      │ 5.244 ms      │ 100     │ 100
│     │                                         8.287 Mitem/s │ 4.334 Mitem/s │ 5.692 Mitem/s │ 5.72 Mitem/s  │         │
│     ├─ 4                                      4.72 ms       │ 8.565 ms      │ 6.232 ms      │ 6.323 ms      │ 100     │ 100
│     │                                         8.473 Mitem/s │ 4.669 Mitem/s │ 6.418 Mitem/s │ 6.326 Mitem/s │         │
│     ├─ 5                                      5.404 ms      │ 10.21 ms      │ 7.387 ms      │ 7.42 ms       │ 100     │ 100
│     │                                         9.251 Mitem/s │ 4.894 Mitem/s │ 6.768 Mitem/s │ 6.737 Mitem/s │         │
│     ╰─ 6                                      6.588 ms      │ 12.11 ms      │ 7.951 ms      │ 8.369 ms      │ 100     │ 100
│                                               9.107 Mitem/s │ 4.953 Mitem/s │ 7.545 Mitem/s │ 7.168 Mitem/s │         │
├─ 13_concurrent_reads_long_keys_shared_prefix                │               │               │               │         │
│  ╰─ masstree24_32b                                          │               │               │               │         │
│     ├─ 1                                      30.7 ms       │ 34.66 ms      │ 31.66 ms      │ 31.75 ms      │ 100     │ 100
│     ├─ 2                                      32.19 ms      │ 42.38 ms      │ 33.88 ms      │ 34.22 ms      │ 100     │ 100
│     ├─ 3                                      34.61 ms      │ 47.51 ms      │ 36.46 ms      │ 37.47 ms      │ 100     │ 100
│     ├─ 4                                      35.47 ms      │ 46.7 ms       │ 37.84 ms      │ 38.75 ms      │ 100     │ 100
│     ├─ 5                                      36.07 ms      │ 67.8 ms       │ 39.9 ms       │ 40.22 ms      │ 100     │ 100
│     ╰─ 6                                      36.72 ms      │ 49.45 ms      │ 40.75 ms      │ 41.16 ms      │ 100     │ 100
├─ 14a_aggressive_shared_prefix_read                          │               │               │               │         │
│  ╰─ masstree24                                              │               │               │               │         │
│     ├─ 1                                      39.91 ms      │ 51.16 ms      │ 41.77 ms      │ 41.87 ms      │ 100     │ 100
│     │                                         2.505 Mitem/s │ 1.954 Mitem/s │ 2.393 Mitem/s │ 2.388 Mitem/s │         │
│     ├─ 2                                      40.7 ms       │ 50.66 ms      │ 42.08 ms      │ 42.31 ms      │ 100     │ 100
│     │                                         4.913 Mitem/s │ 3.947 Mitem/s │ 4.751 Mitem/s │ 4.726 Mitem/s │         │
│     ├─ 3                                      40.64 ms      │ 58.52 ms      │ 45.46 ms      │ 48.1 ms       │ 100     │ 100
│     │                                         7.38 Mitem/s  │ 5.126 Mitem/s │ 6.597 Mitem/s │ 6.236 Mitem/s │         │
│     ├─ 4                                      41.8 ms       │ 59.78 ms      │ 43.88 ms      │ 46.73 ms      │ 100     │ 100
│     │                                         9.568 Mitem/s │ 6.69 Mitem/s  │ 9.114 Mitem/s │ 8.559 Mitem/s │         │
│     ├─ 5                                      41.71 ms      │ 57.19 ms      │ 47.51 ms      │ 47.21 ms      │ 100     │ 100
│     │                                         11.98 Mitem/s │ 8.742 Mitem/s │ 10.52 Mitem/s │ 10.58 Mitem/s │         │
│     ╰─ 6                                      43.1 ms       │ 65.67 ms      │ 49.41 ms      │ 50.47 ms      │ 100     │ 100
│                                               13.91 Mitem/s │ 9.136 Mitem/s │ 12.14 Mitem/s │ 11.88 Mitem/s │         │
╰─ 14b_aggressive_shared_prefix_write                         │               │               │               │         │
   ╰─ masstree24                                              │               │               │               │         │
      ├─ 1                                      2.241 ms      │ 5.3 ms        │ 3.764 ms      │ 3.688 ms      │ 100     │ 100
      │                                         4.461 Mitem/s │ 1.886 Mitem/s │ 2.656 Mitem/s │ 2.711 Mitem/s │         │
      ├─ 2                                      3.789 ms      │ 7.429 ms      │ 5.163 ms      │ 5.306 ms      │ 100     │ 100
      │                                         5.278 Mitem/s │ 2.692 Mitem/s │ 3.873 Mitem/s │ 3.769 Mitem/s │         │
      ├─ 3                                      4.413 ms      │ 8.657 ms      │ 7.337 ms      │ 6.966 ms      │ 100     │ 100
      │                                         6.797 Mitem/s │ 3.465 Mitem/s │ 4.088 Mitem/s │ 4.306 Mitem/s │         │
      ├─ 4                                      6.229 ms      │ 11.3 ms       │ 8.685 ms      │ 8.637 ms      │ 100     │ 100
      │                                         6.421 Mitem/s │ 3.537 Mitem/s │ 4.605 Mitem/s │ 4.631 Mitem/s │         │
      ├─ 5                                      8.107 ms      │ 21.36 ms      │ 11.78 ms      │ 11.8 ms       │ 100     │ 100
      │                                         6.167 Mitem/s │ 2.339 Mitem/s │ 4.243 Mitem/s │ 4.235 Mitem/s │         │
      ╰─ 6                                      13.14 ms      │ 19.69 ms      │ 15.73 ms      │ 15.95 ms      │ 100     │ 100
                                                4.566 Mitem/s │ 3.047 Mitem/s │ 3.813 Mitem/s │ 3.759 Mitem/s │         │
```
