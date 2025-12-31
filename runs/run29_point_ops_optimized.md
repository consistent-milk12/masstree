```bash
Timer precision: 30 ns
concurrent_maps24                               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_concurrent_writes_disjoint                              │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      6.148 ms      │ 8.374 ms      │ 6.269 ms      │ 6.521 ms      │ 100     │ 100
│  │  ├─ 2                                      6.577 ms      │ 11.75 ms      │ 10.04 ms      │ 10.01 ms      │ 100     │ 100
│  │  ├─ 3                                      9.693 ms      │ 15.74 ms      │ 11.68 ms      │ 12.12 ms      │ 100     │ 100
│  │  ├─ 4                                      10.77 ms      │ 18.05 ms      │ 14.21 ms      │ 14.45 ms      │ 100     │ 100
│  │  ├─ 5                                      11.71 ms      │ 21.41 ms      │ 16.92 ms      │ 16.66 ms      │ 100     │ 100
│  │  ╰─ 6                                      12.38 ms      │ 25.56 ms      │ 18.29 ms      │ 18.51 ms      │ 100     │ 100
├─ 02_concurrent_writes_contention                            │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      1.138 ms      │ 2.465 ms      │ 1.191 ms      │ 1.353 ms      │ 100     │ 100
│  │  ├─ 2                                      2.041 ms      │ 4.921 ms      │ 3.968 ms      │ 3.822 ms      │ 100     │ 100
│  │  ├─ 3                                      2.626 ms      │ 6.054 ms      │ 4.789 ms      │ 4.563 ms      │ 100     │ 100
│  │  ├─ 4                                      3.473 ms      │ 7.111 ms      │ 5.578 ms      │ 5.594 ms      │ 100     │ 100
│  │  ├─ 5                                      4.07 ms       │ 8.522 ms      │ 6.498 ms      │ 6.586 ms      │ 100     │ 100
│  │  ╰─ 6                                      5.14 ms       │ 9.801 ms      │ 8.075 ms      │ 8.081 ms      │ 100     │ 100
├─ 03_single_threaded_insert                                  │               │               │               │         │
│  ├─ masstree24                                11.13 ms      │ 12.16 ms      │ 11.51 ms      │ 11.52 ms      │ 100     │ 100
├─ 04_read_after_write                                        │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      2.979 ms      │ 4.922 ms      │ 3.522 ms      │ 3.667 ms      │ 100     │ 100
│  │  ├─ 2                                      1.335 ms      │ 2.93 ms       │ 2.145 ms      │ 2.16 ms       │ 100     │ 100
│  │  ├─ 3                                      1.309 ms      │ 2.091 ms      │ 1.546 ms      │ 1.596 ms      │ 100     │ 100
│  │  ├─ 4                                      1.058 ms      │ 1.823 ms      │ 1.301 ms      │ 1.333 ms      │ 100     │ 100
│  │  ├─ 5                                      769.4 µs      │ 1.656 ms      │ 1.034 ms      │ 1.099 ms      │ 100     │ 100
│  │  ╰─ 6                                      625.5 µs      │ 1.557 ms      │ 973.7 µs      │ 982.4 µs      │ 100     │ 100
├─ 05_get_by_key_size                                         │               │               │               │         │
│  ├─ masstree24_8B                             57.26 µs      │ 75.79 µs      │ 58.63 µs      │ 59.44 µs      │ 100     │ 100
│  ├─ masstree24_16B                            68.96 µs      │ 93.21 µs      │ 72.62 µs      │ 73.34 µs      │ 100     │ 100
│  ├─ masstree24_24B                            79.19 µs      │ 261 µs        │ 80.74 µs      │ 96.52 µs      │ 100     │ 100
│  ├─ masstree24_32B                            79.89 µs      │ 110.6 µs      │ 81.19 µs      │ 83.7 µs       │ 100     │ 100
├─ 06_insert_by_key_size                                      │               │               │               │         │
│  ├─ masstree24_8B                             90.03 µs      │ 144.2 µs      │ 91.88 µs      │ 94.59 µs      │ 100     │ 100
│  ├─ masstree24_16B                            98.61 µs      │ 124.6 µs      │ 99.39 µs      │ 100.5 µs      │ 100     │ 100
│  ├─ masstree24_24B                            114.2 µs      │ 166.1 µs      │ 115.2 µs      │ 117.2 µs      │ 100     │ 100
│  ├─ masstree24_32B                            106 µs        │ 161.7 µs      │ 107 µs        │ 108.6 µs      │ 100     │ 100
├─ 07_concurrent_reads_scaling                                │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      21.79 ms      │ 24.53 ms      │ 22.37 ms      │ 22.47 ms      │ 100     │ 100
│  │  ├─ 2                                      21.84 ms      │ 25.52 ms      │ 22.89 ms      │ 22.88 ms      │ 100     │ 100
│  │  ├─ 3                                      22.78 ms      │ 29.83 ms      │ 27.09 ms      │ 26.65 ms      │ 100     │ 100
│  │  ├─ 4                                      23.37 ms      │ 30.2 ms       │ 26.37 ms      │ 26.54 ms      │ 100     │ 100
│  │  ├─ 5                                      24.46 ms      │ 32.53 ms      │ 26.81 ms      │ 27.11 ms      │ 100     │ 100
│  │  ╰─ 6                                      24.51 ms      │ 31.33 ms      │ 27.2 ms       │ 27.4 ms       │ 100     │ 100
├─ 08_concurrent_reads_long_keys                              │               │               │               │         │
│  ├─ masstree24_32b                                          │               │               │               │         │
│  │  ├─ 1                                      27.82 ms      │ 35.4 ms       │ 28.96 ms      │ 29.39 ms      │ 100     │ 100
│  │  ├─ 2                                      28.32 ms      │ 40.54 ms      │ 29.91 ms      │ 30.37 ms      │ 100     │ 100
│  │  ├─ 3                                      29.85 ms      │ 38.28 ms      │ 31.22 ms      │ 32.07 ms      │ 100     │ 100
│  │  ├─ 4                                      30.64 ms      │ 40.41 ms      │ 34.34 ms      │ 34.24 ms      │ 100     │ 100
│  │  ├─ 5                                      31.34 ms      │ 44.95 ms      │ 35.16 ms      │ 35.32 ms      │ 100     │ 100
│  │  ╰─ 6                                      32.13 ms      │ 53.01 ms      │ 36.39 ms      │ 36.7 ms       │ 100     │ 100
├─ 09_mixed_uniform                                           │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      1.703 ms      │ 4.413 ms      │ 2.709 ms      │ 2.653 ms      │ 100     │ 100
│  │  ├─ 2                                      1.853 ms      │ 3.736 ms      │ 2.981 ms      │ 2.945 ms      │ 100     │ 100
│  │  ├─ 3                                      2.199 ms      │ 4.755 ms      │ 3.033 ms      │ 3.058 ms      │ 100     │ 100
│  │  ├─ 4                                      1.78 ms       │ 5.619 ms      │ 3.093 ms      │ 3.101 ms      │ 100     │ 100
│  │  ├─ 5                                      2.223 ms      │ 4.616 ms      │ 3.147 ms      │ 3.175 ms      │ 100     │ 100
│  │  ╰─ 6                                      2.087 ms      │ 5.068 ms      │ 3.181 ms      │ 3.212 ms      │ 100     │ 100
├─ 10a_read_scaling_8B                                        │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      2.87 ms       │ 4.983 ms      │ 3.021 ms      │ 3.323 ms      │ 100     │ 100
│  │  │                                         17.41 Mitem/s │ 10.03 Mitem/s │ 16.54 Mitem/s │ 15.04 Mitem/s │         │
│  │  ├─ 2                                      2.979 ms      │ 6.898 ms      │ 4.279 ms      │ 4.389 ms      │ 100     │ 100
│  │  │                                         33.56 Mitem/s │ 14.49 Mitem/s │ 23.36 Mitem/s │ 22.78 Mitem/s │         │
│  │  ├─ 3                                      3.016 ms      │ 7.29 ms       │ 3.611 ms      │ 4.032 ms      │ 100     │ 100
│  │  │                                         49.73 Mitem/s │ 20.57 Mitem/s │ 41.53 Mitem/s │ 37.19 Mitem/s │         │
│  │  ├─ 4                                      3.03 ms       │ 6.99 ms       │ 4.468 ms      │ 4.29 ms       │ 100     │ 100
│  │  │                                         65.98 Mitem/s │ 28.61 Mitem/s │ 44.75 Mitem/s │ 46.61 Mitem/s │         │
│  │  ├─ 5                                      3.108 ms      │ 8.046 ms      │ 4.664 ms      │ 4.657 ms      │ 100     │ 100
│  │  │                                         80.42 Mitem/s │ 31.06 Mitem/s │ 53.59 Mitem/s │ 53.67 Mitem/s │         │
│  │  ╰─ 6                                      3.127 ms      │ 8.271 ms      │ 4.674 ms      │ 4.479 ms      │ 100     │ 100
│  │                                            95.91 Mitem/s │ 36.27 Mitem/s │ 64.18 Mitem/s │ 66.97 Mitem/s │         │
├─ 10b_read_scaling_32B                                       │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      3.601 ms      │ 7.058 ms      │ 3.697 ms      │ 4.026 ms      │ 100     │ 100
│  │  │                                         13.88 Mitem/s │ 7.083 Mitem/s │ 13.52 Mitem/s │ 12.41 Mitem/s │         │
│  │  ├─ 2                                      3.669 ms      │ 7.73 ms       │ 6.435 ms      │ 5.839 ms      │ 100     │ 100
│  │  │                                         27.24 Mitem/s │ 12.93 Mitem/s │ 15.53 Mitem/s │ 17.12 Mitem/s │         │
│  │  ├─ 3                                      3.821 ms      │ 7.875 ms      │ 6.988 ms      │ 6.645 ms      │ 100     │ 100
│  │  │                                         39.24 Mitem/s │ 19.04 Mitem/s │ 21.46 Mitem/s │ 22.57 Mitem/s │         │
│  │  ├─ 4                                      3.812 ms      │ 8.07 ms       │ 6.71 ms       │ 6.284 ms      │ 100     │ 100
│  │  │                                         52.46 Mitem/s │ 24.78 Mitem/s │ 29.8 Mitem/s  │ 31.82 Mitem/s │         │
│  │  ├─ 5                                      3.84 ms       │ 8.38 ms       │ 5.657 ms      │ 5.387 ms      │ 100     │ 100
│  │  │                                         65.09 Mitem/s │ 29.83 Mitem/s │ 44.18 Mitem/s │ 46.4 Mitem/s  │         │
│  │  ╰─ 6                                      3.928 ms      │ 7.334 ms      │ 5.57 ms       │ 5.309 ms      │ 100     │ 100
│  │                                            76.36 Mitem/s │ 40.9 Mitem/s  │ 53.85 Mitem/s │ 56.49 Mitem/s │         │
├─ 10c_write_scaling_32B                                      │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      2.277 ms      │ 4.753 ms      │ 3.075 ms      │ 3.009 ms      │ 100     │ 100
│  │  │                                         4.391 Mitem/s │ 2.103 Mitem/s │ 3.251 Mitem/s │ 3.323 Mitem/s │         │
│  │  ├─ 2                                      3.674 ms      │ 18.76 ms      │ 5.562 ms      │ 6.441 ms      │ 100     │ 100
│  │  │                                         5.442 Mitem/s │ 1.065 Mitem/s │ 3.595 Mitem/s │ 3.105 Mitem/s │         │
│  │  ├─ 3                                      5.136 ms      │ 27.44 ms      │ 10.77 ms      │ 11.58 ms      │ 100     │ 100
│  │  │                                         5.84 Mitem/s  │ 1.093 Mitem/s │ 2.784 Mitem/s │ 2.59 Mitem/s  │         │
│  │  ├─ 4                                      6.031 ms      │ 11.26 ms      │ 8.659 ms      │ 8.517 ms      │ 100     │ 100
│  │  │                                         6.631 Mitem/s │ 3.551 Mitem/s │ 4.618 Mitem/s │ 4.696 Mitem/s │         │
│  │  ├─ 5                                      8.086 ms      │ 13.19 ms      │ 11.18 ms      │ 11.18 ms      │ 100     │ 100
│  │  │                                         6.183 Mitem/s │ 3.788 Mitem/s │ 4.468 Mitem/s │ 4.47 Mitem/s  │         │
│  │  ╰─ 6                                      9.277 ms      │ 25.11 ms      │ 13.75 ms      │ 13.91 ms      │ 100     │ 100
│  │                                            6.467 Mitem/s │ 2.389 Mitem/s │ 4.362 Mitem/s │ 4.312 Mitem/s │         │
├─ 11_single_hot_key                                          │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 2                                      997 µs        │ 2.358 ms      │ 1.91 ms       │ 1.866 ms      │ 100     │ 100
│  │  ├─ 4                                      1.696 ms      │ 3.734 ms      │ 3.243 ms      │ 3.139 ms      │ 100     │ 100
│  │  ├─ 8                                      4.536 ms      │ 7.094 ms      │ 5.958 ms      │ 5.946 ms      │ 100     │ 100
│  │  ├─ 16                                     10.96 ms      │ 17.15 ms      │ 12.08 ms      │ 12.64 ms      │ 100     │ 100
│  │  ╰─ 32                                     21.04 ms      │ 57.84 ms      │ 24.35 ms      │ 26.14 ms      │ 100     │ 100
├─ 11a_random_read_8B                                         │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      30.79 ms      │ 35.5 ms       │ 32.11 ms      │ 32.31 ms      │ 100     │ 100
│  │  │                                         3.247 Mitem/s │ 2.816 Mitem/s │ 3.113 Mitem/s │ 3.094 Mitem/s │         │
│  │  ├─ 2                                      31.62 ms      │ 40.77 ms      │ 32.76 ms      │ 33 ms         │ 100     │ 100
│  │  │                                         6.323 Mitem/s │ 4.904 Mitem/s │ 6.104 Mitem/s │ 6.06 Mitem/s  │         │
│  │  ├─ 3                                      31.72 ms      │ 42.19 ms      │ 34.14 ms      │ 35.14 ms      │ 100     │ 100
│  │  │                                         9.455 Mitem/s │ 7.109 Mitem/s │ 8.785 Mitem/s │ 8.536 Mitem/s │         │
│  │  ├─ 4                                      31.99 ms      │ 44.21 ms      │ 33.88 ms      │ 34.77 ms      │ 100     │ 100
│  │  │                                         12.5 Mitem/s  │ 9.046 Mitem/s │ 11.8 Mitem/s  │ 11.5 Mitem/s  │         │
│  │  ├─ 5                                      32.93 ms      │ 46.73 ms      │ 36.15 ms      │ 36.76 ms      │ 100     │ 100
│  │  │                                         15.18 Mitem/s │ 10.69 Mitem/s │ 13.83 Mitem/s │ 13.59 Mitem/s │         │
│  │  ╰─ 6                                      33.71 ms      │ 45.11 ms      │ 36.96 ms      │ 37.55 ms      │ 100     │ 100
│  │                                            17.79 Mitem/s │ 13.29 Mitem/s │ 16.23 Mitem/s │ 15.97 Mitem/s │         │
├─ 11b_random_read_32B                                        │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      44.61 ms      │ 51.91 ms      │ 47.06 ms      │ 47.29 ms      │ 100     │ 100
│  │  │                                         2.241 Mitem/s │ 1.926 Mitem/s │ 2.124 Mitem/s │ 2.114 Mitem/s │         │
│  │  ├─ 2                                      45.51 ms      │ 58.38 ms      │ 46.94 ms      │ 47.29 ms      │ 100     │ 100
│  │  │                                         4.394 Mitem/s │ 3.425 Mitem/s │ 4.26 Mitem/s  │ 4.228 Mitem/s │         │
│  │  ├─ 3                                      46.11 ms      │ 63 ms         │ 48.87 ms      │ 50.8 ms       │ 100     │ 100
│  │  │                                         6.505 Mitem/s │ 4.761 Mitem/s │ 6.138 Mitem/s │ 5.905 Mitem/s │         │
│  │  ├─ 4                                      46.62 ms      │ 60.25 ms      │ 49.83 ms      │ 50.79 ms      │ 100     │ 100
│  │  │                                         8.579 Mitem/s │ 6.638 Mitem/s │ 8.025 Mitem/s │ 7.874 Mitem/s │         │
│  │  ├─ 5                                      47.52 ms      │ 61.41 ms      │ 53.17 ms      │ 52.37 ms      │ 100     │ 100
│  │  │                                         10.52 Mitem/s │ 8.141 Mitem/s │ 9.402 Mitem/s │ 9.545 Mitem/s │         │
│  │  ╰─ 6                                      48.03 ms      │ 62.95 ms      │ 54.09 ms      │ 54.16 ms      │ 100     │ 100
│  │                                            12.49 Mitem/s │ 9.531 Mitem/s │ 11.09 Mitem/s │ 11.07 Mitem/s │         │
├─ 12_get_by_key_size_shared_prefix                           │               │               │               │         │
│  ├─ masstree24_16B                            102.2 µs      │ 116.4 µs      │ 103.6 µs      │ 105 µs        │ 100     │ 100
│  ├─ masstree24_24B                            112.4 µs      │ 156.7 µs      │ 114.1 µs      │ 116.6 µs      │ 100     │ 100
│  ├─ masstree24_32B                            119.9 µs      │ 139.2 µs      │ 121.2 µs      │ 122.8 µs      │ 100     │ 100
├─ 12a_string_values_read                                     │               │               │               │         │
│  ├─ masstree24_string                                       │               │               │               │         │
│  │  ├─ 1                                      14.04 ms      │ 18.15 ms      │ 14.53 ms      │ 14.75 ms      │ 100     │ 100
│  │  │                                         3.56 Mitem/s  │ 2.754 Mitem/s │ 3.44 Mitem/s  │ 3.388 Mitem/s │         │
│  │  ├─ 2                                      14.15 ms      │ 16.27 ms      │ 14.69 ms      │ 14.81 ms      │ 100     │ 100
│  │  │                                         7.064 Mitem/s │ 6.146 Mitem/s │ 6.805 Mitem/s │ 6.749 Mitem/s │         │
│  │  ├─ 3                                      14.72 ms      │ 24.05 ms      │ 17.27 ms      │ 16.98 ms      │ 100     │ 100
│  │  │                                         10.18 Mitem/s │ 6.236 Mitem/s │ 8.684 Mitem/s │ 8.829 Mitem/s │         │
│  │  ├─ 4                                      14.79 ms      │ 21.95 ms      │ 17.57 ms      │ 17.43 ms      │ 100     │ 100
│  │  │                                         13.51 Mitem/s │ 9.111 Mitem/s │ 11.37 Mitem/s │ 11.47 Mitem/s │         │
│  │  ├─ 5                                      15.33 ms      │ 23.66 ms      │ 17.93 ms      │ 18.18 ms      │ 100     │ 100
│  │  │                                         16.3 Mitem/s  │ 10.56 Mitem/s │ 13.93 Mitem/s │ 13.74 Mitem/s │         │
│  │  ╰─ 6                                      17.75 ms      │ 23.44 ms      │ 18.63 ms      │ 18.99 ms      │ 100     │ 100
│  │                                            16.89 Mitem/s │ 12.79 Mitem/s │ 16.09 Mitem/s │ 15.79 Mitem/s │         │
├─ 12b_string_values_write                                    │               │               │               │         │
│  ├─ masstree24_string                                       │               │               │               │         │
│  │  ├─ 1                                      2.382 ms      │ 5.656 ms      │ 3.491 ms      │ 3.474 ms      │ 100     │ 100
│  │  │                                         4.198 Mitem/s │ 1.768 Mitem/s │ 2.863 Mitem/s │ 2.878 Mitem/s │         │
│  │  ├─ 2                                      3.428 ms      │ 7.093 ms      │ 4.718 ms      │ 4.673 ms      │ 100     │ 100
│  │  │                                         5.833 Mitem/s │ 2.819 Mitem/s │ 4.238 Mitem/s │ 4.279 Mitem/s │         │
│  │  ├─ 3                                      3.746 ms      │ 7.839 ms      │ 5.708 ms      │ 5.7 ms        │ 100     │ 100
│  │  │                                         8.008 Mitem/s │ 3.826 Mitem/s │ 5.255 Mitem/s │ 5.262 Mitem/s │         │
│  │  ├─ 4                                      4.771 ms      │ 9.752 ms      │ 6.763 ms      │ 6.713 ms      │ 100     │ 100
│  │  │                                         8.382 Mitem/s │ 4.101 Mitem/s │ 5.914 Mitem/s │ 5.958 Mitem/s │         │
│  │  ├─ 5                                      5.823 ms      │ 9.537 ms      │ 7.437 ms      │ 7.484 ms      │ 100     │ 100
│  │  │                                         8.586 Mitem/s │ 5.242 Mitem/s │ 6.722 Mitem/s │ 6.68 Mitem/s  │         │
│  │  ╰─ 6                                      6.818 ms      │ 12.46 ms      │ 8.767 ms      │ 8.83 ms       │ 100     │ 100
│  │                                            8.799 Mitem/s │ 4.811 Mitem/s │ 6.843 Mitem/s │ 6.794 Mitem/s │         │
├─ 13_concurrent_reads_long_keys_shared_prefix                │               │               │               │         │
│  ├─ masstree24_32b                                          │               │               │               │         │
│  │  ├─ 1                                      31.63 ms      │ 36.11 ms      │ 32.98 ms      │ 33.12 ms      │ 100     │ 100
│  │  ├─ 2                                      34.38 ms      │ 37.72 ms      │ 35.41 ms      │ 35.53 ms      │ 100     │ 100
│  │  ├─ 3                                      36.02 ms      │ 46.15 ms      │ 37.7 ms       │ 38.84 ms      │ 100     │ 100
│  │  ├─ 4                                      36.63 ms      │ 48.08 ms      │ 39.39 ms      │ 39.97 ms      │ 100     │ 100
│  │  ├─ 5                                      37.67 ms      │ 51.03 ms      │ 41.4 ms       │ 41.13 ms      │ 100     │ 100
│  │  ╰─ 6                                      38.62 ms      │ 50.18 ms      │ 42.09 ms      │ 42.64 ms      │ 100     │ 100
├─ 14a_aggressive_shared_prefix_read                          │               │               │               │         │
│  ├─ masstree24                                              │               │               │               │         │
│  │  ├─ 1                                      37 ms         │ 42.51 ms      │ 38.28 ms      │ 38.38 ms      │ 100     │ 100
│  │  │                                         2.702 Mitem/s │ 2.352 Mitem/s │ 2.612 Mitem/s │ 2.605 Mitem/s │         │
│  │  ├─ 2                                      37.52 ms      │ 41.26 ms      │ 39.23 ms      │ 39.21 ms      │ 100     │ 100
│  │  │                                         5.329 Mitem/s │ 4.846 Mitem/s │ 5.097 Mitem/s │ 5.099 Mitem/s │         │
│  │  ├─ 3                                      38.49 ms      │ 54.04 ms      │ 41.17 ms      │ 44.17 ms      │ 100     │ 100
│  │  │                                         7.792 Mitem/s │ 5.551 Mitem/s │ 7.286 Mitem/s │ 6.79 Mitem/s  │         │
│  │  ├─ 4                                      38.84 ms      │ 52.8 ms       │ 40.89 ms      │ 43.14 ms      │ 100     │ 100
│  │  │                                         10.29 Mitem/s │ 7.574 Mitem/s │ 9.782 Mitem/s │ 9.27 Mitem/s  │         │
│  │  ├─ 5                                      39.54 ms      │ 53.55 ms      │ 44.97 ms      │ 44.71 ms      │ 100     │ 100
│  │  │                                         12.64 Mitem/s │ 9.335 Mitem/s │ 11.11 Mitem/s │ 11.18 Mitem/s │         │
│  │  ╰─ 6                                      40.07 ms      │ 57.3 ms       │ 45.77 ms      │ 46.6 ms       │ 100     │ 100
│  │                                            14.97 Mitem/s │ 10.47 Mitem/s │ 13.1 Mitem/s  │ 12.87 Mitem/s │         │
╰─ 14b_aggressive_shared_prefix_write                         │               │               │               │         │
   ├─ masstree24                                              │               │               │               │         │
   │  ├─ 1                                      2.126 ms      │ 4.851 ms      │ 3.374 ms      │ 3.463 ms      │ 100     │ 100
   │  │                                         4.702 Mitem/s │ 2.061 Mitem/s │ 2.963 Mitem/s │ 2.886 Mitem/s │         │
   │  ├─ 2                                      4.054 ms      │ 7.221 ms      │ 5.915 ms      │ 5.809 ms      │ 100     │ 100
   │  │                                         4.933 Mitem/s │ 2.769 Mitem/s │ 3.38 Mitem/s  │ 3.442 Mitem/s │         │
   │  ├─ 3                                      4.953 ms      │ 8.483 ms      │ 6.842 ms      │ 6.819 ms      │ 100     │ 100
   │  │                                         6.055 Mitem/s │ 3.536 Mitem/s │ 4.384 Mitem/s │ 4.399 Mitem/s │         │
   │  ├─ 4                                      6.273 ms      │ 10.8 ms       │ 8.923 ms      │ 8.951 ms      │ 100     │ 100
   │  │                                         6.375 Mitem/s │ 3.7 Mitem/s   │ 4.482 Mitem/s │ 4.468 Mitem/s │         │
   │  ├─ 5                                      7.697 ms      │ 14.59 ms      │ 12.86 ms      │ 12.49 ms      │ 100     │ 100
   │  │                                         6.495 Mitem/s │ 3.426 Mitem/s │ 3.885 Mitem/s │ 4 Mitem/s     │         │
   │  ╰─ 6                                      9.353 ms      │ 23.37 ms      │ 14.36 ms      │ 14.43 ms      │ 100     │ 100
   │                                            6.414 Mitem/s │ 2.566 Mitem/s │ 4.176 Mitem/s │ 4.155 Mitem/s │         │
```
