# Final Benches on Pure Reads

After increasing sample sizes for all tests to 10M keys and making all tests
and making all tests use up to 32 threads consistently.

```bash

╰─❯ cargo bench --features mimalloc --bench concurrent_maps
Timer precision: 20 ns
concurrent_maps                                      fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_get_by_key_size                                              │               │               │               │         │
│  ├─ dashmap_8B                                     25.52 µs      │ 46.09 µs      │ 27.07 µs      │ 29.12 µs      │ 100     │ 100
│  ├─ dashmap_16B                                    28.03 µs      │ 37.9 µs       │ 30.58 µs      │ 30.33 µs      │ 100     │ 100
│  ├─ dashmap_24B                                    31.59 µs      │ 107.7 µs      │ 31.65 µs      │ 32.61 µs      │ 100     │ 100
│  ├─ dashmap_32B                                    35.29 µs      │ 40.94 µs      │ 35.35 µs      │ 35.61 µs      │ 100     │ 100
│  ├─ indexset_8B                                    79.2 µs       │ 290.3 µs      │ 80.3 µs       │ 83.75 µs      │ 100     │ 100
│  ├─ indexset_16B                                   91.44 µs      │ 122.8 µs      │ 94.49 µs      │ 97.06 µs      │ 100     │ 100
│  ├─ indexset_24B                                   147.5 µs      │ 230.5 µs      │ 151.2 µs      │ 154.9 µs      │ 100     │ 100
│  ├─ indexset_32B                                   139.3 µs      │ 220.7 µs      │ 141.3 µs      │ 145.4 µs      │ 100     │ 100
│  ├─ masstree_8B                                    73.28 µs      │ 102.7 µs      │ 76.2 µs       │ 76.75 µs      │ 100     │ 100
│  ├─ masstree_16B                                   85.47 µs      │ 296.6 µs      │ 86.34 µs      │ 89.86 µs      │ 100     │ 100
│  ├─ masstree_24B                                   88.41 µs      │ 302.2 µs      │ 89.66 µs      │ 95.38 µs      │ 100     │ 100
│  ├─ masstree_32B                                   87.54 µs      │ 133.4 µs      │ 92.68 µs      │ 95.57 µs      │ 100     │ 100
│  ├─ skipmap_8B                                     110.3 µs      │ 188.9 µs      │ 114.2 µs      │ 117.1 µs      │ 100     │ 100
│  ├─ skipmap_16B                                    130.6 µs      │ 347.7 µs      │ 137.7 µs      │ 141 µs        │ 100     │ 100
│  ├─ skipmap_24B                                    226 µs        │ 320.1 µs      │ 229.6 µs      │ 234.3 µs      │ 100     │ 100
│  ╰─ skipmap_32B                                    200.8 µs      │ 288.2 µs      │ 204.4 µs      │ 210 µs        │ 100     │ 100
├─ 02_insert_by_key_size                                           │               │               │               │         │
│  ├─ dashmap_8B                                     53.36 µs      │ 78.01 µs      │ 55.37 µs      │ 58.19 µs      │ 100     │ 100
│  ├─ dashmap_16B                                    57.15 µs      │ 132.9 µs      │ 59.56 µs      │ 61.47 µs      │ 100     │ 100
│  ├─ dashmap_24B                                    61.31 µs      │ 103.5 µs      │ 63.97 µs      │ 67.5 µs       │ 100     │ 100
│  ├─ dashmap_32B                                    66.36 µs      │ 99.51 µs      │ 69.43 µs      │ 70.35 µs      │ 100     │ 100
│  ├─ indexset_8B                                    342.6 µs      │ 564.6 µs      │ 348.5 µs      │ 352.7 µs      │ 100     │ 100
│  ├─ indexset_16B                                   341 µs        │ 551.7 µs      │ 347.2 µs      │ 354.1 µs      │ 100     │ 100
│  ├─ indexset_24B                                   389.9 µs      │ 608.2 µs      │ 396.4 µs      │ 402.3 µs      │ 100     │ 100
│  ├─ indexset_32B                                   388.3 µs      │ 607.2 µs      │ 392.4 µs      │ 400.3 µs      │ 100     │ 100
│  ├─ masstree_8B                                    31.6 µs       │ 50.41 µs      │ 32.03 µs      │ 32.53 µs      │ 100     │ 100
│  ├─ masstree_16B                                   107.7 µs      │ 182.1 µs      │ 109.9 µs      │ 114.5 µs      │ 100     │ 100
│  ├─ masstree_24B                                   108 µs        │ 158 µs        │ 109.8 µs      │ 112.7 µs      │ 100     │ 100
│  ├─ masstree_32B                                   109 µs        │ 182.1 µs      │ 110.8 µs      │ 114.3 µs      │ 100     │ 100
│  ├─ skipmap_8B                                     95.64 µs      │ 310.9 µs      │ 96.58 µs      │ 99.7 µs       │ 100     │ 100
│  ├─ skipmap_16B                                    99.28 µs      │ 128 µs        │ 101.2 µs      │ 103.8 µs      │ 100     │ 100
│  ├─ skipmap_24B                                    122.5 µs      │ 199 µs        │ 123.8 µs      │ 127.6 µs      │ 100     │ 100
│  ╰─ skipmap_32B                                    115.2 µs      │ 189.9 µs      │ 116.3 µs      │ 118.9 µs      │ 100     │ 100
├─ 03_concurrent_reads_scaling                                     │               │               │               │         │
│  ├─ dashmap                                                      │               │               │               │         │
│  │  ├─ 1                                           7.233 ms      │ 10.49 ms      │ 8.685 ms      │ 8.725 ms      │ 100     │ 100
│  │  ├─ 2                                           9.29 ms       │ 12.05 ms      │ 10.59 ms      │ 10.56 ms      │ 100     │ 100
│  │  ├─ 4                                           10.26 ms      │ 13.33 ms      │ 11.44 ms      │ 11.47 ms      │ 100     │ 100
│  │  ├─ 8                                           11.03 ms      │ 15.82 ms      │ 13.75 ms      │ 13.71 ms      │ 100     │ 100
│  │  ├─ 16                                          18.97 ms      │ 25.32 ms      │ 22.02 ms      │ 22.13 ms      │ 100     │ 100
│  │  ╰─ 32                                          32.8 ms       │ 61.79 ms      │ 35.91 ms      │ 36.48 ms      │ 100     │ 100
│  ├─ indexset                                                     │               │               │               │         │
│  │  ├─ 1                                           30 ms         │ 32.39 ms      │ 30.79 ms      │ 30.97 ms      │ 100     │ 100
│  │  ├─ 2                                           31.27 ms      │ 37.46 ms      │ 35.12 ms      │ 34.94 ms      │ 100     │ 100
│  │  ├─ 4                                           35.31 ms      │ 44.34 ms      │ 35.99 ms      │ 36.36 ms      │ 100     │ 100
│  │  ├─ 8                                           40.74 ms      │ 43.16 ms      │ 41.94 ms      │ 41.92 ms      │ 100     │ 100
│  │  ├─ 16                                          62.25 ms      │ 73.88 ms      │ 66.71 ms      │ 66.86 ms      │ 100     │ 100
│  │  ╰─ 32                                          115.9 ms      │ 157.8 ms      │ 120.9 ms      │ 123.7 ms      │ 100     │ 100
│  ├─ masstree                                                     │               │               │               │         │
│  │  ├─ 1                                           22.12 ms      │ 23.01 ms      │ 22.18 ms      │ 22.21 ms      │ 100     │ 100
│  │  ├─ 2                                           22.65 ms      │ 25.56 ms      │ 23.32 ms      │ 23.37 ms      │ 100     │ 100
│  │  ├─ 4                                           23.53 ms      │ 25.54 ms      │ 23.91 ms      │ 23.99 ms      │ 100     │ 100
│  │  ├─ 8                                           26.58 ms      │ 29.55 ms      │ 27.37 ms      │ 27.23 ms      │ 100     │ 100
│  │  ├─ 16                                          45.62 ms      │ 50.88 ms      │ 48.59 ms      │ 48.71 ms      │ 100     │ 100
│  │  ╰─ 32                                          76.71 ms      │ 92 ms         │ 80.07 ms      │ 80.48 ms      │ 100     │ 100
│  ╰─ skipmap                                                      │               │               │               │         │
│     ├─ 1                                           37.74 ms      │ 40.66 ms      │ 38.31 ms      │ 38.33 ms      │ 100     │ 100
│     ├─ 2                                           38.85 ms      │ 40.62 ms      │ 39.38 ms      │ 39.45 ms      │ 100     │ 100
│     ├─ 4                                           40.98 ms      │ 48.38 ms      │ 41.41 ms      │ 41.52 ms      │ 100     │ 100
│     ├─ 8                                           49.33 ms      │ 53.37 ms      │ 49.76 ms      │ 49.91 ms      │ 100     │ 100
│     ├─ 16                                          85.97 ms      │ 95.49 ms      │ 90.98 ms      │ 90.9 ms       │ 100     │ 100
│     ╰─ 32                                          152.1 ms      │ 228.4 ms      │ 173.9 ms      │ 175.1 ms      │ 100     │ 100
├─ 04_concurrent_reads_long_keys                                   │               │               │               │         │
│  ├─ dashmap_32b                                                  │               │               │               │         │
│  │  ├─ 1                                           7.955 ms      │ 14.51 ms      │ 9.474 ms      │ 9.611 ms      │ 100     │ 100
│  │  ├─ 2                                           9.957 ms      │ 16.29 ms      │ 12.38 ms      │ 12.44 ms      │ 100     │ 100
│  │  ├─ 4                                           11.42 ms      │ 17.54 ms      │ 12.62 ms      │ 12.81 ms      │ 100     │ 100
│  │  ├─ 8                                           13.63 ms      │ 18.17 ms      │ 15.44 ms      │ 15.42 ms      │ 100     │ 100
│  │  ├─ 16                                          21.03 ms      │ 49.52 ms      │ 25.07 ms      │ 25.36 ms      │ 100     │ 100
│  │  ╰─ 32                                          37.39 ms      │ 46.81 ms      │ 42.21 ms      │ 42.17 ms      │ 100     │ 100
│  ├─ indexset_32b                                                 │               │               │               │         │
│  │  ├─ 1                                           38.88 ms      │ 42.89 ms      │ 39.8 ms       │ 39.95 ms      │ 100     │ 100
│  │  ├─ 2                                           39.91 ms      │ 51.67 ms      │ 44.17 ms      │ 44.25 ms      │ 100     │ 100
│  │  ├─ 4                                           44.44 ms      │ 49.05 ms      │ 45.23 ms      │ 45.5 ms       │ 100     │ 100
│  │  ├─ 8                                           53.56 ms      │ 56.13 ms      │ 54.52 ms      │ 54.56 ms      │ 100     │ 100
│  │  ├─ 16                                          80.2 ms       │ 104.5 ms      │ 86.4 ms       │ 87.1 ms       │ 100     │ 100
│  │  ╰─ 32                                          150.8 ms      │ 173.9 ms      │ 159 ms        │ 159.5 ms      │ 100     │ 100
│  ├─ masstree_32b                                                 │               │               │               │         │
│  │  ├─ 1                                           28.76 ms      │ 37.01 ms      │ 30.18 ms      │ 30.8 ms       │ 100     │ 100
│  │  ├─ 2                                           29.47 ms      │ 38.89 ms      │ 31.13 ms      │ 31.75 ms      │ 100     │ 100
│  │  ├─ 4                                           30.74 ms      │ 46.52 ms      │ 34.75 ms      │ 35.03 ms      │ 100     │ 100
│  │  ├─ 8                                           35.2 ms       │ 47.47 ms      │ 38.97 ms      │ 39.26 ms      │ 100     │ 100
│  │  ├─ 16                                          63.35 ms      │ 91.28 ms      │ 71.72 ms      │ 72.33 ms      │ 100     │ 100
│  │  ╰─ 32                                          109.7 ms      │ 183.1 ms      │ 125 ms        │ 125.9 ms      │ 100     │ 100
│  ╰─ skipmap_32b                                                  │               │               │               │         │
│     ├─ 1                                           50.87 ms      │ 60.31 ms      │ 53.37 ms      │ 53.91 ms      │ 100     │ 100
│     ├─ 2                                           52.66 ms      │ 65.76 ms      │ 54.11 ms      │ 54.71 ms      │ 100     │ 100
│     ├─ 4                                           55.3 ms       │ 72.53 ms      │ 59.4 ms       │ 59.95 ms      │ 100     │ 100
│     ├─ 8                                           66.79 ms      │ 84.54 ms      │ 67.87 ms      │ 68.22 ms      │ 100     │ 100
│     ├─ 16                                          116.2 ms      │ 135.1 ms      │ 123.8 ms      │ 123.2 ms      │ 100     │ 100
│     ╰─ 32                                          205.9 ms      │ 266.6 ms      │ 214.5 ms      │ 216.2 ms      │ 100     │ 100
├─ 08a_read_scaling_8B                                             │               │               │               │         │
│  ├─ congee                                                       │               │               │               │         │
│  │  ├─ 1                                           952.7 µs      │ 2.11 ms       │ 1.026 ms      │ 1.161 ms      │ 100     │ 100
│  │  │                                              52.47 Mitem/s │ 23.69 Mitem/s │ 48.72 Mitem/s │ 43.04 Mitem/s │         │
│  │  ├─ 2                                           977.6 µs      │ 1.947 ms      │ 1.41 ms       │ 1.317 ms      │ 100     │ 100
│  │  │                                              102.2 Mitem/s │ 51.35 Mitem/s │ 70.87 Mitem/s │ 75.89 Mitem/s │         │
│  │  ├─ 4                                           1.379 ms      │ 1.938 ms      │ 1.522 ms      │ 1.545 ms      │ 100     │ 100
│  │  │                                              144.9 Mitem/s │ 103.1 Mitem/s │ 131.3 Mitem/s │ 129.4 Mitem/s │         │
│  │  ├─ 8                                           1.635 ms      │ 2.957 ms      │ 1.787 ms      │ 2.028 ms      │ 100     │ 100
│  │  │                                              244.6 Mitem/s │ 135.2 Mitem/s │ 223.7 Mitem/s │ 197.2 Mitem/s │         │
│  │  ├─ 16                                          2.694 ms      │ 3.842 ms      │ 2.977 ms      │ 3.048 ms      │ 100     │ 100
│  │  │                                              296.8 Mitem/s │ 208.2 Mitem/s │ 268.6 Mitem/s │ 262.4 Mitem/s │         │
│  │  ╰─ 32                                          4.955 ms      │ 6.649 ms      │ 5.461 ms      │ 5.491 ms      │ 100     │ 100
│  │                                                 322.9 Mitem/s │ 240.6 Mitem/s │ 292.9 Mitem/s │ 291.3 Mitem/s │         │
│  ├─ dashmap                                                      │               │               │               │         │
│  │  ├─ 1                                           3.696 ms      │ 10.33 ms      │ 4.193 ms      │ 4.821 ms      │ 100     │ 100
│  │  │                                              13.52 Mitem/s │ 4.836 Mitem/s │ 11.92 Mitem/s │ 10.36 Mitem/s │         │
│  │  ├─ 2                                           5.231 ms      │ 14 ms         │ 9.182 ms      │ 9.081 ms      │ 100     │ 100
│  │  │                                              19.11 Mitem/s │ 7.14 Mitem/s  │ 10.89 Mitem/s │ 11.01 Mitem/s │         │
│  │  ├─ 4                                           8.112 ms      │ 14.29 ms      │ 9.904 ms      │ 10.09 ms      │ 100     │ 100
│  │  │                                              24.65 Mitem/s │ 13.99 Mitem/s │ 20.19 Mitem/s │ 19.8 Mitem/s  │         │
│  │  ├─ 8                                           10.5 ms       │ 13.82 ms      │ 11.15 ms      │ 11.23 ms      │ 100     │ 100
│  │  │                                              38.06 Mitem/s │ 28.93 Mitem/s │ 35.87 Mitem/s │ 35.59 Mitem/s │         │
│  │  ├─ 16                                          19.22 ms      │ 23.85 ms      │ 21.9 ms       │ 21.86 ms      │ 100     │ 100
│  │  │                                              41.61 Mitem/s │ 33.53 Mitem/s │ 36.52 Mitem/s │ 36.58 Mitem/s │         │
│  │  ╰─ 32                                          33.59 ms      │ 43.47 ms      │ 36.08 ms      │ 36.39 ms      │ 100     │ 100
│  │                                                 47.63 Mitem/s │ 36.8 Mitem/s  │ 44.33 Mitem/s │ 43.96 Mitem/s │         │
│  ├─ indexset                                                     │               │               │               │         │
│  │  ├─ 1                                           3.947 ms      │ 6.887 ms      │ 4.282 ms      │ 4.602 ms      │ 100     │ 100
│  │  │                                              12.66 Mitem/s │ 7.259 Mitem/s │ 11.67 Mitem/s │ 10.86 Mitem/s │         │
│  │  ├─ 2                                           4.344 ms      │ 8.78 ms       │ 5.179 ms      │ 5.421 ms      │ 100     │ 100
│  │  │                                              23.01 Mitem/s │ 11.38 Mitem/s │ 19.3 Mitem/s  │ 18.44 Mitem/s │         │
│  │  ├─ 4                                           6.717 ms      │ 9.788 ms      │ 7.014 ms      │ 7.326 ms      │ 100     │ 100
│  │  │                                              29.77 Mitem/s │ 20.43 Mitem/s │ 28.51 Mitem/s │ 27.29 Mitem/s │         │
│  │  ├─ 8                                           8.559 ms      │ 11.83 ms      │ 9.61 ms       │ 9.629 ms      │ 100     │ 100
│  │  │                                              46.72 Mitem/s │ 33.81 Mitem/s │ 41.62 Mitem/s │ 41.53 Mitem/s │         │
│  │  ├─ 16                                          14.73 ms      │ 19.83 ms      │ 17.13 ms      │ 17.24 ms      │ 100     │ 100
│  │  │                                              54.29 Mitem/s │ 40.34 Mitem/s │ 46.68 Mitem/s │ 46.38 Mitem/s │         │
│  │  ╰─ 32                                          23.23 ms      │ 30.74 ms      │ 26.61 ms      │ 26.71 ms      │ 100     │ 100
│  │                                                 68.84 Mitem/s │ 52.04 Mitem/s │ 60.11 Mitem/s │ 59.88 Mitem/s │         │
│  ├─ masstree                                                     │               │               │               │         │
│  │  ├─ 1                                           3.253 ms      │ 5.21 ms       │ 3.369 ms      │ 3.555 ms      │ 100     │ 100
│  │  │                                              15.36 Mitem/s │ 9.595 Mitem/s │ 14.83 Mitem/s │ 14.06 Mitem/s │         │
│  │  ├─ 2                                           3.284 ms      │ 5.82 ms       │ 3.363 ms      │ 3.595 ms      │ 100     │ 100
│  │  │                                              30.44 Mitem/s │ 17.18 Mitem/s │ 29.73 Mitem/s │ 27.8 Mitem/s  │         │
│  │  ├─ 4                                           3.333 ms      │ 6.137 ms      │ 3.431 ms      │ 4.016 ms      │ 100     │ 100
│  │  │                                              59.99 Mitem/s │ 32.58 Mitem/s │ 58.28 Mitem/s │ 49.79 Mitem/s │         │
│  │  ├─ 8                                           5.292 ms      │ 7.87 ms       │ 5.371 ms      │ 5.407 ms      │ 100     │ 100
│  │  │                                              75.57 Mitem/s │ 50.82 Mitem/s │ 74.46 Mitem/s │ 73.96 Mitem/s │         │
│  │  ├─ 16                                          8.764 ms      │ 10.71 ms      │ 9.983 ms      │ 9.813 ms      │ 100     │ 100
│  │  │                                              91.27 Mitem/s │ 74.64 Mitem/s │ 80.13 Mitem/s │ 81.51 Mitem/s │         │
│  │  ╰─ 32                                          15.6 ms       │ 18.5 ms       │ 16.61 ms      │ 16.74 ms      │ 100     │ 100
│  │                                                 102.5 Mitem/s │ 86.45 Mitem/s │ 96.28 Mitem/s │ 95.55 Mitem/s │         │
│  ├─ skiplist_guarded                                             │               │               │               │         │
│  │  ├─ 1                                           4.089 ms      │ 6.321 ms      │ 4.19 ms       │ 4.456 ms      │ 100     │ 100
│  │  │                                              12.22 Mitem/s │ 7.909 Mitem/s │ 11.93 Mitem/s │ 11.21 Mitem/s │         │
│  │  ├─ 2                                           4.124 ms      │ 6.323 ms      │ 4.2 ms        │ 4.309 ms      │ 100     │ 100
│  │  │                                              24.24 Mitem/s │ 15.81 Mitem/s │ 23.8 Mitem/s  │ 23.2 Mitem/s  │         │
│  │  ├─ 4                                           4.199 ms      │ 7.947 ms      │ 7.262 ms      │ 6.257 ms      │ 100     │ 100
│  │  │                                              47.62 Mitem/s │ 25.16 Mitem/s │ 27.53 Mitem/s │ 31.96 Mitem/s │         │
│  │  ├─ 8                                           6.543 ms      │ 7.773 ms      │ 6.657 ms      │ 6.731 ms      │ 100     │ 100
│  │  │                                              61.12 Mitem/s │ 51.45 Mitem/s │ 60.08 Mitem/s │ 59.42 Mitem/s │         │
│  │  ├─ 16                                          10.69 ms      │ 13.95 ms      │ 12.08 ms      │ 11.94 ms      │ 100     │ 100
│  │  │                                              74.8 Mitem/s  │ 57.34 Mitem/s │ 66.19 Mitem/s │ 66.98 Mitem/s │         │
│  │  ╰─ 32                                          18.62 ms      │ 24.03 ms      │ 19.73 ms      │ 19.96 ms      │ 100     │ 100
│  │                                                 85.91 Mitem/s │ 66.55 Mitem/s │ 81.07 Mitem/s │ 80.13 Mitem/s │         │
│  ╰─ skipmap                                                      │               │               │               │         │
│     ├─ 1                                           3.984 ms      │ 6.856 ms      │ 4.12 ms       │ 4.249 ms      │ 100     │ 100
│     │                                              12.54 Mitem/s │ 7.291 Mitem/s │ 12.13 Mitem/s │ 11.76 Mitem/s │         │
│     ├─ 2                                           4.104 ms      │ 8.401 ms      │ 4.423 ms      │ 4.784 ms      │ 100     │ 100
│     │                                              24.36 Mitem/s │ 11.9 Mitem/s  │ 22.6 Mitem/s  │ 20.9 Mitem/s  │         │
│     ├─ 4                                           4.496 ms      │ 9.437 ms      │ 4.669 ms      │ 5.183 ms      │ 100     │ 100
│     │                                              44.47 Mitem/s │ 21.19 Mitem/s │ 42.82 Mitem/s │ 38.58 Mitem/s │         │
│     ├─ 8                                           7.21 ms       │ 10.14 ms      │ 7.35 ms       │ 7.519 ms      │ 100     │ 100
│     │                                              55.47 Mitem/s │ 39.42 Mitem/s │ 54.41 Mitem/s │ 53.19 Mitem/s │         │
│     ├─ 16                                          11.31 ms      │ 15.12 ms      │ 13.12 ms      │ 13.21 ms      │ 100     │ 100
│     │                                              70.68 Mitem/s │ 52.88 Mitem/s │ 60.97 Mitem/s │ 60.55 Mitem/s │         │
│     ╰─ 32                                          19.94 ms      │ 29.35 ms      │ 22.38 ms      │ 22.98 ms      │ 100     │ 100
│                                                    80.2 Mitem/s  │ 54.51 Mitem/s │ 71.46 Mitem/s │ 69.62 Mitem/s │         │
├─ 08b_read_scaling_32B                                            │               │               │               │         │
│  ├─ dashmap                                                      │               │               │               │         │
│  │  ├─ 1                                           6.242 ms      │ 11.22 ms      │ 8.752 ms      │ 8.807 ms      │ 100     │ 100
│  │  │                                              8.009 Mitem/s │ 4.454 Mitem/s │ 5.712 Mitem/s │ 5.676 Mitem/s │         │
│  │  ├─ 2                                           7.765 ms      │ 15.11 ms      │ 9.855 ms      │ 10.11 ms      │ 100     │ 100
│  │  │                                              12.87 Mitem/s │ 6.616 Mitem/s │ 10.14 Mitem/s │ 9.889 Mitem/s │         │
│  │  ├─ 4                                           9.74 ms       │ 13.36 ms      │ 11.31 ms      │ 11.27 ms      │ 100     │ 100
│  │  │                                              20.53 Mitem/s │ 14.96 Mitem/s │ 17.66 Mitem/s │ 17.73 Mitem/s │         │
│  │  ├─ 8                                           12.34 ms      │ 14.34 ms      │ 12.85 ms      │ 12.94 ms      │ 100     │ 100
│  │  │                                              32.39 Mitem/s │ 27.87 Mitem/s │ 31.11 Mitem/s │ 30.89 Mitem/s │         │
│  │  ├─ 16                                          21.31 ms      │ 27.08 ms      │ 24.1 ms       │ 24.04 ms      │ 100     │ 100
│  │  │                                              37.53 Mitem/s │ 29.53 Mitem/s │ 33.18 Mitem/s │ 33.26 Mitem/s │         │
│  │  ╰─ 32                                          37.63 ms      │ 44.82 ms      │ 40.14 ms      │ 40.26 ms      │ 100     │ 100
│  │                                                 42.5 Mitem/s  │ 35.69 Mitem/s │ 39.86 Mitem/s │ 39.73 Mitem/s │         │
│  ├─ indexset                                                     │               │               │               │         │
│  │  ├─ 1                                           7.685 ms      │ 11.18 ms      │ 8.005 ms      │ 8.468 ms      │ 100     │ 100
│  │  │                                              6.505 Mitem/s │ 4.469 Mitem/s │ 6.246 Mitem/s │ 5.904 Mitem/s │         │
│  │  ├─ 2                                           8.059 ms      │ 14.19 ms      │ 9.061 ms      │ 9.537 ms      │ 100     │ 100
│  │  │                                              12.4 Mitem/s  │ 7.047 Mitem/s │ 11.03 Mitem/s │ 10.48 Mitem/s │         │
│  │  ├─ 4                                           9.775 ms      │ 16.52 ms      │ 10.6 ms       │ 11.75 ms      │ 100     │ 100
│  │  │                                              20.45 Mitem/s │ 12.1 Mitem/s  │ 18.85 Mitem/s │ 17.01 Mitem/s │         │
│  │  ├─ 8                                           15.26 ms      │ 19.21 ms      │ 16.79 ms      │ 16.84 ms      │ 100     │ 100
│  │  │                                              26.19 Mitem/s │ 20.81 Mitem/s │ 23.81 Mitem/s │ 23.73 Mitem/s │         │
│  │  ├─ 16                                          23.49 ms      │ 33.6 ms       │ 27.39 ms      │ 27.71 ms      │ 100     │ 100
│  │  │                                              34.04 Mitem/s │ 23.8 Mitem/s  │ 29.2 Mitem/s  │ 28.86 Mitem/s │         │
│  │  ╰─ 32                                          42.27 ms      │ 50.04 ms      │ 45.81 ms      │ 45.93 ms      │ 100     │ 100
│  │                                                 37.84 Mitem/s │ 31.96 Mitem/s │ 34.92 Mitem/s │ 34.83 Mitem/s │         │
│  ├─ masstree                                                     │               │               │               │         │
│  │  ├─ 1                                           3.408 ms      │ 6.409 ms      │ 3.55 ms       │ 3.907 ms      │ 100     │ 100
│  │  │                                              14.66 Mitem/s │ 7.801 Mitem/s │ 14.08 Mitem/s │ 12.79 Mitem/s │         │
│  │  ├─ 2                                           3.492 ms      │ 6.325 ms      │ 3.702 ms      │ 4.015 ms      │ 100     │ 100
│  │  │                                              28.63 Mitem/s │ 15.8 Mitem/s  │ 27 Mitem/s    │ 24.9 Mitem/s  │         │
│  │  ├─ 4                                           3.554 ms      │ 6.203 ms      │ 3.793 ms      │ 4.058 ms      │ 100     │ 100
│  │  │                                              56.26 Mitem/s │ 32.23 Mitem/s │ 52.71 Mitem/s │ 49.27 Mitem/s │         │
│  │  ├─ 8                                           5.845 ms      │ 8.546 ms      │ 6.549 ms      │ 6.491 ms      │ 100     │ 100
│  │  │                                              68.42 Mitem/s │ 46.8 Mitem/s  │ 61.07 Mitem/s │ 61.61 Mitem/s │         │
│  │  ├─ 16                                          10.06 ms      │ 12.67 ms      │ 11.39 ms      │ 11.32 ms      │ 100     │ 100
│  │  │                                              79.47 Mitem/s │ 63.13 Mitem/s │ 70.22 Mitem/s │ 70.62 Mitem/s │         │
│  │  ╰─ 32                                          18.11 ms      │ 21.28 ms      │ 19.2 ms       │ 19.19 ms      │ 100     │ 100
│  │                                                 88.3 Mitem/s  │ 75.17 Mitem/s │ 83.32 Mitem/s │ 83.34 Mitem/s │         │
│  ├─ skiplist_guarded                                             │               │               │               │         │
│  │  ├─ 1                                           8.86 ms       │ 12.1 ms       │ 9.092 ms      │ 9.465 ms      │ 100     │ 100
│  │  │                                              5.643 Mitem/s │ 4.13 Mitem/s  │ 5.498 Mitem/s │ 5.282 Mitem/s │         │
│  │  ├─ 2                                           9.034 ms      │ 15.93 ms      │ 9.26 ms       │ 9.993 ms      │ 100     │ 100
│  │  │                                              11.06 Mitem/s │ 6.273 Mitem/s │ 10.79 Mitem/s │ 10 Mitem/s    │         │
│  │  ├─ 4                                           9.218 ms      │ 19.78 ms      │ 9.794 ms      │ 10.83 ms      │ 100     │ 100
│  │  │                                              21.69 Mitem/s │ 10.1 Mitem/s  │ 20.41 Mitem/s │ 18.45 Mitem/s │         │
│  │  ├─ 8                                           15.7 ms       │ 18.7 ms       │ 15.95 ms      │ 16 ms         │ 100     │ 100
│  │  │                                              25.46 Mitem/s │ 21.37 Mitem/s │ 25.06 Mitem/s │ 24.98 Mitem/s │         │
│  │  ├─ 16                                          24.97 ms      │ 31.46 ms      │ 29.75 ms      │ 29.31 ms      │ 100     │ 100
│  │  │                                              32.03 Mitem/s │ 25.42 Mitem/s │ 26.88 Mitem/s │ 27.28 Mitem/s │         │
│  │  ╰─ 32                                          46.48 ms      │ 57.47 ms      │ 49.51 ms      │ 49.93 ms      │ 100     │ 100
│  │                                                 34.42 Mitem/s │ 27.83 Mitem/s │ 32.31 Mitem/s │ 32.04 Mitem/s │         │
│  ╰─ skipmap                                                      │               │               │               │         │
│     ├─ 1                                           9.369 ms      │ 13.13 ms      │ 9.73 ms       │ 10.2 ms       │ 100     │ 100
│     │                                              5.336 Mitem/s │ 3.807 Mitem/s │ 5.138 Mitem/s │ 4.897 Mitem/s │         │
│     ├─ 2                                           9.595 ms      │ 15.63 ms      │ 9.956 ms      │ 10.59 ms      │ 100     │ 100
│     │                                              10.42 Mitem/s │ 6.396 Mitem/s │ 10.04 Mitem/s │ 9.442 Mitem/s │         │
│     ├─ 4                                           9.854 ms      │ 15.93 ms      │ 10.25 ms      │ 11.52 ms      │ 100     │ 100
│     │                                              20.29 Mitem/s │ 12.55 Mitem/s │ 19.49 Mitem/s │ 17.35 Mitem/s │         │
│     ├─ 8                                           16.69 ms      │ 18.91 ms      │ 17.08 ms      │ 17.15 ms      │ 100     │ 100
│     │                                              23.95 Mitem/s │ 21.14 Mitem/s │ 23.41 Mitem/s │ 23.31 Mitem/s │         │
│     ├─ 16                                          26.61 ms      │ 33.31 ms      │ 31.83 ms      │ 31.35 ms      │ 100     │ 100
│     │                                              30.05 Mitem/s │ 24.01 Mitem/s │ 25.13 Mitem/s │ 25.51 Mitem/s │         │
│     ╰─ 32                                          49.35 ms      │ 57.94 ms      │ 52.96 ms      │ 53.12 ms      │ 100     │ 100
│                                                    32.42 Mitem/s │ 27.61 Mitem/s │ 30.2 Mitem/s  │ 30.11 Mitem/s │         │
```
