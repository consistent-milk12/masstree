```text
Timer precision: 20 ns
range_concurrent_scaled        fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.475 ms      │ 6.971 ms      │ 4.64 ms       │ 4.988 ms      │ 100     │ 100
│  │  │                        2.234 Mitem/s │ 1.434 Mitem/s │ 2.154 Mitem/s │ 2.004 Mitem/s │         │
│  │  ├─ 2                     11.48 ms      │ 21.28 ms      │ 15.3 ms       │ 15.42 ms      │ 100     │ 100
│  │  │                        1.741 Mitem/s │ 939.8 Kitem/s │ 1.306 Mitem/s │ 1.296 Mitem/s │         │
│  │  ├─ 4                     35.55 ms      │ 58.7 ms       │ 44.69 ms      │ 45 ms         │ 100     │ 100
│  │  │                        1.124 Mitem/s │ 681.3 Kitem/s │ 895 Kitem/s   │ 888.7 Kitem/s │         │
│  │  ├─ 8                     161.1 ms      │ 255.8 ms      │ 175 ms        │ 177.2 ms      │ 100     │ 100
│  │  │                        496.3 Kitem/s │ 312.6 Kitem/s │ 457 Kitem/s   │ 451.3 Kitem/s │         │
│  │  ├─ 16                    357.4 ms      │ 508.5 ms      │ 368.8 ms      │ 375.5 ms      │ 100     │ 100
│  │  │                        447.6 Kitem/s │ 314.5 Kitem/s │ 433.7 Kitem/s │ 426 Kitem/s   │         │
│  │  ╰─ 32                    744 ms        │ 929.8 ms      │ 813.8 ms      │ 821.5 ms      │ 100     │ 100
│  │                           430.1 Kitem/s │ 344.1 Kitem/s │ 393.2 Kitem/s │ 389.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.569 ms      │ 11.02 ms      │ 8.002 ms      │ 8.337 ms      │ 100     │ 100
│  │  │                        1.321 Mitem/s │ 907.1 Kitem/s │ 1.249 Mitem/s │ 1.199 Mitem/s │         │
│  │  ├─ 2                     7.932 ms      │ 16.5 ms       │ 11.4 ms       │ 11.51 ms      │ 100     │ 100
│  │  │                        2.521 Mitem/s │ 1.211 Mitem/s │ 1.753 Mitem/s │ 1.736 Mitem/s │         │
│  │  ├─ 4                     8.924 ms      │ 24.05 ms      │ 13.04 ms      │ 12.79 ms      │ 100     │ 100
│  │  │                        4.481 Mitem/s │ 1.662 Mitem/s │ 3.066 Mitem/s │ 3.127 Mitem/s │         │
│  │  ├─ 8                     13.39 ms      │ 24.07 ms      │ 15.04 ms      │ 16.08 ms      │ 100     │ 100
│  │  │                        5.972 Mitem/s │ 3.323 Mitem/s │ 5.317 Mitem/s │ 4.973 Mitem/s │         │
│  │  ├─ 16                    21.37 ms      │ 27.51 ms      │ 23.29 ms      │ 23.4 ms       │ 100     │ 100
│  │  │                        7.485 Mitem/s │ 5.815 Mitem/s │ 6.867 Mitem/s │ 6.835 Mitem/s │         │
│  │  ╰─ 32                    40.47 ms      │ 49.38 ms      │ 43.6 ms       │ 43.54 ms      │ 100     │ 100
│  │                           7.905 Mitem/s │ 6.479 Mitem/s │ 7.337 Mitem/s │ 7.349 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.945 ms      │ 3.774 ms      │ 2.045 ms      │ 2.323 ms      │ 100     │ 100
│  │  │                        5.139 Mitem/s │ 2.649 Mitem/s │ 4.887 Mitem/s │ 4.304 Mitem/s │         │
│  │  ├─ 2                     2.178 ms      │ 4.444 ms      │ 2.991 ms      │ 3.194 ms      │ 100     │ 100
│  │  │                        9.179 Mitem/s │ 4.499 Mitem/s │ 6.685 Mitem/s │ 6.26 Mitem/s  │         │
│  │  ├─ 4                     2.95 ms       │ 5.15 ms       │ 4.149 ms      │ 3.883 ms      │ 100     │ 100
│  │  │                        13.55 Mitem/s │ 7.766 Mitem/s │ 9.64 Mitem/s  │ 10.29 Mitem/s │         │
│  │  ├─ 8                     4.536 ms      │ 7.318 ms      │ 4.8 ms        │ 4.986 ms      │ 100     │ 100
│  │  │                        17.63 Mitem/s │ 10.93 Mitem/s │ 16.66 Mitem/s │ 16.04 Mitem/s │         │
│  │  ├─ 16                    7.542 ms      │ 10.2 ms       │ 8.463 ms      │ 8.562 ms      │ 100     │ 100
│  │  │                        21.21 Mitem/s │ 15.68 Mitem/s │ 18.9 Mitem/s  │ 18.68 Mitem/s │         │
│  │  ╰─ 32                    14.91 ms      │ 17.56 ms      │ 15.78 ms      │ 15.82 ms      │ 100     │ 100
│  │                           21.45 Mitem/s │ 18.21 Mitem/s │ 20.27 Mitem/s │ 20.21 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.09 ms      │ 15.2 ms       │ 10.58 ms      │ 11.52 ms      │ 100     │ 100
│     │                        990.9 Kitem/s │ 657.7 Kitem/s │ 944.4 Kitem/s │ 867.8 Kitem/s │         │
│     ├─ 2                     10.15 ms      │ 22.27 ms      │ 14.68 ms      │ 14.5 ms       │ 100     │ 100
│     │                        1.97 Mitem/s  │ 897.6 Kitem/s │ 1.362 Mitem/s │ 1.378 Mitem/s │         │
│     ├─ 4                     10.69 ms      │ 22.24 ms      │ 15.8 ms       │ 16.31 ms      │ 100     │ 100
│     │                        3.738 Mitem/s │ 1.797 Mitem/s │ 2.53 Mitem/s  │ 2.452 Mitem/s │         │
│     ├─ 8                     19.32 ms      │ 38.54 ms      │ 20.99 ms      │ 24.13 ms      │ 100     │ 100
│     │                        4.14 Mitem/s  │ 2.075 Mitem/s │ 3.809 Mitem/s │ 3.315 Mitem/s │         │
│     ├─ 16                    29.71 ms      │ 36.55 ms      │ 32.32 ms      │ 32.69 ms      │ 100     │ 100
│     │                        5.383 Mitem/s │ 4.377 Mitem/s │ 4.95 Mitem/s  │ 4.893 Mitem/s │         │
│     ╰─ 32                    56.45 ms      │ 64.61 ms      │ 59.46 ms      │ 59.6 ms       │ 100     │ 100
│                              5.668 Mitem/s │ 4.952 Mitem/s │ 5.381 Mitem/s │ 5.368 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.699 ms      │ 8.568 ms      │ 4.794 ms      │ 4.955 ms      │ 100     │ 100
│  │  │                        2.128 Mitem/s │ 1.167 Mitem/s │ 2.085 Mitem/s │ 2.018 Mitem/s │         │
│  │  ├─ 2                     11.72 ms      │ 21.51 ms      │ 16.73 ms      │ 16.89 ms      │ 100     │ 100
│  │  │                        1.705 Mitem/s │ 929.6 Kitem/s │ 1.195 Mitem/s │ 1.183 Mitem/s │         │
│  │  ├─ 4                     36.77 ms      │ 58.69 ms      │ 45.03 ms      │ 45.76 ms      │ 100     │ 100
│  │  │                        1.087 Mitem/s │ 681.5 Kitem/s │ 888.2 Kitem/s │ 874 Kitem/s   │         │
│  │  ├─ 8                     161.1 ms      │ 269.9 ms      │ 175.3 ms      │ 176.6 ms      │ 100     │ 100
│  │  │                        496.3 Kitem/s │ 296.3 Kitem/s │ 456.3 Kitem/s │ 452.8 Kitem/s │         │
│  │  ├─ 16                    360.8 ms      │ 527.1 ms      │ 378.4 ms      │ 386.3 ms      │ 100     │ 100
│  │  │                        443.3 Kitem/s │ 303.5 Kitem/s │ 422.8 Kitem/s │ 414.1 Kitem/s │         │
│  │  ╰─ 32                    726.7 ms      │ 1.103 s       │ 808.2 ms      │ 814.5 ms      │ 100     │ 100
│  │                           440.2 Kitem/s │ 289.8 Kitem/s │ 395.9 Kitem/s │ 392.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.459 ms      │ 9.559 ms      │ 7.715 ms      │ 7.949 ms      │ 100     │ 100
│  │  │                        1.34 Mitem/s  │ 1.046 Mitem/s │ 1.296 Mitem/s │ 1.257 Mitem/s │         │
│  │  ├─ 2                     7.831 ms      │ 15.46 ms      │ 12.08 ms      │ 12.17 ms      │ 100     │ 100
│  │  │                        2.553 Mitem/s │ 1.293 Mitem/s │ 1.655 Mitem/s │ 1.642 Mitem/s │         │
│  │  ├─ 4                     8.686 ms      │ 19.58 ms      │ 13.6 ms       │ 13.2 ms       │ 100     │ 100
│  │  │                        4.605 Mitem/s │ 2.042 Mitem/s │ 2.939 Mitem/s │ 3.03 Mitem/s  │         │
│  │  ├─ 8                     13.53 ms      │ 25.41 ms      │ 15.26 ms      │ 16.45 ms      │ 100     │ 100
│  │  │                        5.908 Mitem/s │ 3.147 Mitem/s │ 5.24 Mitem/s  │ 4.862 Mitem/s │         │
│  │  ├─ 16                    21.6 ms       │ 28.49 ms      │ 23.46 ms      │ 23.73 ms      │ 100     │ 100
│  │  │                        7.405 Mitem/s │ 5.614 Mitem/s │ 6.817 Mitem/s │ 6.739 Mitem/s │         │
│  │  ╰─ 32                    41.4 ms       │ 47.99 ms      │ 43.79 ms      │ 43.79 ms      │ 100     │ 100
│  │                           7.728 Mitem/s │ 6.666 Mitem/s │ 7.306 Mitem/s │ 7.306 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.876 ms      │ 3.698 ms      │ 1.968 ms      │ 2.188 ms      │ 100     │ 100
│  │  │                        5.329 Mitem/s │ 2.703 Mitem/s │ 5.08 Mitem/s  │ 4.569 Mitem/s │         │
│  │  ├─ 2                     2.096 ms      │ 4.288 ms      │ 2.817 ms      │ 2.765 ms      │ 100     │ 100
│  │  │                        9.541 Mitem/s │ 4.663 Mitem/s │ 7.097 Mitem/s │ 7.231 Mitem/s │         │
│  │  ├─ 4                     2.83 ms       │ 4.988 ms      │ 4.016 ms      │ 3.745 ms      │ 100     │ 100
│  │  │                        14.13 Mitem/s │ 8.018 Mitem/s │ 9.96 Mitem/s  │ 10.67 Mitem/s │         │
│  │  ├─ 8                     4.415 ms      │ 7.086 ms      │ 4.589 ms      │ 4.829 ms      │ 100     │ 100
│  │  │                        18.11 Mitem/s │ 11.28 Mitem/s │ 17.42 Mitem/s │ 16.56 Mitem/s │         │
│  │  ├─ 16                    7.151 ms      │ 9.271 ms      │ 7.781 ms      │ 7.888 ms      │ 100     │ 100
│  │  │                        22.37 Mitem/s │ 17.25 Mitem/s │ 20.56 Mitem/s │ 20.28 Mitem/s │         │
│  │  ╰─ 32                    13.96 ms      │ 17.04 ms      │ 14.94 ms      │ 15.03 ms      │ 100     │ 100
│  │                           22.91 Mitem/s │ 18.77 Mitem/s │ 21.41 Mitem/s │ 21.28 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.23 ms      │ 13.72 ms      │ 10.81 ms      │ 11.17 ms      │ 100     │ 100
│     │                        977.2 Kitem/s │ 728.5 Kitem/s │ 924.7 Kitem/s │ 894.6 Kitem/s │         │
│     ├─ 2                     10.57 ms      │ 21.18 ms      │ 12.52 ms      │ 13.37 ms      │ 100     │ 100
│     │                        1.891 Mitem/s │ 943.9 Kitem/s │ 1.596 Mitem/s │ 1.494 Mitem/s │         │
│     ├─ 4                     10.82 ms      │ 30.52 ms      │ 16.46 ms      │ 16.99 ms      │ 100     │ 100
│     │                        3.695 Mitem/s │ 1.31 Mitem/s  │ 2.428 Mitem/s │ 2.353 Mitem/s │         │
│     ├─ 8                     20.06 ms      │ 35.7 ms       │ 26.14 ms      │ 26.54 ms      │ 100     │ 100
│     │                        3.986 Mitem/s │ 2.24 Mitem/s  │ 3.059 Mitem/s │ 3.014 Mitem/s │         │
│     ├─ 16                    31.29 ms      │ 39.62 ms      │ 33.86 ms      │ 34.2 ms       │ 100     │ 100
│     │                        5.112 Mitem/s │ 4.038 Mitem/s │ 4.724 Mitem/s │ 4.678 Mitem/s │         │
│     ╰─ 32                    59.19 ms      │ 129.5 ms      │ 62.4 ms       │ 64.84 ms      │ 100     │ 100
│                              5.406 Mitem/s │ 2.47 Mitem/s  │ 5.127 Mitem/s │ 4.934 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.624 ms      │ 8.279 ms      │ 4.742 ms      │ 4.933 ms      │ 100     │ 100
│  │  │                        2.162 Mitem/s │ 1.207 Mitem/s │ 2.108 Mitem/s │ 2.026 Mitem/s │         │
│  │  ├─ 2                     11.02 ms      │ 21 ms         │ 15.47 ms      │ 15.47 ms      │ 100     │ 100
│  │  │                        1.814 Mitem/s │ 952 Kitem/s   │ 1.292 Mitem/s │ 1.292 Mitem/s │         │
│  │  ├─ 4                     33.43 ms      │ 62.21 ms      │ 43.21 ms      │ 43.75 ms      │ 100     │ 100
│  │  │                        1.196 Mitem/s │ 642.8 Kitem/s │ 925.5 Kitem/s │ 914 Kitem/s   │         │
│  │  ├─ 8                     161.8 ms      │ 191.9 ms      │ 173.5 ms      │ 173.7 ms      │ 100     │ 100
│  │  │                        494.4 Kitem/s │ 416.7 Kitem/s │ 461 Kitem/s   │ 460.4 Kitem/s │         │
│  │  ├─ 16                    354.4 ms      │ 527.2 ms      │ 362.4 ms      │ 368.2 ms      │ 100     │ 100
│  │  │                        451.4 Kitem/s │ 303.4 Kitem/s │ 441.5 Kitem/s │ 434.4 Kitem/s │         │
│  │  ╰─ 32                    754.5 ms      │ 1.056 s       │ 827.6 ms      │ 836.9 ms      │ 100     │ 100
│  │                           424.1 Kitem/s │ 303 Kitem/s   │ 386.6 Kitem/s │ 382.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.654 ms      │ 10.95 ms      │ 7.828 ms      │ 8.017 ms      │ 100     │ 100
│  │  │                        1.306 Mitem/s │ 912.7 Kitem/s │ 1.277 Mitem/s │ 1.247 Mitem/s │         │
│  │  ├─ 2                     7.962 ms      │ 16.3 ms       │ 11.76 ms      │ 11.93 ms      │ 100     │ 100
│  │  │                        2.511 Mitem/s │ 1.226 Mitem/s │ 1.7 Mitem/s   │ 1.675 Mitem/s │         │
│  │  ├─ 4                     8.651 ms      │ 21.39 ms      │ 13.82 ms      │ 13.23 ms      │ 100     │ 100
│  │  │                        4.623 Mitem/s │ 1.869 Mitem/s │ 2.892 Mitem/s │ 3.022 Mitem/s │         │
│  │  ├─ 8                     13.62 ms      │ 25.76 ms      │ 16.49 ms      │ 17.62 ms      │ 100     │ 100
│  │  │                        5.87 Mitem/s  │ 3.105 Mitem/s │ 4.848 Mitem/s │ 4.538 Mitem/s │         │
│  │  ├─ 16                    22.11 ms      │ 54.31 ms      │ 24.32 ms      │ 25.68 ms      │ 100     │ 100
│  │  │                        7.236 Mitem/s │ 2.945 Mitem/s │ 6.577 Mitem/s │ 6.229 Mitem/s │         │
│  │  ╰─ 32                    41.13 ms      │ 48.57 ms      │ 44.04 ms      │ 44.02 ms      │ 100     │ 100
│  │                           7.778 Mitem/s │ 6.588 Mitem/s │ 7.265 Mitem/s │ 7.267 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.925 ms      │ 3.558 ms      │ 2.686 ms      │ 2.471 ms      │ 100     │ 100
│  │  │                        5.194 Mitem/s │ 2.81 Mitem/s  │ 3.721 Mitem/s │ 4.046 Mitem/s │         │
│  │  ├─ 2                     2.029 ms      │ 4.692 ms      │ 3.589 ms      │ 3.526 ms      │ 100     │ 100
│  │  │                        9.855 Mitem/s │ 4.262 Mitem/s │ 5.571 Mitem/s │ 5.67 Mitem/s  │         │
│  │  ├─ 4                     2.265 ms      │ 6.119 ms      │ 3.874 ms      │ 3.748 ms      │ 100     │ 100
│  │  │                        17.65 Mitem/s │ 6.536 Mitem/s │ 10.32 Mitem/s │ 10.67 Mitem/s │         │
│  │  ├─ 8                     4.567 ms      │ 7.389 ms      │ 5.029 ms      │ 5.141 ms      │ 100     │ 100
│  │  │                        17.51 Mitem/s │ 10.82 Mitem/s │ 15.9 Mitem/s  │ 15.55 Mitem/s │         │
│  │  ├─ 16                    7.162 ms      │ 9.646 ms      │ 8.4 ms        │ 8.437 ms      │ 100     │ 100
│  │  │                        22.33 Mitem/s │ 16.58 Mitem/s │ 19.04 Mitem/s │ 18.96 Mitem/s │         │
│  │  ╰─ 32                    14.78 ms      │ 17.83 ms      │ 15.6 ms       │ 15.74 ms      │ 100     │ 100
│  │                           21.64 Mitem/s │ 17.94 Mitem/s │ 20.5 Mitem/s  │ 20.32 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.998 ms      │ 15.79 ms      │ 10.58 ms      │ 11.05 ms      │ 100     │ 100
│     │                        1 Mitem/s     │ 632.9 Kitem/s │ 944.6 Kitem/s │ 904.4 Kitem/s │         │
│     ├─ 2                     10.34 ms      │ 22.71 ms      │ 11.86 ms      │ 12.97 ms      │ 100     │ 100
│     │                        1.933 Mitem/s │ 880.5 Kitem/s │ 1.685 Mitem/s │ 1.541 Mitem/s │         │
│     ├─ 4                     11.09 ms      │ 22.13 ms      │ 16.53 ms      │ 17.47 ms      │ 100     │ 100
│     │                        3.604 Mitem/s │ 1.807 Mitem/s │ 2.419 Mitem/s │ 2.288 Mitem/s │         │
│     ├─ 8                     19.71 ms      │ 35.11 ms      │ 21.15 ms      │ 23.47 ms      │ 100     │ 100
│     │                        4.057 Mitem/s │ 2.277 Mitem/s │ 3.78 Mitem/s  │ 3.407 Mitem/s │         │
│     ├─ 16                    30.5 ms       │ 37.41 ms      │ 32.81 ms      │ 32.96 ms      │ 100     │ 100
│     │                        5.245 Mitem/s │ 4.276 Mitem/s │ 4.875 Mitem/s │ 4.854 Mitem/s │         │
│     ╰─ 32                    58.1 ms       │ 64.37 ms      │ 60.74 ms      │ 60.86 ms      │ 100     │ 100
│                              5.507 Mitem/s │ 4.97 Mitem/s  │ 5.267 Mitem/s │ 5.257 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.548 ms      │ 6.995 ms      │ 4.657 ms      │ 4.896 ms      │ 100     │ 100
│  │  │                        2.198 Mitem/s │ 1.429 Mitem/s │ 2.147 Mitem/s │ 2.042 Mitem/s │         │
│  │  ├─ 2                     11.01 ms      │ 21.12 ms      │ 14.88 ms      │ 14.91 ms      │ 100     │ 100
│  │  │                        1.815 Mitem/s │ 946.6 Kitem/s │ 1.344 Mitem/s │ 1.34 Mitem/s  │         │
│  │  ├─ 4                     33.99 ms      │ 58.84 ms      │ 42.72 ms      │ 43.84 ms      │ 100     │ 100
│  │  │                        1.176 Mitem/s │ 679.7 Kitem/s │ 936.2 Kitem/s │ 912.3 Kitem/s │         │
│  │  ├─ 8                     158.3 ms      │ 203.7 ms      │ 173.6 ms      │ 174.3 ms      │ 100     │ 100
│  │  │                        505 Kitem/s   │ 392.6 Kitem/s │ 460.8 Kitem/s │ 458.9 Kitem/s │         │
│  │  ├─ 16                    357.7 ms      │ 595.2 ms      │ 372.4 ms      │ 377.9 ms      │ 100     │ 100
│  │  │                        447.2 Kitem/s │ 268.7 Kitem/s │ 429.6 Kitem/s │ 423.3 Kitem/s │         │
│  │  ╰─ 32                    749.5 ms      │ 998.6 ms      │ 824 ms        │ 837.2 ms      │ 100     │ 100
│  │                           426.8 Kitem/s │ 320.4 Kitem/s │ 388.3 Kitem/s │ 382.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.606 ms      │ 10.34 ms      │ 7.899 ms      │ 8.106 ms      │ 100     │ 100
│  │  │                        1.314 Mitem/s │ 966.6 Kitem/s │ 1.265 Mitem/s │ 1.233 Mitem/s │         │
│  │  ├─ 2                     7.945 ms      │ 16.12 ms      │ 10.07 ms      │ 11.08 ms      │ 100     │ 100
│  │  │                        2.517 Mitem/s │ 1.24 Mitem/s  │ 1.984 Mitem/s │ 1.803 Mitem/s │         │
│  │  ├─ 4                     8.825 ms      │ 22.52 ms      │ 13.21 ms      │ 12.9 ms       │ 100     │ 100
│  │  │                        4.532 Mitem/s │ 1.775 Mitem/s │ 3.027 Mitem/s │ 3.098 Mitem/s │         │
│  │  ├─ 8                     13.82 ms      │ 23.4 ms       │ 15.17 ms      │ 16.36 ms      │ 100     │ 100
│  │  │                        5.786 Mitem/s │ 3.418 Mitem/s │ 5.272 Mitem/s │ 4.888 Mitem/s │         │
│  │  ├─ 16                    21.94 ms      │ 27.14 ms      │ 23.86 ms      │ 24.06 ms      │ 100     │ 100
│  │  │                        7.291 Mitem/s │ 5.894 Mitem/s │ 6.704 Mitem/s │ 6.647 Mitem/s │         │
│  │  ╰─ 32                    42.07 ms      │ 48.91 ms      │ 45.07 ms      │ 45.18 ms      │ 100     │ 100
│  │                           7.605 Mitem/s │ 6.542 Mitem/s │ 7.099 Mitem/s │ 7.082 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.948 ms      │ 3.502 ms      │ 2.012 ms      │ 2.219 ms      │ 100     │ 100
│  │  │                        5.131 Mitem/s │ 2.855 Mitem/s │ 4.968 Mitem/s │ 4.506 Mitem/s │         │
│  │  ├─ 2                     2.21 ms       │ 4.849 ms      │ 3.037 ms      │ 3.211 ms      │ 100     │ 100
│  │  │                        9.047 Mitem/s │ 4.124 Mitem/s │ 6.584 Mitem/s │ 6.228 Mitem/s │         │
│  │  ├─ 4                     2.927 ms      │ 6.638 ms      │ 3.541 ms      │ 3.665 ms      │ 100     │ 100
│  │  │                        13.66 Mitem/s │ 6.025 Mitem/s │ 11.29 Mitem/s │ 10.91 Mitem/s │         │
│  │  ├─ 8                     4.506 ms      │ 7.424 ms      │ 4.779 ms      │ 4.99 ms       │ 100     │ 100
│  │  │                        17.75 Mitem/s │ 10.77 Mitem/s │ 16.73 Mitem/s │ 16.02 Mitem/s │         │
│  │  ├─ 16                    7.806 ms      │ 24.31 ms      │ 8.957 ms      │ 10.29 ms      │ 100     │ 100
│  │  │                        20.49 Mitem/s │ 6.581 Mitem/s │ 17.86 Mitem/s │ 15.53 Mitem/s │         │
│  │  ╰─ 32                    14.88 ms      │ 17.13 ms      │ 15.79 ms      │ 15.84 ms      │ 100     │ 100
│  │                           21.5 Mitem/s  │ 18.67 Mitem/s │ 20.25 Mitem/s │ 20.19 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.07 ms      │ 14.69 ms      │ 10.54 ms      │ 10.8 ms       │ 100     │ 100
│     │                        992.4 Kitem/s │ 680.5 Kitem/s │ 948.5 Kitem/s │ 925.2 Kitem/s │         │
│     ├─ 2                     10.29 ms      │ 20.75 ms      │ 13.96 ms      │ 13.63 ms      │ 100     │ 100
│     │                        1.942 Mitem/s │ 963.6 Kitem/s │ 1.431 Mitem/s │ 1.467 Mitem/s │         │
│     ├─ 4                     10.61 ms      │ 30.55 ms      │ 16.17 ms      │ 16.66 ms      │ 100     │ 100
│     │                        3.767 Mitem/s │ 1.308 Mitem/s │ 2.473 Mitem/s │ 2.4 Mitem/s   │         │
│     ├─ 8                     19.57 ms      │ 32.7 ms       │ 21.14 ms      │ 23.35 ms      │ 100     │ 100
│     │                        4.087 Mitem/s │ 2.445 Mitem/s │ 3.782 Mitem/s │ 3.425 Mitem/s │         │
│     ├─ 16                    30.39 ms      │ 39.06 ms      │ 33.83 ms      │ 33.78 ms      │ 100     │ 100
│     │                        5.263 Mitem/s │ 4.096 Mitem/s │ 4.728 Mitem/s │ 4.736 Mitem/s │         │
│     ╰─ 32                    57.07 ms      │ 66.96 ms      │ 61.86 ms      │ 61.84 ms      │ 100     │ 100
│                              5.606 Mitem/s │ 4.778 Mitem/s │ 5.172 Mitem/s │ 5.174 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.944 ms      │ 6.459 ms      │ 4.08 ms       │ 4.372 ms      │ 100     │ 100
│  │  │                        2.535 Mitem/s │ 1.548 Mitem/s │ 2.45 Mitem/s  │ 2.286 Mitem/s │         │
│  │  ├─ 2                     8.885 ms      │ 18.49 ms      │ 12.9 ms       │ 12.9 ms       │ 100     │ 100
│  │  │                        2.25 Mitem/s  │ 1.081 Mitem/s │ 1.549 Mitem/s │ 1.549 Mitem/s │         │
│  │  ├─ 4                     31.29 ms      │ 54.56 ms      │ 38.61 ms      │ 38.81 ms      │ 100     │ 100
│  │  │                        1.278 Mitem/s │ 733 Kitem/s   │ 1.035 Mitem/s │ 1.03 Mitem/s  │         │
│  │  ├─ 8                     150.7 ms      │ 248.3 ms      │ 166.9 ms      │ 168.6 ms      │ 100     │ 100
│  │  │                        530.5 Kitem/s │ 322.1 Kitem/s │ 479 Kitem/s   │ 474.4 Kitem/s │         │
│  │  ├─ 16                    347 ms        │ 503.9 ms      │ 358 ms        │ 362.1 ms      │ 100     │ 100
│  │  │                        460.9 Kitem/s │ 317.5 Kitem/s │ 446.8 Kitem/s │ 441.7 Kitem/s │         │
│  │  ╰─ 32                    736.1 ms      │ 969 ms        │ 767.9 ms      │ 785.2 ms      │ 100     │ 100
│  │                           434.6 Kitem/s │ 330.2 Kitem/s │ 416.7 Kitem/s │ 407.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.937 ms      │ 9.274 ms      │ 7.058 ms      │ 7.176 ms      │ 100     │ 100
│  │  │                        1.441 Mitem/s │ 1.078 Mitem/s │ 1.416 Mitem/s │ 1.393 Mitem/s │         │
│  │  ├─ 2                     7.243 ms      │ 15.09 ms      │ 12.82 ms      │ 11.62 ms      │ 100     │ 100
│  │  │                        2.761 Mitem/s │ 1.324 Mitem/s │ 1.559 Mitem/s │ 1.72 Mitem/s  │         │
│  │  ├─ 4                     8.049 ms      │ 20.48 ms      │ 12.95 ms      │ 12.24 ms      │ 100     │ 100
│  │  │                        4.969 Mitem/s │ 1.952 Mitem/s │ 3.087 Mitem/s │ 3.266 Mitem/s │         │
│  │  ├─ 8                     12.9 ms       │ 22.56 ms      │ 14.2 ms       │ 15.7 ms       │ 100     │ 100
│  │  │                        6.197 Mitem/s │ 3.545 Mitem/s │ 5.631 Mitem/s │ 5.093 Mitem/s │         │
│  │  ├─ 16                    19.91 ms      │ 26.28 ms      │ 22.12 ms      │ 22.25 ms      │ 100     │ 100
│  │  │                        8.035 Mitem/s │ 6.086 Mitem/s │ 7.23 Mitem/s  │ 7.189 Mitem/s │         │
│  │  ╰─ 32                    39.28 ms      │ 45.46 ms      │ 41.78 ms      │ 41.77 ms      │ 100     │ 100
│  │                           8.145 Mitem/s │ 7.037 Mitem/s │ 7.659 Mitem/s │ 7.659 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.961 ms      │ 3.91 ms       │ 2.029 ms      │ 2.278 ms      │ 100     │ 100
│  │  │                        5.099 Mitem/s │ 2.556 Mitem/s │ 4.927 Mitem/s │ 4.389 Mitem/s │         │
│  │  ├─ 2                     2.196 ms      │ 4.58 ms       │ 3.221 ms      │ 3.36 ms       │ 100     │ 100
│  │  │                        9.105 Mitem/s │ 4.366 Mitem/s │ 6.207 Mitem/s │ 5.952 Mitem/s │         │
│  │  ├─ 4                     2.98 ms       │ 5.463 ms      │ 3.822 ms      │ 3.701 ms      │ 100     │ 100
│  │  │                        13.41 Mitem/s │ 7.321 Mitem/s │ 10.46 Mitem/s │ 10.8 Mitem/s  │         │
│  │  ├─ 8                     4.528 ms      │ 7.519 ms      │ 4.837 ms      │ 5.323 ms      │ 100     │ 100
│  │  │                        17.66 Mitem/s │ 10.63 Mitem/s │ 16.53 Mitem/s │ 15.02 Mitem/s │         │
│  │  ├─ 16                    7.685 ms      │ 10.2 ms       │ 8.421 ms      │ 8.519 ms      │ 100     │ 100
│  │  │                        20.81 Mitem/s │ 15.67 Mitem/s │ 18.99 Mitem/s │ 18.78 Mitem/s │         │
│  │  ╰─ 32                    14.33 ms      │ 17.53 ms      │ 15.61 ms      │ 15.66 ms      │ 100     │ 100
│  │                           22.31 Mitem/s │ 18.25 Mitem/s │ 20.49 Mitem/s │ 20.42 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.121 ms      │ 14.27 ms      │ 9.484 ms      │ 9.791 ms      │ 100     │ 100
│     │                        1.096 Mitem/s │ 700.6 Kitem/s │ 1.054 Mitem/s │ 1.021 Mitem/s │         │
│     ├─ 2                     9.199 ms      │ 19.26 ms      │ 9.744 ms      │ 10.72 ms      │ 100     │ 100
│     │                        2.174 Mitem/s │ 1.038 Mitem/s │ 2.052 Mitem/s │ 1.864 Mitem/s │         │
│     ├─ 4                     9.495 ms      │ 23.33 ms      │ 14.26 ms      │ 14.37 ms      │ 100     │ 100
│     │                        4.212 Mitem/s │ 1.714 Mitem/s │ 2.804 Mitem/s │ 2.782 Mitem/s │         │
│     ├─ 8                     16.65 ms      │ 31.85 ms      │ 19 ms         │ 21.47 ms      │ 100     │ 100
│     │                        4.802 Mitem/s │ 2.511 Mitem/s │ 4.209 Mitem/s │ 3.725 Mitem/s │         │
│     ├─ 16                    26.63 ms      │ 32.54 ms      │ 28.78 ms      │ 28.95 ms      │ 100     │ 100
│     │                        6.007 Mitem/s │ 4.916 Mitem/s │ 5.558 Mitem/s │ 5.526 Mitem/s │         │
│     ╰─ 32                    50.38 ms      │ 112.8 ms      │ 53.98 ms      │ 57.98 ms      │ 100     │ 100
│                              6.35 Mitem/s  │ 2.836 Mitem/s │ 5.927 Mitem/s │ 5.518 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.671 ms      │ 6.143 ms      │ 3.763 ms      │ 4.102 ms      │ 100     │ 100
│  │  │                        2.724 Mitem/s │ 1.627 Mitem/s │ 2.656 Mitem/s │ 2.437 Mitem/s │         │
│  │  ├─ 2                     9.761 ms      │ 15.96 ms      │ 12.11 ms      │ 12.39 ms      │ 100     │ 100
│  │  │                        2.048 Mitem/s │ 1.252 Mitem/s │ 1.65 Mitem/s  │ 1.612 Mitem/s │         │
│  │  ├─ 4                     27.98 ms      │ 43.68 ms      │ 34.94 ms      │ 34.61 ms      │ 100     │ 100
│  │  │                        1.429 Mitem/s │ 915.6 Kitem/s │ 1.144 Mitem/s │ 1.155 Mitem/s │         │
│  │  ├─ 8                     145.1 ms      │ 181.4 ms      │ 158.8 ms      │ 159.8 ms      │ 100     │ 100
│  │  │                        551.1 Kitem/s │ 440.9 Kitem/s │ 503.5 Kitem/s │ 500.4 Kitem/s │         │
│  │  ├─ 16                    344.3 ms      │ 593.2 ms      │ 358.3 ms      │ 364.1 ms      │ 100     │ 100
│  │  │                        464.6 Kitem/s │ 269.7 Kitem/s │ 446.4 Kitem/s │ 439.4 Kitem/s │         │
│  │  ╰─ 32                    707.8 ms      │ 863 ms        │ 748.2 ms      │ 758.1 ms      │ 100     │ 100
│  │                           452 Kitem/s   │ 370.7 Kitem/s │ 427.6 Kitem/s │ 422.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     10.43 ms      │ 17.81 ms      │ 10.81 ms      │ 11.2 ms       │ 100     │ 100
│  │  │                        958.4 Kitem/s │ 561.2 Kitem/s │ 924.4 Kitem/s │ 892.4 Kitem/s │         │
│  │  ├─ 2                     10.61 ms      │ 21.42 ms      │ 14.15 ms      │ 13.95 ms      │ 100     │ 100
│  │  │                        1.883 Mitem/s │ 933.3 Kitem/s │ 1.412 Mitem/s │ 1.432 Mitem/s │         │
│  │  ├─ 4                     12.72 ms      │ 33.85 ms      │ 18.89 ms      │ 18.97 ms      │ 100     │ 100
│  │  │                        3.143 Mitem/s │ 1.181 Mitem/s │ 2.117 Mitem/s │ 2.108 Mitem/s │         │
│  │  ├─ 8                     19.04 ms      │ 36 ms         │ 21.77 ms      │ 24.96 ms      │ 100     │ 100
│  │  │                        4.2 Mitem/s   │ 2.221 Mitem/s │ 3.674 Mitem/s │ 3.203 Mitem/s │         │
│  │  ├─ 16                    30.03 ms      │ 38.4 ms       │ 33.3 ms       │ 33.39 ms      │ 100     │ 100
│  │  │                        5.327 Mitem/s │ 4.166 Mitem/s │ 4.804 Mitem/s │ 4.791 Mitem/s │         │
│  │  ╰─ 32                    58.48 ms      │ 71.47 ms      │ 61.8 ms       │ 62.25 ms      │ 100     │ 100
│  │                           5.471 Mitem/s │ 4.477 Mitem/s │ 5.177 Mitem/s │ 5.14 Mitem/s  │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.941 ms      │ 4.003 ms      │ 2.096 ms      │ 2.507 ms      │ 100     │ 100
│  │  │                        5.15 Mitem/s  │ 2.497 Mitem/s │ 4.77 Mitem/s  │ 3.988 Mitem/s │         │
│  │  ├─ 2                     2.173 ms      │ 4.958 ms      │ 4.076 ms      │ 3.906 ms      │ 100     │ 100
│  │  │                        9.199 Mitem/s │ 4.033 Mitem/s │ 4.906 Mitem/s │ 5.119 Mitem/s │         │
│  │  ├─ 4                     2.967 ms      │ 7.154 ms      │ 4.218 ms      │ 4.204 ms      │ 100     │ 100
│  │  │                        13.48 Mitem/s │ 5.59 Mitem/s  │ 9.481 Mitem/s │ 9.514 Mitem/s │         │
│  │  ├─ 8                     4.93 ms       │ 8.4 ms        │ 5.344 ms      │ 5.584 ms      │ 100     │ 100
│  │  │                        16.22 Mitem/s │ 9.523 Mitem/s │ 14.96 Mitem/s │ 14.32 Mitem/s │         │
│  │  ├─ 16                    8.292 ms      │ 22.7 ms       │ 9.277 ms      │ 10.16 ms      │ 100     │ 100
│  │  │                        19.29 Mitem/s │ 7.045 Mitem/s │ 17.24 Mitem/s │ 15.74 Mitem/s │         │
│  │  ╰─ 32                    16.4 ms       │ 19 ms         │ 17.47 ms      │ 17.54 ms      │ 100     │ 100
│  │                           19.5 Mitem/s  │ 16.83 Mitem/s │ 18.31 Mitem/s │ 18.24 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.06 ms      │ 14.78 ms      │ 10.4 ms       │ 10.97 ms      │ 100     │ 100
│     │                        993.6 Kitem/s │ 676.5 Kitem/s │ 961.2 Kitem/s │ 911.4 Kitem/s │         │
│     ├─ 2                     10.32 ms      │ 16.87 ms      │ 11.07 ms      │ 11.61 ms      │ 100     │ 100
│     │                        1.936 Mitem/s │ 1.185 Mitem/s │ 1.805 Mitem/s │ 1.721 Mitem/s │         │
│     ├─ 4                     10.58 ms      │ 33.65 ms      │ 17.93 ms      │ 18.73 ms      │ 100     │ 100
│     │                        3.777 Mitem/s │ 1.188 Mitem/s │ 2.229 Mitem/s │ 2.134 Mitem/s │         │
│     ├─ 8                     19.79 ms      │ 35.77 ms      │ 21.68 ms      │ 23.57 ms      │ 100     │ 100
│     │                        4.041 Mitem/s │ 2.236 Mitem/s │ 3.689 Mitem/s │ 3.393 Mitem/s │         │
│     ├─ 16                    30.46 ms      │ 57.64 ms      │ 34.02 ms      │ 34.43 ms      │ 100     │ 100
│     │                        5.252 Mitem/s │ 2.775 Mitem/s │ 4.702 Mitem/s │ 4.645 Mitem/s │         │
│     ╰─ 32                    58.06 ms      │ 141.9 ms      │ 63.03 ms      │ 66.39 ms      │ 100     │ 100
│                              5.51 Mitem/s  │ 2.254 Mitem/s │ 5.076 Mitem/s │ 4.819 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.661 ms      │ 8.287 ms      │ 3.825 ms      │ 4.401 ms      │ 100     │ 100
│  │  │                        2.731 Mitem/s │ 1.206 Mitem/s │ 2.614 Mitem/s │ 2.271 Mitem/s │         │
│  │  ├─ 2                     9.149 ms      │ 14.91 ms      │ 10.79 ms      │ 11.23 ms      │ 100     │ 100
│  │  │                        2.185 Mitem/s │ 1.34 Mitem/s  │ 1.853 Mitem/s │ 1.779 Mitem/s │         │
│  │  ├─ 4                     26.59 ms      │ 46.2 ms       │ 34.63 ms      │ 34.92 ms      │ 100     │ 100
│  │  │                        1.504 Mitem/s │ 865.7 Kitem/s │ 1.155 Mitem/s │ 1.145 Mitem/s │         │
│  │  ├─ 8                     145.4 ms      │ 264.3 ms      │ 159.5 ms      │ 165.1 ms      │ 100     │ 100
│  │  │                        549.9 Kitem/s │ 302.5 Kitem/s │ 501.3 Kitem/s │ 484.2 Kitem/s │         │
│  │  ├─ 16                    338.1 ms      │ 523.9 ms      │ 350.1 ms      │ 355.8 ms      │ 100     │ 100
│  │  │                        473.2 Kitem/s │ 305.3 Kitem/s │ 456.9 Kitem/s │ 449.6 Kitem/s │         │
│  │  ╰─ 32                    711.6 ms      │ 1.069 s       │ 745.2 ms      │ 761.5 ms      │ 100     │ 100
│  │                           449.6 Kitem/s │ 299.3 Kitem/s │ 429.3 Kitem/s │ 420.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.531 ms      │ 10.59 ms      │ 7.734 ms      │ 8.046 ms      │ 100     │ 100
│  │  │                        1.327 Mitem/s │ 943.6 Kitem/s │ 1.292 Mitem/s │ 1.242 Mitem/s │         │
│  │  ├─ 2                     7.838 ms      │ 15.79 ms      │ 9.063 ms      │ 9.627 ms      │ 100     │ 100
│  │  │                        2.551 Mitem/s │ 1.266 Mitem/s │ 2.206 Mitem/s │ 2.077 Mitem/s │         │
│  │  ├─ 4                     8.799 ms      │ 21.54 ms      │ 13.04 ms      │ 12.72 ms      │ 100     │ 100
│  │  │                        4.545 Mitem/s │ 1.856 Mitem/s │ 3.065 Mitem/s │ 3.142 Mitem/s │         │
│  │  ├─ 8                     13.73 ms      │ 23.98 ms      │ 15.37 ms      │ 16.59 ms      │ 100     │ 100
│  │  │                        5.824 Mitem/s │ 3.335 Mitem/s │ 5.204 Mitem/s │ 4.819 Mitem/s │         │
│  │  ├─ 16                    21.77 ms      │ 31.25 ms      │ 23.5 ms       │ 23.85 ms      │ 100     │ 100
│  │  │                        7.347 Mitem/s │ 5.118 Mitem/s │ 6.806 Mitem/s │ 6.707 Mitem/s │         │
│  │  ╰─ 32                    42.24 ms      │ 94.21 ms      │ 45.42 ms      │ 49.03 ms      │ 100     │ 100
│  │                           7.575 Mitem/s │ 3.396 Mitem/s │ 7.044 Mitem/s │ 6.526 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.968 ms      │ 4.123 ms      │ 2.104 ms      │ 2.417 ms      │ 100     │ 100
│  │  │                        5.08 Mitem/s  │ 2.425 Mitem/s │ 4.751 Mitem/s │ 4.136 Mitem/s │         │
│  │  ├─ 2                     2.197 ms      │ 4.855 ms      │ 3.681 ms      │ 3.631 ms      │ 100     │ 100
│  │  │                        9.1 Mitem/s   │ 4.118 Mitem/s │ 5.432 Mitem/s │ 5.506 Mitem/s │         │
│  │  ├─ 4                     2.945 ms      │ 8.429 ms      │ 4.127 ms      │ 4.065 ms      │ 100     │ 100
│  │  │                        13.57 Mitem/s │ 4.745 Mitem/s │ 9.692 Mitem/s │ 9.839 Mitem/s │         │
│  │  ├─ 8                     4.864 ms      │ 7.931 ms      │ 5.219 ms      │ 5.393 ms      │ 100     │ 100
│  │  │                        16.44 Mitem/s │ 10.08 Mitem/s │ 15.32 Mitem/s │ 14.83 Mitem/s │         │
│  │  ├─ 16                    7.95 ms       │ 10.88 ms      │ 8.808 ms      │ 8.896 ms      │ 100     │ 100
│  │  │                        20.12 Mitem/s │ 14.7 Mitem/s  │ 18.16 Mitem/s │ 17.98 Mitem/s │         │
│  │  ╰─ 32                    15.55 ms      │ 18.76 ms      │ 16.61 ms      │ 16.69 ms      │ 100     │ 100
│  │                           20.56 Mitem/s │ 17.05 Mitem/s │ 19.25 Mitem/s │ 19.17 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.42 ms      │ 20.4 ms       │ 10.78 ms      │ 11.51 ms      │ 100     │ 100
│     │                        958.9 Kitem/s │ 490 Kitem/s   │ 926.9 Kitem/s │ 868.6 Kitem/s │         │
│     ├─ 2                     10.65 ms      │ 21.65 ms      │ 14.54 ms      │ 14.08 ms      │ 100     │ 100
│     │                        1.876 Mitem/s │ 923.4 Kitem/s │ 1.374 Mitem/s │ 1.419 Mitem/s │         │
│     ├─ 4                     11.03 ms      │ 29.91 ms      │ 17.88 ms      │ 17.46 ms      │ 100     │ 100
│     │                        3.623 Mitem/s │ 1.337 Mitem/s │ 2.236 Mitem/s │ 2.29 Mitem/s  │         │
│     ├─ 8                     19.24 ms      │ 45.99 ms      │ 21.83 ms      │ 25.2 ms       │ 100     │ 100
│     │                        4.156 Mitem/s │ 1.739 Mitem/s │ 3.663 Mitem/s │ 3.174 Mitem/s │         │
│     ├─ 16                    31.02 ms      │ 71.98 ms      │ 33.25 ms      │ 36.17 ms      │ 100     │ 100
│     │                        5.157 Mitem/s │ 2.222 Mitem/s │ 4.811 Mitem/s │ 4.422 Mitem/s │         │
│     ╰─ 32                    58.77 ms      │ 68.92 ms      │ 61.71 ms      │ 62.03 ms      │ 100     │ 100
│                              5.444 Mitem/s │ 4.642 Mitem/s │ 5.185 Mitem/s │ 5.158 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.624 ms      │ 8.382 ms      │ 4.736 ms      │ 5.163 ms      │ 100     │ 100
│  │  │                        2.162 Mitem/s │ 1.193 Mitem/s │ 2.111 Mitem/s │ 1.936 Mitem/s │         │
│  │  ├─ 2                     11.69 ms      │ 20.78 ms      │ 16.06 ms      │ 16.28 ms      │ 100     │ 100
│  │  │                        1.71 Mitem/s  │ 962.2 Kitem/s │ 1.244 Mitem/s │ 1.228 Mitem/s │         │
│  │  ├─ 4                     34.98 ms      │ 55.6 ms       │ 42.19 ms      │ 42.37 ms      │ 100     │ 100
│  │  │                        1.143 Mitem/s │ 719.3 Kitem/s │ 947.9 Kitem/s │ 943.9 Kitem/s │         │
│  │  ├─ 8                     157.3 ms      │ 267.3 ms      │ 174.7 ms      │ 177.5 ms      │ 100     │ 100
│  │  │                        508.2 Kitem/s │ 299.2 Kitem/s │ 457.9 Kitem/s │ 450.5 Kitem/s │         │
│  │  ├─ 16                    350.2 ms      │ 645.8 ms      │ 359.7 ms      │ 369.6 ms      │ 100     │ 100
│  │  │                        456.8 Kitem/s │ 247.7 Kitem/s │ 444.7 Kitem/s │ 432.7 Kitem/s │         │
│  │  ╰─ 32                    753.7 ms      │ 1.062 s       │ 819.5 ms      │ 833.5 ms      │ 100     │ 100
│  │                           424.5 Kitem/s │ 301.1 Kitem/s │ 390.4 Kitem/s │ 383.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.994 ms      │ 12.39 ms      │ 7.173 ms      │ 7.506 ms      │ 100     │ 100
│  │  │                        1.429 Mitem/s │ 806.6 Kitem/s │ 1.394 Mitem/s │ 1.332 Mitem/s │         │
│  │  ├─ 2                     7.979 ms      │ 14.77 ms      │ 11.44 ms      │ 11.64 ms      │ 100     │ 100
│  │  │                        2.506 Mitem/s │ 1.354 Mitem/s │ 1.748 Mitem/s │ 1.717 Mitem/s │         │
│  │  ├─ 4                     8.168 ms      │ 19.82 ms      │ 12.85 ms      │ 12.42 ms      │ 100     │ 100
│  │  │                        4.896 Mitem/s │ 2.017 Mitem/s │ 3.112 Mitem/s │ 3.22 Mitem/s  │         │
│  │  ├─ 8                     12.79 ms      │ 23.44 ms      │ 14.65 ms      │ 16.44 ms      │ 100     │ 100
│  │  │                        6.252 Mitem/s │ 3.412 Mitem/s │ 5.459 Mitem/s │ 4.865 Mitem/s │         │
│  │  ├─ 16                    20.32 ms      │ 26.63 ms      │ 22.22 ms      │ 22.53 ms      │ 100     │ 100
│  │  │                        7.871 Mitem/s │ 6.007 Mitem/s │ 7.2 Mitem/s   │ 7.098 Mitem/s │         │
│  │  ╰─ 32                    39.14 ms      │ 51.08 ms      │ 41.51 ms      │ 41.73 ms      │ 100     │ 100
│  │                           8.174 Mitem/s │ 6.264 Mitem/s │ 7.707 Mitem/s │ 7.666 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.962 ms      │ 3.442 ms      │ 2.042 ms      │ 2.241 ms      │ 100     │ 100
│  │  │                        5.096 Mitem/s │ 2.904 Mitem/s │ 4.895 Mitem/s │ 4.46 Mitem/s  │         │
│  │  ├─ 2                     2.195 ms      │ 5.05 ms       │ 3.787 ms      │ 3.739 ms      │ 100     │ 100
│  │  │                        9.108 Mitem/s │ 3.96 Mitem/s  │ 5.28 Mitem/s  │ 5.348 Mitem/s │         │
│  │  ├─ 4                     2.947 ms      │ 6.837 ms      │ 4.06 ms       │ 4.193 ms      │ 100     │ 100
│  │  │                        13.56 Mitem/s │ 5.849 Mitem/s │ 9.85 Mitem/s  │ 9.537 Mitem/s │         │
│  │  ├─ 8                     5.11 ms       │ 11.99 ms      │ 5.662 ms      │ 6.826 ms      │ 100     │ 100
│  │  │                        15.65 Mitem/s │ 6.666 Mitem/s │ 14.12 Mitem/s │ 11.71 Mitem/s │         │
│  │  ├─ 16                    8.686 ms      │ 10.78 ms      │ 9.327 ms      │ 9.509 ms      │ 100     │ 100
│  │  │                        18.41 Mitem/s │ 14.83 Mitem/s │ 17.15 Mitem/s │ 16.82 Mitem/s │         │
│  │  ╰─ 32                    17.1 ms       │ 19.16 ms      │ 18.1 ms       │ 18.13 ms      │ 100     │ 100
│  │                           18.7 Mitem/s  │ 16.69 Mitem/s │ 17.67 Mitem/s │ 17.64 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.406 ms      │ 14.47 ms      │ 9.857 ms      │ 10.33 ms      │ 100     │ 100
│     │                        1.063 Mitem/s │ 691 Kitem/s   │ 1.014 Mitem/s │ 967.4 Kitem/s │         │
│     ├─ 2                     9.654 ms      │ 18.38 ms      │ 12.29 ms      │ 12.36 ms      │ 100     │ 100
│     │                        2.071 Mitem/s │ 1.088 Mitem/s │ 1.627 Mitem/s │ 1.617 Mitem/s │         │
│     ├─ 4                     10.11 ms      │ 28.46 ms      │ 16.7 ms       │ 16.19 ms      │ 100     │ 100
│     │                        3.953 Mitem/s │ 1.405 Mitem/s │ 2.394 Mitem/s │ 2.469 Mitem/s │         │
│     ├─ 8                     17.53 ms      │ 33.46 ms      │ 20.41 ms      │ 23.18 ms      │ 100     │ 100
│     │                        4.562 Mitem/s │ 2.39 Mitem/s  │ 3.919 Mitem/s │ 3.45 Mitem/s  │         │
│     ├─ 16                    27.51 ms      │ 39.4 ms       │ 29.44 ms      │ 29.7 ms       │ 100     │ 100
│     │                        5.814 Mitem/s │ 4.06 Mitem/s  │ 5.434 Mitem/s │ 5.386 Mitem/s │         │
│     ╰─ 32                    52.11 ms      │ 65.29 ms      │ 55.03 ms      │ 55.45 ms      │ 100     │ 100
│                              6.14 Mitem/s  │ 4.901 Mitem/s │ 5.814 Mitem/s │ 5.77 Mitem/s  │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.567 ms      │ 6.742 ms      │ 4.675 ms      │ 4.928 ms      │ 100     │ 100
│  │  │                        2.189 Mitem/s │ 1.483 Mitem/s │ 2.138 Mitem/s │ 2.029 Mitem/s │         │
│  │  ├─ 2                     11.73 ms      │ 20.49 ms      │ 14.66 ms      │ 14.87 ms      │ 100     │ 100
│  │  │                        1.704 Mitem/s │ 975.8 Kitem/s │ 1.363 Mitem/s │ 1.344 Mitem/s │         │
│  │  ├─ 4                     35.27 ms      │ 61.25 ms      │ 44.77 ms      │ 45.27 ms      │ 100     │ 100
│  │  │                        1.134 Mitem/s │ 652.9 Kitem/s │ 893.3 Kitem/s │ 883.5 Kitem/s │         │
│  │  ├─ 8                     161.3 ms      │ 311.8 ms      │ 175.2 ms      │ 180.2 ms      │ 100     │ 100
│  │  │                        495.7 Kitem/s │ 256.5 Kitem/s │ 456.3 Kitem/s │ 443.7 Kitem/s │         │
│  │  ├─ 16                    356.8 ms      │ 753.4 ms      │ 367.1 ms      │ 388.6 ms      │ 100     │ 100
│  │  │                        448.3 Kitem/s │ 212.3 Kitem/s │ 435.7 Kitem/s │ 411.6 Kitem/s │         │
│  │  ╰─ 32                    752.3 ms      │ 944.5 ms      │ 828.2 ms      │ 832.5 ms      │ 100     │ 100
│  │                           425.3 Kitem/s │ 338.7 Kitem/s │ 386.3 Kitem/s │ 384.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.62 ms       │ 11.19 ms      │ 7.909 ms      │ 8.199 ms      │ 100     │ 100
│  │  │                        1.312 Mitem/s │ 893.2 Kitem/s │ 1.264 Mitem/s │ 1.219 Mitem/s │         │
│  │  ├─ 2                     7.971 ms      │ 14.31 ms      │ 9.628 ms      │ 10.12 ms      │ 100     │ 100
│  │  │                        2.508 Mitem/s │ 1.396 Mitem/s │ 2.077 Mitem/s │ 1.975 Mitem/s │         │
│  │  ├─ 4                     8.942 ms      │ 22.54 ms      │ 13.59 ms      │ 13.37 ms      │ 100     │ 100
│  │  │                        4.472 Mitem/s │ 1.774 Mitem/s │ 2.943 Mitem/s │ 2.99 Mitem/s  │         │
│  │  ├─ 8                     13.84 ms      │ 25.27 ms      │ 15.97 ms      │ 17.64 ms      │ 100     │ 100
│  │  │                        5.776 Mitem/s │ 3.164 Mitem/s │ 5.006 Mitem/s │ 4.532 Mitem/s │         │
│  │  ├─ 16                    21.77 ms      │ 54.39 ms      │ 24.65 ms      │ 27.88 ms      │ 100     │ 100
│  │  │                        7.347 Mitem/s │ 2.941 Mitem/s │ 6.49 Mitem/s  │ 5.738 Mitem/s │         │
│  │  ╰─ 32                    41.62 ms      │ 64.84 ms      │ 45.44 ms      │ 46.09 ms      │ 100     │ 100
│  │                           7.688 Mitem/s │ 4.934 Mitem/s │ 7.041 Mitem/s │ 6.942 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.937 ms      │ 3.597 ms      │ 2.075 ms      │ 2.185 ms      │ 100     │ 100
│  │  │                        5.161 Mitem/s │ 2.78 Mitem/s  │ 4.818 Mitem/s │ 4.576 Mitem/s │         │
│  │  ├─ 2                     2.227 ms      │ 4.399 ms      │ 2.967 ms      │ 3.085 ms      │ 100     │ 100
│  │  │                        8.977 Mitem/s │ 4.546 Mitem/s │ 6.739 Mitem/s │ 6.481 Mitem/s │         │
│  │  ├─ 4                     2.614 ms      │ 5.633 ms      │ 4.014 ms      │ 4.044 ms      │ 100     │ 100
│  │  │                        15.29 Mitem/s │ 7.1 Mitem/s   │ 9.964 Mitem/s │ 9.89 Mitem/s  │         │
│  │  ├─ 8                     4.55 ms       │ 7.522 ms      │ 4.946 ms      │ 5.152 ms      │ 100     │ 100
│  │  │                        17.57 Mitem/s │ 10.63 Mitem/s │ 16.17 Mitem/s │ 15.52 Mitem/s │         │
│  │  ├─ 16                    7.8 ms        │ 31.92 ms      │ 10.19 ms      │ 12.28 ms      │ 100     │ 100
│  │  │                        20.51 Mitem/s │ 5.011 Mitem/s │ 15.68 Mitem/s │ 13.02 Mitem/s │         │
│  │  ╰─ 32                    14.71 ms      │ 17.65 ms      │ 15.7 ms       │ 15.78 ms      │ 100     │ 100
│  │                           21.74 Mitem/s │ 18.12 Mitem/s │ 20.37 Mitem/s │ 20.27 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.22 ms      │ 14.07 ms      │ 10.78 ms      │ 11.07 ms      │ 100     │ 100
│     │                        978.1 Kitem/s │ 710.5 Kitem/s │ 926.8 Kitem/s │ 902.8 Kitem/s │         │
│     ├─ 2                     10.52 ms      │ 20.22 ms      │ 11.59 ms      │ 12.4 ms       │ 100     │ 100
│     │                        1.899 Mitem/s │ 988.8 Kitem/s │ 1.724 Mitem/s │ 1.611 Mitem/s │         │
│     ├─ 4                     10.82 ms      │ 32.41 ms      │ 16.86 ms      │ 17.59 ms      │ 100     │ 100
│     │                        3.693 Mitem/s │ 1.233 Mitem/s │ 2.371 Mitem/s │ 2.273 Mitem/s │         │
│     ├─ 8                     19.64 ms      │ 34.87 ms      │ 21.64 ms      │ 23.78 ms      │ 100     │ 100
│     │                        4.073 Mitem/s │ 2.293 Mitem/s │ 3.696 Mitem/s │ 3.363 Mitem/s │         │
│     ├─ 16                    30.61 ms      │ 43.67 ms      │ 33.53 ms      │ 33.85 ms      │ 100     │ 100
│     │                        5.226 Mitem/s │ 3.663 Mitem/s │ 4.771 Mitem/s │ 4.725 Mitem/s │         │
│     ╰─ 32                    58.3 ms       │ 77.88 ms      │ 60.78 ms      │ 61.99 ms      │ 100     │ 100
│                              5.488 Mitem/s │ 4.108 Mitem/s │ 5.264 Mitem/s │ 5.161 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.553 ms      │ 6.208 ms      │ 4.637 ms      │ 4.724 ms      │ 100     │ 100
│  │  │                        2.196 Mitem/s │ 1.61 Mitem/s  │ 2.156 Mitem/s │ 2.116 Mitem/s │         │
│  │  ├─ 2                     10.81 ms      │ 20.75 ms      │ 16.49 ms      │ 16.11 ms      │ 100     │ 100
│  │  │                        1.848 Mitem/s │ 963.4 Kitem/s │ 1.212 Mitem/s │ 1.241 Mitem/s │         │
│  │  ├─ 4                     36.51 ms      │ 60.69 ms      │ 46.43 ms      │ 46.35 ms      │ 100     │ 100
│  │  │                        1.095 Mitem/s │ 659 Kitem/s   │ 861.5 Kitem/s │ 862.8 Kitem/s │         │
│  │  ├─ 8                     97.45 ms      │ 272.3 ms      │ 175.4 ms      │ 178.2 ms      │ 100     │ 100
│  │  │                        820.9 Kitem/s │ 293.7 Kitem/s │ 456 Kitem/s   │ 448.9 Kitem/s │         │
│  │  ├─ 16                    357.7 ms      │ 692.8 ms      │ 363.5 ms      │ 375.3 ms      │ 100     │ 100
│  │  │                        447.2 Kitem/s │ 230.9 Kitem/s │ 440.1 Kitem/s │ 426.2 Kitem/s │         │
│  │  ╰─ 32                    748.9 ms      │ 984.8 ms      │ 859 ms        │ 852.3 ms      │ 100     │ 100
│  │                           427.2 Kitem/s │ 324.9 Kitem/s │ 372.5 Kitem/s │ 375.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.608 ms      │ 10.72 ms      │ 7.772 ms      │ 8.033 ms      │ 100     │ 100
│  │  │                        1.314 Mitem/s │ 932 Kitem/s   │ 1.286 Mitem/s │ 1.244 Mitem/s │         │
│  │  ├─ 2                     7.917 ms      │ 16.47 ms      │ 10 ms         │ 10.41 ms      │ 100     │ 100
│  │  │                        2.526 Mitem/s │ 1.213 Mitem/s │ 1.999 Mitem/s │ 1.919 Mitem/s │         │
│  │  ├─ 4                     8.779 ms      │ 25.74 ms      │ 12.7 ms       │ 12.78 ms      │ 100     │ 100
│  │  │                        4.556 Mitem/s │ 1.553 Mitem/s │ 3.148 Mitem/s │ 3.127 Mitem/s │         │
│  │  ├─ 8                     13.83 ms      │ 25.25 ms      │ 15.31 ms      │ 16.9 ms       │ 100     │ 100
│  │  │                        5.782 Mitem/s │ 3.167 Mitem/s │ 5.223 Mitem/s │ 4.731 Mitem/s │         │
│  │  ├─ 16                    21.93 ms      │ 35.73 ms      │ 23.97 ms      │ 24.26 ms      │ 100     │ 100
│  │  │                        7.295 Mitem/s │ 4.477 Mitem/s │ 6.673 Mitem/s │ 6.594 Mitem/s │         │
│  │  ╰─ 32                    41.39 ms      │ 65.93 ms      │ 44.4 ms       │ 45.16 ms      │ 100     │ 100
│  │                           7.729 Mitem/s │ 4.853 Mitem/s │ 7.205 Mitem/s │ 7.085 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.952 ms      │ 3.683 ms      │ 2.726 ms      │ 2.68 ms       │ 100     │ 100
│  │  │                        5.122 Mitem/s │ 2.714 Mitem/s │ 3.667 Mitem/s │ 3.731 Mitem/s │         │
│  │  ├─ 2                     2.23 ms       │ 5.061 ms      │ 3.458 ms      │ 3.562 ms      │ 100     │ 100
│  │  │                        8.967 Mitem/s │ 3.951 Mitem/s │ 5.782 Mitem/s │ 5.613 Mitem/s │         │
│  │  ├─ 4                     2.43 ms       │ 7.005 ms      │ 4 ms          │ 3.95 ms       │ 100     │ 100
│  │  │                        16.45 Mitem/s │ 5.709 Mitem/s │ 9.999 Mitem/s │ 10.12 Mitem/s │         │
│  │  ├─ 8                     4.458 ms      │ 7.825 ms      │ 5.062 ms      │ 5.229 ms      │ 100     │ 100
│  │  │                        17.94 Mitem/s │ 10.22 Mitem/s │ 15.8 Mitem/s  │ 15.29 Mitem/s │         │
│  │  ├─ 16                    7.464 ms      │ 17.92 ms      │ 8.354 ms      │ 8.624 ms      │ 100     │ 100
│  │  │                        21.43 Mitem/s │ 8.926 Mitem/s │ 19.15 Mitem/s │ 18.55 Mitem/s │         │
│  │  ╰─ 32                    14.88 ms      │ 21 ms         │ 16.08 ms      │ 16.57 ms      │ 100     │ 100
│  │                           21.49 Mitem/s │ 15.23 Mitem/s │ 19.88 Mitem/s │ 19.3 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.14 ms      │ 16.73 ms      │ 10.68 ms      │ 11.22 ms      │ 100     │ 100
│     │                        986 Kitem/s   │ 597.3 Kitem/s │ 936.2 Kitem/s │ 891.1 Kitem/s │         │
│     ├─ 2                     10.24 ms      │ 21.53 ms      │ 11.79 ms      │ 12.73 ms      │ 100     │ 100
│     │                        1.952 Mitem/s │ 928.7 Kitem/s │ 1.695 Mitem/s │ 1.569 Mitem/s │         │
│     ├─ 4                     10.71 ms      │ 30.49 ms      │ 17.14 ms      │ 17.42 ms      │ 100     │ 100
│     │                        3.732 Mitem/s │ 1.311 Mitem/s │ 2.332 Mitem/s │ 2.295 Mitem/s │         │
│     ├─ 8                     19.62 ms      │ 39.15 ms      │ 21.88 ms      │ 24.76 ms      │ 100     │ 100
│     │                        4.076 Mitem/s │ 2.043 Mitem/s │ 3.655 Mitem/s │ 3.23 Mitem/s  │         │
│     ├─ 16                    30.52 ms      │ 50.37 ms      │ 32.57 ms      │ 33.21 ms      │ 100     │ 100
│     │                        5.242 Mitem/s │ 3.176 Mitem/s │ 4.911 Mitem/s │ 4.817 Mitem/s │         │
│     ╰─ 32                    58.23 ms      │ 134.7 ms      │ 61.95 ms      │ 66.1 ms       │ 100     │ 100
│                              5.495 Mitem/s │ 2.374 Mitem/s │ 5.165 Mitem/s │ 4.84 Mitem/s  │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.592 ms      │ 6.715 ms      │ 4.677 ms      │ 4.829 ms      │ 100     │ 100
│  │  │                        2.177 Mitem/s │ 1.488 Mitem/s │ 2.138 Mitem/s │ 2.07 Mitem/s  │         │
│  │  ├─ 2                     11.03 ms      │ 20.7 ms       │ 13.98 ms      │ 14.45 ms      │ 100     │ 100
│  │  │                        1.811 Mitem/s │ 966.1 Kitem/s │ 1.429 Mitem/s │ 1.383 Mitem/s │         │
│  │  ├─ 4                     37.03 ms      │ 60.6 ms       │ 46.59 ms      │ 46.47 ms      │ 100     │ 100
│  │  │                        1.08 Mitem/s  │ 660 Kitem/s   │ 858.4 Kitem/s │ 860.6 Kitem/s │         │
│  │  ├─ 8                     159.2 ms      │ 216.1 ms      │ 174.7 ms      │ 176.9 ms      │ 100     │ 100
│  │  │                        502.3 Kitem/s │ 370 Kitem/s   │ 457.7 Kitem/s │ 452.1 Kitem/s │         │
│  │  ├─ 16                    354.1 ms      │ 402.7 ms      │ 361 ms        │ 366.3 ms      │ 100     │ 100
│  │  │                        451.7 Kitem/s │ 397.2 Kitem/s │ 443.1 Kitem/s │ 436.7 Kitem/s │         │
│  │  ╰─ 32                    758.7 ms      │ 1.069 s       │ 880.5 ms      │ 872.3 ms      │ 100     │ 100
│  │                           421.7 Kitem/s │ 299.2 Kitem/s │ 363.3 Kitem/s │ 366.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.583 ms      │ 13.61 ms      │ 7.718 ms      │ 8.452 ms      │ 100     │ 100
│  │  │                        1.318 Mitem/s │ 734.3 Kitem/s │ 1.295 Mitem/s │ 1.183 Mitem/s │         │
│  │  ├─ 2                     7.99 ms       │ 16.13 ms      │ 12.09 ms      │ 12.03 ms      │ 100     │ 100
│  │  │                        2.502 Mitem/s │ 1.239 Mitem/s │ 1.653 Mitem/s │ 1.661 Mitem/s │         │
│  │  ├─ 4                     8.688 ms      │ 22.35 ms      │ 14.29 ms      │ 14.13 ms      │ 100     │ 100
│  │  │                        4.603 Mitem/s │ 1.789 Mitem/s │ 2.798 Mitem/s │ 2.829 Mitem/s │         │
│  │  ├─ 8                     13.55 ms      │ 25.93 ms      │ 15.66 ms      │ 17.36 ms      │ 100     │ 100
│  │  │                        5.901 Mitem/s │ 3.085 Mitem/s │ 5.105 Mitem/s │ 4.607 Mitem/s │         │
│  │  ├─ 16                    21.98 ms      │ 50.38 ms      │ 24.85 ms      │ 28.08 ms      │ 100     │ 100
│  │  │                        7.278 Mitem/s │ 3.175 Mitem/s │ 6.436 Mitem/s │ 5.697 Mitem/s │         │
│  │  ╰─ 32                    41.48 ms      │ 64.89 ms      │ 44.06 ms      │ 44.75 ms      │ 100     │ 100
│  │                           7.714 Mitem/s │ 4.93 Mitem/s  │ 7.261 Mitem/s │ 7.15 Mitem/s  │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.944 ms      │ 3.559 ms      │ 1.998 ms      │ 2.162 ms      │ 100     │ 100
│  │  │                        5.141 Mitem/s │ 2.809 Mitem/s │ 5.003 Mitem/s │ 4.623 Mitem/s │         │
│  │  ├─ 2                     2.225 ms      │ 4.737 ms      │ 3.622 ms      │ 3.716 ms      │ 100     │ 100
│  │  │                        8.985 Mitem/s │ 4.221 Mitem/s │ 5.521 Mitem/s │ 5.382 Mitem/s │         │
│  │  ├─ 4                     2.977 ms      │ 6.786 ms      │ 4.126 ms      │ 4.04 ms       │ 100     │ 100
│  │  │                        13.43 Mitem/s │ 5.893 Mitem/s │ 9.692 Mitem/s │ 9.898 Mitem/s │         │
│  │  ├─ 8                     4.925 ms      │ 9.331 ms      │ 5.261 ms      │ 5.463 ms      │ 100     │ 100
│  │  │                        16.24 Mitem/s │ 8.572 Mitem/s │ 15.2 Mitem/s  │ 14.64 Mitem/s │         │
│  │  ├─ 16                    8.048 ms      │ 20.87 ms      │ 8.966 ms      │ 9.219 ms      │ 100     │ 100
│  │  │                        19.87 Mitem/s │ 7.662 Mitem/s │ 17.84 Mitem/s │ 17.35 Mitem/s │         │
│  │  ╰─ 32                    15.6 ms       │ 38.46 ms      │ 17.21 ms      │ 20.66 ms      │ 100     │ 100
│  │                           20.5 Mitem/s  │ 8.319 Mitem/s │ 18.58 Mitem/s │ 15.48 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.05 ms      │ 17.25 ms      │ 10.63 ms      │ 12.1 ms       │ 100     │ 100
│     │                        994.9 Kitem/s │ 579.6 Kitem/s │ 940.2 Kitem/s │ 825.9 Kitem/s │         │
│     ├─ 2                     10.3 ms       │ 20.92 ms      │ 13.15 ms      │ 13.35 ms      │ 100     │ 100
│     │                        1.941 Mitem/s │ 955.6 Kitem/s │ 1.519 Mitem/s │ 1.497 Mitem/s │         │
│     ├─ 4                     10.62 ms      │ 32.02 ms      │ 16.26 ms      │ 16.7 ms       │ 100     │ 100
│     │                        3.764 Mitem/s │ 1.249 Mitem/s │ 2.459 Mitem/s │ 2.394 Mitem/s │         │
│     ├─ 8                     19.38 ms      │ 35.06 ms      │ 21.99 ms      │ 24.39 ms      │ 100     │ 100
│     │                        4.126 Mitem/s │ 2.281 Mitem/s │ 3.637 Mitem/s │ 3.279 Mitem/s │         │
│     ├─ 16                    30.46 ms      │ 52.51 ms      │ 32.6 ms       │ 33.06 ms      │ 100     │ 100
│     │                        5.252 Mitem/s │ 3.046 Mitem/s │ 4.907 Mitem/s │ 4.839 Mitem/s │         │
│     ╰─ 32                    57.29 ms      │ 83.02 ms      │ 60.32 ms      │ 60.88 ms      │ 100     │ 100
│                              5.584 Mitem/s │ 3.854 Mitem/s │ 5.304 Mitem/s │ 5.255 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     5.281 ms      │ 7.401 ms      │ 5.367 ms      │ 5.416 ms      │ 100     │ 100
│  │  │                        1.893 Mitem/s │ 1.351 Mitem/s │ 1.862 Mitem/s │ 1.846 Mitem/s │         │
│  │  ├─ 2                     12.53 ms      │ 23.02 ms      │ 14.87 ms      │ 15.91 ms      │ 100     │ 100
│  │  │                        1.595 Mitem/s │ 868.7 Kitem/s │ 1.344 Mitem/s │ 1.256 Mitem/s │         │
│  │  ├─ 4                     38.9 ms       │ 62.26 ms      │ 47.57 ms      │ 48.33 ms      │ 100     │ 100
│  │  │                        1.028 Mitem/s │ 642.4 Kitem/s │ 840.8 Kitem/s │ 827.4 Kitem/s │         │
│  │  ├─ 8                     163.2 ms      │ 221.8 ms      │ 186.2 ms      │ 187.3 ms      │ 100     │ 100
│  │  │                        490 Kitem/s   │ 360.5 Kitem/s │ 429.4 Kitem/s │ 426.9 Kitem/s │         │
│  │  ├─ 16                    364.6 ms      │ 617.1 ms      │ 376.8 ms      │ 384.6 ms      │ 100     │ 100
│  │  │                        438.7 Kitem/s │ 259.2 Kitem/s │ 424.5 Kitem/s │ 415.9 Kitem/s │         │
│  │  ╰─ 32                    768.4 ms      │ 989.6 ms      │ 848 ms        │ 853.5 ms      │ 100     │ 100
│  │                           416.3 Kitem/s │ 323.3 Kitem/s │ 377.3 Kitem/s │ 374.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.686 ms      │ 11.44 ms      │ 7.891 ms      │ 8.138 ms      │ 100     │ 100
│  │  │                        1.3 Mitem/s   │ 873.9 Kitem/s │ 1.267 Mitem/s │ 1.228 Mitem/s │         │
│  │  ├─ 2                     7.952 ms      │ 15.01 ms      │ 9.969 ms      │ 10.33 ms      │ 100     │ 100
│  │  │                        2.514 Mitem/s │ 1.331 Mitem/s │ 2.006 Mitem/s │ 1.934 Mitem/s │         │
│  │  ├─ 4                     8.773 ms      │ 21.86 ms      │ 13.46 ms      │ 13.01 ms      │ 100     │ 100
│  │  │                        4.559 Mitem/s │ 1.829 Mitem/s │ 2.971 Mitem/s │ 3.074 Mitem/s │         │
│  │  ├─ 8                     14.05 ms      │ 24.25 ms      │ 15.75 ms      │ 16.73 ms      │ 100     │ 100
│  │  │                        5.69 Mitem/s  │ 3.298 Mitem/s │ 5.078 Mitem/s │ 4.78 Mitem/s  │         │
│  │  ├─ 16                    22.05 ms      │ 29.73 ms      │ 24.75 ms      │ 24.84 ms      │ 100     │ 100
│  │  │                        7.256 Mitem/s │ 5.381 Mitem/s │ 6.463 Mitem/s │ 6.439 Mitem/s │         │
│  │  ╰─ 32                    42.31 ms      │ 65.91 ms      │ 46.4 ms       │ 46.92 ms      │ 100     │ 100
│  │                           7.561 Mitem/s │ 4.854 Mitem/s │ 6.895 Mitem/s │ 6.819 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.976 ms      │ 4.044 ms      │ 2.05 ms       │ 2.255 ms      │ 100     │ 100
│  │  │                        5.059 Mitem/s │ 2.472 Mitem/s │ 4.876 Mitem/s │ 4.432 Mitem/s │         │
│  │  ├─ 2                     2.232 ms      │ 5.217 ms      │ 3.041 ms      │ 3.158 ms      │ 100     │ 100
│  │  │                        8.958 Mitem/s │ 3.833 Mitem/s │ 6.576 Mitem/s │ 6.332 Mitem/s │         │
│  │  ├─ 4                     2.966 ms      │ 6.017 ms      │ 4.132 ms      │ 3.944 ms      │ 100     │ 100
│  │  │                        13.48 Mitem/s │ 6.647 Mitem/s │ 9.678 Mitem/s │ 10.13 Mitem/s │         │
│  │  ├─ 8                     4.85 ms       │ 7.089 ms      │ 5.191 ms      │ 5.294 ms      │ 100     │ 100
│  │  │                        16.49 Mitem/s │ 11.28 Mitem/s │ 15.4 Mitem/s  │ 15.1 Mitem/s  │         │
│  │  ├─ 16                    7.971 ms      │ 12.34 ms      │ 8.955 ms      │ 9.133 ms      │ 100     │ 100
│  │  │                        20.07 Mitem/s │ 12.96 Mitem/s │ 17.86 Mitem/s │ 17.51 Mitem/s │         │
│  │  ╰─ 32                    15.58 ms      │ 21.38 ms      │ 17.24 ms      │ 17.74 ms      │ 100     │ 100
│  │                           20.53 Mitem/s │ 14.96 Mitem/s │ 18.55 Mitem/s │ 18.03 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.763 ms      │ 15.07 ms      │ 10.63 ms      │ 11.22 ms      │ 100     │ 100
│     │                        1.024 Mitem/s │ 663.3 Kitem/s │ 940.5 Kitem/s │ 890.7 Kitem/s │         │
│     ├─ 2                     10.18 ms      │ 19.01 ms      │ 11.4 ms       │ 11.79 ms      │ 100     │ 100
│     │                        1.964 Mitem/s │ 1.051 Mitem/s │ 1.753 Mitem/s │ 1.694 Mitem/s │         │
│     ├─ 4                     11.52 ms      │ 31.95 ms      │ 17.46 ms      │ 18.29 ms      │ 100     │ 100
│     │                        3.472 Mitem/s │ 1.251 Mitem/s │ 2.29 Mitem/s  │ 2.186 Mitem/s │         │
│     ├─ 8                     19.19 ms      │ 34.05 ms      │ 21.67 ms      │ 23.94 ms      │ 100     │ 100
│     │                        4.167 Mitem/s │ 2.349 Mitem/s │ 3.69 Mitem/s  │ 3.34 Mitem/s  │         │
│     ├─ 16                    30.12 ms      │ 43.75 ms      │ 32.97 ms      │ 33.21 ms      │ 100     │ 100
│     │                        5.31 Mitem/s  │ 3.656 Mitem/s │ 4.852 Mitem/s │ 4.816 Mitem/s │         │
│     ╰─ 32                    57.1 ms       │ 74.42 ms      │ 60.64 ms      │ 61.51 ms      │ 100     │ 100
│                              5.603 Mitem/s │ 4.299 Mitem/s │ 5.276 Mitem/s │ 5.202 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 3                     9.168 ms      │ 15.75 ms      │ 11.08 ms      │ 11.45 ms      │ 100     │ 100
│  │  ├─ 4                     10.6 ms       │ 16.05 ms      │ 13.06 ms      │ 13.27 ms      │ 100     │ 100
│  │  ├─ 5                     10.78 ms      │ 24.03 ms      │ 14.98 ms      │ 14.58 ms      │ 100     │ 100
│  │  ╰─ 6                     11.51 ms      │ 25.55 ms      │ 15.33 ms      │ 15.29 ms      │ 100     │ 100
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 3                     21.96 ms      │ 30.97 ms      │ 27.25 ms      │ 27.16 ms      │ 100     │ 100
│  │  ├─ 4                     24.59 ms      │ 35.87 ms      │ 30.67 ms      │ 30.73 ms      │ 100     │ 100
│  │  ├─ 5                     26.8 ms       │ 39.01 ms      │ 32.59 ms      │ 32.35 ms      │ 100     │ 100
│  │  ╰─ 6                     27.23 ms      │ 47.37 ms      │ 33.93 ms      │ 34.52 ms      │ 100     │ 100
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 3                     ^[z13.12 ms      │ 22.83 ms      │ 14.97 ms      │ 16.66 ms      │ 100     │ 100
│     ├─ 4                    13.36 ms      │ 30.68 ms      │ 19.28 ms      │ 18.67 ms      │ 100     │ 100
│     ├─ 5                      13.39 ms      │ 33.83 ms      │ 19.71 ms      │ 20.48 ms      │ 100     │ 100
│     ╰─ 6                     14.19 ms      │ 33.89 ms      │ 20.61 ms      │ 21.66 ms      │ 100     │ 100
├─ 14_prefix_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     23.43 ms      │ 25.9 ms       │ 24.11 ms      │ 24.24 ms      │ 20      │ 20
│     │                        4.267 Kitem/s │ 3.859 Kitem/s │ 4.147 Kitem/s │ 4.124 Kitem/s │         │
│     ├─ 2                     24.55 ms      │ 43 ms         │ 39.42 ms      │ 36.02 ms      │ 20      │ 20
│     │                        8.143 Kitem/s │ 4.65 Kitem/s  │ 5.072 Kitem/s │ 5.551 Kitem/s │         │
│     ├─ 4                     36.37 ms      │ 44.78 ms      │ 37.96 ms      │ 39.51 ms      │ 20      │ 20
│     │                        10.99 Kitem/s │ 8.931 Kitem/s │ 10.53 Kitem/s │ 10.12 Kitem/s │         │
│     ├─ 8                     45.46 ms      │ 67.43 ms      │ 50.36 ms      │ 51.03 ms      │ 20      │ 20
│     │                        17.59 Kitem/s │ 11.86 Kitem/s │ 15.88 Kitem/s │ 15.67 Kitem/s │         │
│     ├─ 16                    82.82 ms      │ 174.5 ms      │ 87.9 ms       │ 93.68 ms      │ 20      │ 20
│     │                        19.31 Kitem/s │ 9.164 Kitem/s │ 18.2 Kitem/s  │ 17.07 Kitem/s │         │
│     ╰─ 32                    162 ms        │ 194 ms        │ 168 ms        │ 168.9 ms      │ 20      │ 20
│                              19.74 Kitem/s │ 16.49 Kitem/s │ 19.03 Kitem/s │ 18.94 Kitem/s │         │
├─ 15_full_scan_aggregate                    │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     15.1 ms       │ 21.16 ms      │ 15.47 ms      │ 16.02 ms      │ 100     │ 100
│  │  │                        6.619 Kitem/s │ 4.725 Kitem/s │ 6.463 Kitem/s │ 6.241 Kitem/s │         │
│  │  ├─ 2                     32.32 ms      │ 60.95 ms      │ 47.67 ms      │ 47.15 ms      │ 100     │ 100
│  │  │                        6.187 Kitem/s │ 3.281 Kitem/s │ 4.194 Kitem/s │ 4.241 Kitem/s │         │
│  │  ├─ 4                     68.26 ms      │ 119.9 ms      │ 103 ms        │ 99.18 ms      │ 100     │ 100
│  │  │                        5.859 Kitem/s │ 3.334 Kitem/s │ 3.882 Kitem/s │ 4.032 Kitem/s │         │
│  │  ├─ 8                     146 ms        │ 232.2 ms      │ 216.2 ms      │ 212.4 ms      │ 100     │ 100
│  │  │                        5.477 Kitem/s │ 3.444 Kitem/s │ 3.699 Kitem/s │ 3.765 Kitem/s │         │
│  │  ├─ 16                    287.8 ms      │ 462.9 ms      │ 442.1 ms      │ 437.9 ms      │ 100     │ 100
│  │  │                        5.559 Kitem/s │ 3.455 Kitem/s │ 3.618 Kitem/s │ 3.653 Kitem/s │         │
│  │  ╰─ 32                    810.4 ms      │ 935.2 ms      │ 905.5 ms      │ 897.5 ms      │ 100     │ 100
│  │                           3.948 Kitem/s │ 3.421 Kitem/s │ 3.533 Kitem/s │ 3.565 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     31.44 ms      │ 37.42 ms      │ 33.49 ms      │ 33.86 ms      │ 100     │ 100
│  │  │                        3.18 Kitem/s  │ 2.671 Kitem/s │ 2.985 Kitem/s │ 2.952 Kitem/s │         │
│  │  ├─ 2                     32.23 ms      │ 53.95 ms      │ 34.55 ms      │ 35.74 ms      │ 100     │ 100
│  │  │                        6.205 Kitem/s │ 3.706 Kitem/s │ 5.787 Kitem/s │ 5.594 Kitem/s │         │
│  │  ├─ 4                     33.65 ms      │ 68.68 ms      │ 46.21 ms      │ 45.12 ms      │ 100     │ 100
│  │  │                        11.88 Kitem/s │ 5.823 Kitem/s │ 8.655 Kitem/s │ 8.863 Kitem/s │         │
│  │  ├─ 8                     58.88 ms      │ 81.7 ms       │ 61.06 ms      │ 63.06 ms      │ 100     │ 100
│  │  │                        13.58 Kitem/s │ 9.791 Kitem/s │ 13.1 Kitem/s  │ 12.68 Kitem/s │         │
│  │  ├─ 16                    91.54 ms      │ 197.7 ms      │ 95.69 ms      │ 98.74 ms      │ 100     │ 100
│  │  │                        17.47 Kitem/s │ 8.09 Kitem/s  │ 16.71 Kitem/s │ 16.2 Kitem/s  │         │
│  │  ╰─ 32                    173.3 ms      │ 211.9 ms      │ 181 ms        │ 181.5 ms      │ 100     │ 100
│  │                           18.46 Kitem/s │ 15.09 Kitem/s │ 17.67 Kitem/s │ 17.62 Kitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     6.853 ms      │ 11.16 ms      │ 7.397 ms      │ 7.848 ms      │ 100     │ 100
│  │  │                        14.59 Kitem/s │ 8.952 Kitem/s │ 13.51 Kitem/s │ 12.74 Kitem/s │         │
│  │  ├─ 2                     7.2 ms        │ 15.51 ms      │ 12.85 ms      │ 12.05 ms      │ 100     │ 100
│  │  │                        27.77 Kitem/s │ 12.89 Kitem/s │ 15.55 Kitem/s │ 16.58 Kitem/s │         │
│  │  ├─ 4                     7.434 ms      │ 19.39 ms      │ 12.06 ms      │ 11.43 ms      │ 100     │ 100
│  │  │                        53.8 Kitem/s  │ 20.62 Kitem/s │ 33.16 Kitem/s │ 34.97 Kitem/s │         │
│  │  ├─ 8                     12.16 ms      │ 21.81 ms      │ 13.2 ms       │ 14.39 ms      │ 100     │ 100
│  │  │                        65.77 Kitem/s │ 36.66 Kitem/s │ 60.59 Kitem/s │ 55.56 Kitem/s │         │
│  │  ├─ 16                    18.81 ms      │ 24.27 ms      │ 20.95 ms      │ 21.11 ms      │ 100     │ 100
│  │  │                        85.05 Kitem/s │ 65.91 Kitem/s │ 76.35 Kitem/s │ 75.77 Kitem/s │         │
│  │  ╰─ 32                    36.03 ms      │ 98.09 ms      │ 38.24 ms      │ 39.89 ms      │ 100     │ 100
│  │                           88.8 Kitem/s  │ 32.62 Kitem/s │ 83.66 Kitem/s │ 80.21 Kitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     47.4 ms       │ 66.96 ms      │ 49.62 ms      │ 50.02 ms      │ 100     │ 100
│     │                        2.109 Kitem/s │ 1.493 Kitem/s │ 2.015 Kitem/s │ 1.999 Kitem/s │         │
│     ├─ 2                     48.82 ms      │ 72.06 ms      │ 50.99 ms      │ 51.29 ms      │ 100     │ 100
│     │                        4.095 Kitem/s │ 2.775 Kitem/s │ 3.921 Kitem/s │ 3.899 Kitem/s │         │
│     ├─ 4                     50.75 ms      │ 110.7 ms      │ 67.58 ms      │ 69.8 ms       │ 100     │ 100
│     │                        7.88 Kitem/s  │ 3.612 Kitem/s │ 5.918 Kitem/s │ 5.729 Kitem/s │         │
│     ├─ 8                     86.61 ms      │ 117.6 ms      │ 95.06 ms      │ 97.9 ms       │ 100     │ 100
│     │                        9.235 Kitem/s │ 6.8 Kitem/s   │ 8.415 Kitem/s │ 8.17 Kitem/s  │         │
│     ├─ 16                    140.5 ms      │ 248.6 ms      │ 146.2 ms      │ 150.5 ms      │ 100     │ 100
│     │                        11.38 Kitem/s │ 6.433 Kitem/s │ 10.94 Kitem/s │ 10.62 Kitem/s │         │
│     ╰─ 32                    264 ms        │ 418 ms        │ 274.1 ms      │ 276.8 ms      │ 100     │ 100
│                              12.11 Kitem/s │ 7.653 Kitem/s │ 11.67 Kitem/s │ 11.56 Kitem/s │         │
├─ 16_insert_heavy                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.894 ms      │ 7.563 ms      │ 5.893 ms      │ 5.972 ms      │ 100     │ 100
│  │  │                        2.043 Mitem/s │ 1.322 Mitem/s │ 1.696 Mitem/s │ 1.674 Mitem/s │         │
│  │  ├─ 2                     6.92 ms       │ 15.47 ms      │ 10.67 ms      │ 10.52 ms      │ 100     │ 100
│  │  │                        2.89 Mitem/s  │ 1.292 Mitem/s │ 1.873 Mitem/s │ 1.9 Mitem/s   │         │
│  │  ├─ 4                     10.41 ms      │ 24.4 ms       │ 14.64 ms      │ 15 ms         │ 100     │ 100
│  │  │                        3.839 Mitem/s │ 1.638 Mitem/s │ 2.731 Mitem/s │ 2.665 Mitem/s │         │
│  │  ├─ 8                     12.6 ms       │ 28.26 ms      │ 16.72 ms      │ 17.39 ms      │ 100     │ 100
│  │  │                        6.346 Mitem/s │ 2.83 Mitem/s  │ 4.784 Mitem/s │ 4.599 Mitem/s │         │
│  │  ├─ 16                    20.81 ms      │ 37.88 ms      │ 28.52 ms      │ 28.88 ms      │ 100     │ 100
│  │  │                        7.685 Mitem/s │ 4.222 Mitem/s │ 5.608 Mitem/s │ 5.538 Mitem/s │         │
│  │  ╰─ 32                    34.25 ms      │ 54.75 ms      │ 42.36 ms      │ 43.1 ms       │ 100     │ 100
│  │                           9.34 Mitem/s  │ 5.843 Mitem/s │ 7.553 Mitem/s │ 7.423 Mitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     1.499 ms      │ 3.26 ms       │ 2.038 ms      │ 2.16 ms       │ 100     │ 100
│  │  │                        6.671 Mitem/s │ 3.067 Mitem/s │ 4.905 Mitem/s │ 4.627 Mitem/s │         │
│  │  ├─ 2                     2.072 ms      │ 3.353 ms      │ 2.624 ms      │ 2.569 ms      │ 100     │ 100
│  │  │                        9.648 Mitem/s │ 5.963 Mitem/s │ 7.619 Mitem/s │ 7.783 Mitem/s │         │
│  │  ├─ 4                     2.931 ms      │ 4.407 ms      │ 3.556 ms      │ 3.599 ms      │ 100     │ 100
│  │  │                        13.64 Mitem/s │ 9.074 Mitem/s │ 11.24 Mitem/s │ 11.11 Mitem/s │         │
│  │  ├─ 8                     4.879 ms      │ 8.554 ms      │ 6.076 ms      │ 6.225 ms      │ 100     │ 100
│  │  │                        16.39 Mitem/s │ 9.352 Mitem/s │ 13.16 Mitem/s │ 12.85 Mitem/s │         │
│  │  ├─ 16                    8.491 ms      │ 15.34 ms      │ 11.14 ms      │ 11.18 ms      │ 100     │ 100
│  │  │                        18.84 Mitem/s │ 10.42 Mitem/s │ 14.35 Mitem/s │ 14.31 Mitem/s │         │
│  │  ╰─ 32                    17.59 ms      │ 28 ms         │ 19.5 ms       │ 20.06 ms      │ 100     │ 100
│  │                           18.18 Mitem/s │ 11.42 Mitem/s │ 16.41 Mitem/s │ 15.94 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.163 ms      │ 3.803 ms      │ 1.795 ms      │ 1.837 ms      │ 100     │ 100
│  │  │                        8.597 Mitem/s │ 2.629 Mitem/s │ 5.568 Mitem/s │ 5.443 Mitem/s │         │
│  │  ├─ 2                     3.728 ms      │ 6.793 ms      │ 5.111 ms      │ 4.993 ms      │ 100     │ 100
│  │  │                        5.363 Mitem/s │ 2.944 Mitem/s │ 3.913 Mitem/s │ 4.005 Mitem/s │         │
│  │  ├─ 4                     7.741 ms      │ 15.49 ms      │ 11.72 ms      │ 11.67 ms      │ 100     │ 100
│  │  │                        5.167 Mitem/s │ 2.581 Mitem/s │ 3.411 Mitem/s │ 3.425 Mitem/s │         │
│  │  ├─ 8                     19.26 ms      │ 34.84 ms      │ 25.61 ms      │ 25.63 ms      │ 100     │ 100
│  │  │                        4.153 Mitem/s │ 2.295 Mitem/s │ 3.122 Mitem/s │ 3.12 Mitem/s  │         │
│  │  ├─ 16                    55.92 ms      │ 68.69 ms      │ 61.55 ms      │ 61.69 ms      │ 100     │ 100
│  │  │                        2.86 Mitem/s  │ 2.329 Mitem/s │ 2.599 Mitem/s │ 2.593 Mitem/s │         │
│  │  ╰─ 32                    121.9 ms      │ 141.9 ms      │ 132.5 ms      │ 132.7 ms      │ 100     │ 100
│  │                           2.623 Mitem/s │ 2.254 Mitem/s │ 2.414 Mitem/s │ 2.41 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     2.311 ms      │ 4.181 ms      │ 3.463 ms      │ 3.338 ms      │ 100     │ 100
│     │                        4.326 Mitem/s │ 2.391 Mitem/s │ 2.887 Mitem/s │ 2.995 Mitem/s │         │
│     ├─ 2                     2.553 ms      │ 5.718 ms      │ 3.884 ms      │ 3.973 ms      │ 100     │ 100
│     │                        7.832 Mitem/s │ 3.497 Mitem/s │ 5.149 Mitem/s │ 5.032 Mitem/s │         │
│     ├─ 4                     3.622 ms      │ 7.01 ms       │ 5.03 ms       │ 4.958 ms      │ 100     │ 100
│     │                        11.04 Mitem/s │ 5.705 Mitem/s │ 7.951 Mitem/s │ 8.067 Mitem/s │         │
│     ├─ 8                     5.753 ms      │ 10.98 ms      │ 7.608 ms      │ 7.669 ms      │ 100     │ 100
│     │                        13.9 Mitem/s  │ 7.28 Mitem/s  │ 10.51 Mitem/s │ 10.43 Mitem/s │         │
│     ├─ 16                    7.678 ms      │ 15.3 ms       │ 9.708 ms      │ 9.897 ms      │ 100     │ 100
│     │                        20.83 Mitem/s │ 10.45 Mitem/s │ 16.48 Mitem/s │ 16.16 Mitem/s │         │
│     ╰─ 32                    14.87 ms      │ 32.76 ms      │ 16.74 ms      │ 17.31 ms      │ 100     │ 100
│                              21.51 Mitem/s │ 9.765 Mitem/s │ 19.1 Mitem/s  │ 18.47 Mitem/s │         │
├─ 17_hot_spot                               │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     791.5 µs      │ 2.056 ms      │ 1.326 ms      │ 1.342 ms      │ 100     │ 100
│  │  │                        12.63 Mitem/s │ 4.862 Mitem/s │ 7.54 Mitem/s  │ 7.446 Mitem/s │         │
│  │  ├─ 2                     1.595 ms      │ 4.794 ms      │ 3.887 ms      │ 3.844 ms      │ 100     │ 100
│  │  │                        12.53 Mitem/s │ 4.171 Mitem/s │ 5.145 Mitem/s │ 5.201 Mitem/s │         │
│  │  ├─ 4                     5.692 ms      │ 11.7 ms       │ 10.24 ms      │ 10.11 ms      │ 100     │ 100
│  │  │                        7.026 Mitem/s │ 3.417 Mitem/s │ 3.906 Mitem/s │ 3.954 Mitem/s │         │
│  │  ├─ 8                     10.31 ms      │ 29.09 ms      │ 24.88 ms      │ 24.28 ms      │ 100     │ 100
│  │  │                        7.753 Mitem/s │ 2.749 Mitem/s │ 3.215 Mitem/s │ 3.294 Mitem/s │         │
│  │  ├─ 16                    21.88 ms      │ 84.9 ms       │ 76.02 ms      │ 73.08 ms      │ 100     │ 100
│  │  │                        7.309 Mitem/s │ 1.884 Mitem/s │ 2.104 Mitem/s │ 2.189 Mitem/s │         │
│  │  ╰─ 32                    102.8 ms      │ 220.2 ms      │ 193 ms        │ 189.1 ms      │ 100     │ 100
│  │                           3.112 Mitem/s │ 1.452 Mitem/s │ 1.657 Mitem/s │ 1.691 Mitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     1.045 ms      │ 2.378 ms      │ 1.735 ms      │ 1.715 ms      │ 100     │ 100
│  │  │                        9.565 Mitem/s │ 4.204 Mitem/s │ 5.76 Mitem/s  │ 5.827 Mitem/s │         │
│  │  ├─ 2                     2.639 ms      │ 4.818 ms      │ 4.233 ms      │ 4.112 ms      │ 100     │ 100
│  │  │                        7.578 Mitem/s │ 4.15 Mitem/s  │ 4.724 Mitem/s │ 4.862 Mitem/s │         │
│  │  ├─ 4                     6.826 ms      │ 9.43 ms       │ 8.31 ms       │ 8.28 ms       │ 100     │ 100
│  │  │                        5.859 Mitem/s │ 4.241 Mitem/s │ 4.813 Mitem/s │ 4.83 Mitem/s  │         │
│  │  ├─ 8                     13.12 ms      │ 18.24 ms      │ 16.03 ms      │ 15.97 ms      │ 100     │ 100
│  │  │                        6.095 Mitem/s │ 4.383 Mitem/s │ 4.989 Mitem/s │ 5.007 Mitem/s │         │
│  │  ├─ 16                    30.44 ms      │ 55.3 ms       │ 33.48 ms      │ 34.64 ms      │ 100     │ 100
│  │  │                        5.255 Mitem/s │ 2.893 Mitem/s │ 4.778 Mitem/s │ 4.617 Mitem/s │         │
│  │  ╰─ 32                    60.56 ms      │ 149.7 ms      │ 80.22 ms      │ 82.36 ms      │ 100     │ 100
│  │                           5.283 Mitem/s │ 2.136 Mitem/s │ 3.988 Mitem/s │ 3.885 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     462.1 µs      │ 1.366 ms      │ 775.7 µs      │ 788.7 µs      │ 100     │ 100
│  │  │                        21.63 Mitem/s │ 7.32 Mitem/s  │ 12.89 Mitem/s │ 12.67 Mitem/s │         │
│  │  ├─ 2                     1.672 ms      │ 4.173 ms      │ 3.218 ms      │ 3.056 ms      │ 100     │ 100
│  │  │                        11.96 Mitem/s │ 4.792 Mitem/s │ 6.213 Mitem/s │ 6.544 Mitem/s │         │
│  │  ├─ 4                     4.315 ms      │ 8.405 ms      │ 7.259 ms      │ 7.141 ms      │ 100     │ 100
│  │  │                        9.269 Mitem/s │ 4.758 Mitem/s │ 5.509 Mitem/s │ 5.6 Mitem/s   │         │
│  │  ├─ 8                     13.45 ms      │ 18.35 ms      │ 16.65 ms      │ 16.67 ms      │ 100     │ 100
│  │  │                        5.943 Mitem/s │ 4.357 Mitem/s │ 4.804 Mitem/s │ 4.796 Mitem/s │         │
│  │  ├─ 16                    25.47 ms      │ 38.07 ms      │ 36.8 ms       │ 36.69 ms      │ 100     │ 100
│  │  │                        6.281 Mitem/s │ 4.202 Mitem/s │ 4.347 Mitem/s │ 4.359 Mitem/s │         │
│  │  ╰─ 32                    72.25 ms      │ 79.77 ms      │ 77.86 ms      │ 77.72 ms      │ 100     │ 100
│  │                           4.428 Mitem/s │ 4.011 Mitem/s │ 4.109 Mitem/s │ 4.116 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     1.333 ms      │ 2.939 ms      │ 2.189 ms      │ 2.127 ms      │ 100     │ 100
│     │                        7.497 Mitem/s │ 3.401 Mitem/s │ 4.568 Mitem/s │ 4.699 Mitem/s │         │
│     ├─ 2                     2.345 ms      │ 3.955 ms      │ 3.337 ms      │ 3.27 ms       │ 100     │ 100
│     │                        8.527 Mitem/s │ 5.055 Mitem/s │ 5.992 Mitem/s │ 6.114 Mitem/s │         │
│     ├─ 4                     3.196 ms      │ 4.855 ms      │ 4.159 ms      │ 4.153 ms      │ 100     │ 100
│     │                        12.51 Mitem/s │ 8.238 Mitem/s │ 9.617 Mitem/s │ 9.631 Mitem/s │         │
│     ├─ 8                     4.11 ms       │ 7.186 ms      │ 4.726 ms      │ 4.776 ms      │ 100     │ 100
│     │                        19.46 Mitem/s │ 11.13 Mitem/s │ 16.92 Mitem/s │ 16.74 Mitem/s │         │
│     ├─ 16                    5.335 ms      │ 11.85 ms      │ 7.054 ms      │ 7.059 ms      │ 100     │ 100
│     │                        29.98 Mitem/s │ 13.49 Mitem/s │ 22.68 Mitem/s │ 22.66 Mitem/s │         │
│     ╰─ 32                    10.31 ms      │ 14.6 ms       │ 11.74 ms      │ 11.91 ms      │ 100     │ 100
│                              31.03 Mitem/s │ 21.91 Mitem/s │ 27.24 Mitem/s │ 26.85 Mitem/s │         │
╰─ 18_split_inducing_scan                    │               │               │               │         │
   ├─ indexset                               │               │               │               │         │
   │  ├─ 3                     22.18 ms      │ 43.7 ms       │ 31.97 ms      │ 31.45 ms      │ 100     │ 100
   │  ├─ 4                     32.24 ms      │ 59.37 ms      │ 41.97 ms      │ 42.71 ms      │ 100     │ 100
   │  ├─ 5                     42.08 ms      │ 72.34 ms      │ 52.28 ms      │ 52.74 ms      │ 100     │ 100
   │  ╰─ 6                     58.42 ms      │ 92.04 ms      │ 71.43 ms      │ 71.75 ms      │ 100     │ 100
   ├─ masstree24                             │               │               │               │         │
   │  ├─ 3                     8.554 ms      │ 17.38 ms      │ 9.915 ms      │ 10.59 ms      │ 100     │ 100
   │  ├─ 4                     9.061 ms      │ 17.91 ms      │ 15.35 ms      │ 14.59 ms      │ 100     │ 100
   │  ├─ 5                     10.13 ms      │ 27.67 ms      │ 15.43 ms      │ 15.22 ms      │ 100     │ 100
   │  ╰─ 6                     10.97 ms      │ 28.7 ms       │ 16.8 ms       │ 17.5 ms       │ 100     │ 100
   ├─ std_btreemap                           │               │               │               │         │
   │  ├─ 3                     12.29 ms      │ 19.78 ms      │ 16.16 ms      │ 16.13 ms      │ 100     │ 100
   │  ├─ 4                     16.23 ms      │ 26.79 ms      │ 19.88 ms      │ 20.07 ms      │ 100     │ 100
   │  ├─ 5                     13.77 ms      │ 36.27 ms      │ 21.66 ms      │ 22.12 ms      │ 100     │ 100
   │  ╰─ 6                     17.75 ms      │ 38.78 ms      │ 22.44 ms      │ 23.16 ms      │ 100     │ 100
   ╰─ tree_index                             │               │               │               │         │
      ├─ 3                     10.26 ms      │ 22.14 ms      │ 12.07 ms      │ 13.28 ms      │ 100     │ 100
      ├─ 4                     10.88 ms      │ 21.68 ms      │ 19.74 ms      │ 17.92 ms      │ 100     │ 100
      ├─ 5                     10.81 ms      │ 34.22 ms      │ 18.72 ms      │ 17.75 ms      │ 100     │ 100
      ╰─ 6                     12.85 ms      │ 35.01 ms      │ 20.54 ms      │ 21.11 ms      │ 100     │ 100
```
