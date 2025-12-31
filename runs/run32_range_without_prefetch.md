```bash
Timer precision: 40 ns
range_masstree24               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.02 ms       │ 13.07 ms      │ 7.27 ms       │ 7.913 ms      │ 100     │ 100
│     │                        1.424 Mitem/s │ 764.5 Kitem/s │ 1.375 Mitem/s │ 1.263 Mitem/s │         │
│     ├─ 2                     7.518 ms      │ 15.91 ms      │ 10.79 ms      │ 10.81 ms      │ 100     │ 100
│     │                        2.66 Mitem/s  │ 1.256 Mitem/s │ 1.852 Mitem/s │ 1.85 Mitem/s  │         │
│     ├─ 3                     8.486 ms      │ 15.69 ms      │ 11.99 ms      │ 12.83 ms      │ 100     │ 100
│     │                        3.534 Mitem/s │ 1.911 Mitem/s │ 2.501 Mitem/s │ 2.336 Mitem/s │         │
│     ├─ 4                     8.85 ms       │ 25.28 ms      │ 13.34 ms      │ 13.28 ms      │ 100     │ 100
│     │                        4.519 Mitem/s │ 1.581 Mitem/s │ 2.997 Mitem/s │ 3.011 Mitem/s │         │
│     ├─ 5                     8.975 ms      │ 23.28 ms      │ 13.53 ms      │ 14.12 ms      │ 100     │ 100
│     │                        5.57 Mitem/s  │ 2.147 Mitem/s │ 3.693 Mitem/s │ 3.54 Mitem/s  │         │
│     ╰─ 6                     9.146 ms      │ 23.74 ms      │ 13.87 ms      │ 15.21 ms      │ 100     │ 100
│                              6.559 Mitem/s │ 2.526 Mitem/s │ 4.322 Mitem/s │ 3.943 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     6.935 ms      │ 13.77 ms      │ 7.17 ms       │ 7.481 ms      │ 100     │ 100
│     │                        1.441 Mitem/s │ 726 Kitem/s   │ 1.394 Mitem/s │ 1.336 Mitem/s │         │
│     ├─ 2                     7.291 ms      │ 15.79 ms      │ 9.982 ms      │ 10.18 ms      │ 100     │ 100
│     │                        2.742 Mitem/s │ 1.266 Mitem/s │ 2.003 Mitem/s │ 1.964 Mitem/s │         │
│     ├─ 3                     7.698 ms      │ 16.56 ms      │ 13.31 ms      │ 12.83 ms      │ 100     │ 100
│     │                        3.896 Mitem/s │ 1.81 Mitem/s  │ 2.252 Mitem/s │ 2.336 Mitem/s │         │
│     ├─ 4                     8.483 ms      │ 19.02 ms      │ 11.79 ms      │ 11.93 ms      │ 100     │ 100
│     │                        4.714 Mitem/s │ 2.102 Mitem/s │ 3.39 Mitem/s  │ 3.351 Mitem/s │         │
│     ├─ 5                     9.004 ms      │ 21.77 ms      │ 13.04 ms      │ 13.08 ms      │ 100     │ 100
│     │                        5.552 Mitem/s │ 2.296 Mitem/s │ 3.833 Mitem/s │ 3.82 Mitem/s  │         │
│     ╰─ 6                     9.264 ms      │ 22.12 ms      │ 13.3 ms       │ 14.23 ms      │ 100     │ 100
│                              6.476 Mitem/s │ 2.711 Mitem/s │ 4.509 Mitem/s │ 4.215 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.057 ms      │ 11.31 ms      │ 7.801 ms      │ 7.971 ms      │ 100     │ 100
│     │                        1.416 Mitem/s │ 884.1 Kitem/s │ 1.281 Mitem/s │ 1.254 Mitem/s │         │
│     ├─ 2                     7.575 ms      │ 19.4 ms       │ 8.848 ms      │ 9.846 ms      │ 100     │ 100
│     │                        2.639 Mitem/s │ 1.03 Mitem/s  │ 2.26 Mitem/s  │ 2.031 Mitem/s │         │
│     ├─ 3                     7.792 ms      │ 21.16 ms      │ 9.936 ms      │ 10.97 ms      │ 100     │ 100
│     │                        3.849 Mitem/s │ 1.417 Mitem/s │ 3.019 Mitem/s │ 2.732 Mitem/s │         │
│     ├─ 4                     8.623 ms      │ 21.64 ms      │ 12.84 ms      │ 12.69 ms      │ 100     │ 100
│     │                        4.638 Mitem/s │ 1.848 Mitem/s │ 3.114 Mitem/s │ 3.15 Mitem/s  │         │
│     ├─ 5                     8.874 ms      │ 23.1 ms       │ 13.17 ms      │ 12.88 ms      │ 100     │ 100
│     │                        5.634 Mitem/s │ 2.164 Mitem/s │ 3.796 Mitem/s │ 3.879 Mitem/s │         │
│     ╰─ 6                     9.31 ms       │ 23.41 ms      │ 13.98 ms      │ 14.76 ms      │ 100     │ 100
│                              6.444 Mitem/s │ 2.562 Mitem/s │ 4.29 Mitem/s  │ 4.063 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.067 ms      │ 12.64 ms      │ 7.284 ms      │ 7.614 ms      │ 100     │ 100
│     │                        1.414 Mitem/s │ 790.5 Kitem/s │ 1.372 Mitem/s │ 1.313 Mitem/s │         │
│     ├─ 2                     7.485 ms      │ 15.2 ms       │ 8.6 ms        │ 9.386 ms      │ 100     │ 100
│     │                        2.671 Mitem/s │ 1.314 Mitem/s │ 2.325 Mitem/s │ 2.13 Mitem/s  │         │
│     ├─ 3                     8.377 ms      │ 21.5 ms       │ 11.39 ms      │ 11.44 ms      │ 100     │ 100
│     │                        3.581 Mitem/s │ 1.394 Mitem/s │ 2.631 Mitem/s │ 2.622 Mitem/s │         │
│     ├─ 4                     8.596 ms      │ 19.02 ms      │ 12.01 ms      │ 11.84 ms      │ 100     │ 100
│     │                        4.652 Mitem/s │ 2.102 Mitem/s │ 3.329 Mitem/s │ 3.376 Mitem/s │         │
│     ├─ 5                     8.974 ms      │ 22.79 ms      │ 13.26 ms      │ 12.94 ms      │ 100     │ 100
│     │                        5.571 Mitem/s │ 2.193 Mitem/s │ 3.77 Mitem/s  │ 3.861 Mitem/s │         │
│     ╰─ 6                     9.296 ms      │ 22.39 ms      │ 13.84 ms      │ 14.46 ms      │ 100     │ 100
│                              6.453 Mitem/s │ 2.678 Mitem/s │ 4.333 Mitem/s │ 4.147 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     6.497 ms      │ 9.015 ms      │ 6.77 ms       │ 7.047 ms      │ 100     │ 100
│     │                        1.539 Mitem/s │ 1.109 Mitem/s │ 1.476 Mitem/s │ 1.419 Mitem/s │         │
│     ├─ 2                     6.823 ms      │ 14.5 ms       │ 8.154 ms      │ 9.113 ms      │ 100     │ 100
│     │                        2.93 Mitem/s  │ 1.378 Mitem/s │ 2.452 Mitem/s │ 2.194 Mitem/s │         │
│     ├─ 3                     7.58 ms       │ 16.18 ms      │ 10.61 ms      │ 10.86 ms      │ 100     │ 100
│     │                        3.957 Mitem/s │ 1.853 Mitem/s │ 2.826 Mitem/s │ 2.761 Mitem/s │         │
│     ├─ 4                     7.999 ms      │ 21.62 ms      │ 11.26 ms      │ 11.16 ms      │ 100     │ 100
│     │                        5 Mitem/s     │ 1.849 Mitem/s │ 3.55 Mitem/s  │ 3.583 Mitem/s │         │
│     ├─ 5                     8.403 ms      │ 20.32 ms      │ 12.31 ms      │ 12.29 ms      │ 100     │ 100
│     │                        5.949 Mitem/s │ 2.459 Mitem/s │ 4.058 Mitem/s │ 4.068 Mitem/s │         │
│     ╰─ 6                     8.65 ms       │ 20.92 ms      │ 12.76 ms      │ 13.01 ms      │ 100     │ 100
│                              6.936 Mitem/s │ 2.867 Mitem/s │ 4.7 Mitem/s   │ 4.611 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     10.29 ms      │ 16.22 ms      │ 11.16 ms      │ 11.35 ms      │ 100     │ 100
│     │                        971.5 Kitem/s │ 616.1 Kitem/s │ 895.9 Kitem/s │ 880.5 Kitem/s │         │
│     ├─ 2                     10.59 ms      │ 21.46 ms      │ 11.49 ms      │ 12.24 ms      │ 100     │ 100
│     │                        1.886 Mitem/s │ 931.8 Kitem/s │ 1.74 Mitem/s  │ 1.633 Mitem/s │         │
│     ├─ 3                     11.22 ms      │ 22.43 ms      │ 16.32 ms      │ 17.09 ms      │ 100     │ 100
│     │                        2.671 Mitem/s │ 1.337 Mitem/s │ 1.837 Mitem/s │ 1.755 Mitem/s │         │
│     ├─ 4                     11.01 ms      │ 30.07 ms      │ 16.84 ms      │ 17.52 ms      │ 100     │ 100
│     │                        3.631 Mitem/s │ 1.33 Mitem/s  │ 2.373 Mitem/s │ 2.282 Mitem/s │         │
│     ├─ 5                     11.03 ms      │ 32.09 ms      │ 18.53 ms      │ 18.43 ms      │ 100     │ 100
│     │                        4.529 Mitem/s │ 1.557 Mitem/s │ 2.697 Mitem/s │ 2.712 Mitem/s │         │
│     ╰─ 6                     11.49 ms      │ 30.71 ms      │ 19.19 ms      │ 20.18 ms      │ 100     │ 100
│                              5.22 Mitem/s  │ 1.953 Mitem/s │ 3.125 Mitem/s │ 2.972 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.096 ms      │ 9.786 ms      │ 7.352 ms      │ 7.583 ms      │ 100     │ 100
│     │                        1.409 Mitem/s │ 1.021 Mitem/s │ 1.36 Mitem/s  │ 1.318 Mitem/s │         │
│     ├─ 2                     7.857 ms      │ 15.99 ms      │ 10.9 ms       │ 10.82 ms      │ 100     │ 100
│     │                        2.545 Mitem/s │ 1.25 Mitem/s  │ 1.833 Mitem/s │ 1.848 Mitem/s │         │
│     ├─ 3                     7.82 ms       │ 16.33 ms      │ 11.94 ms      │ 12.34 ms      │ 100     │ 100
│     │                        3.835 Mitem/s │ 1.836 Mitem/s │ 2.511 Mitem/s │ 2.429 Mitem/s │         │
│     ├─ 4                     8.42 ms       │ 19.12 ms      │ 12.03 ms      │ 12.16 ms      │ 100     │ 100
│     │                        4.75 Mitem/s  │ 2.091 Mitem/s │ 3.323 Mitem/s │ 3.288 Mitem/s │         │
│     ├─ 5                     8.861 ms      │ 23.68 ms      │ 13.57 ms      │ 14.21 ms      │ 100     │ 100
│     │                        5.642 Mitem/s │ 2.111 Mitem/s │ 3.682 Mitem/s │ 3.516 Mitem/s │         │
│     ╰─ 6                     8.894 ms      │ 22.11 ms      │ 13.75 ms      │ 14.15 ms      │ 100     │ 100
│                              6.745 Mitem/s │ 2.713 Mitem/s │ 4.362 Mitem/s │ 4.237 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     6.484 ms      │ 10.68 ms      │ 6.668 ms      │ 6.973 ms      │ 100     │ 100
│     │                        1.542 Mitem/s │ 935.4 Kitem/s │ 1.499 Mitem/s │ 1.433 Mitem/s │         │
│     ├─ 2                     6.842 ms      │ 15.22 ms      │ 8.785 ms      │ 9.479 ms      │ 100     │ 100
│     │                        2.922 Mitem/s │ 1.313 Mitem/s │ 2.276 Mitem/s │ 2.109 Mitem/s │         │
│     ├─ 3                     7.291 ms      │ 16.81 ms      │ 11.03 ms      │ 11.08 ms      │ 100     │ 100
│     │                        4.114 Mitem/s │ 1.784 Mitem/s │ 2.717 Mitem/s │ 2.706 Mitem/s │         │
│     ├─ 4                     7.968 ms      │ 21.97 ms      │ 11.15 ms      │ 11.18 ms      │ 100     │ 100
│     │                        5.019 Mitem/s │ 1.819 Mitem/s │ 3.586 Mitem/s │ 3.575 Mitem/s │         │
│     ├─ 5                     8.495 ms      │ 22.77 ms      │ 12.69 ms      │ 12.99 ms      │ 100     │ 100
│     │                        5.885 Mitem/s │ 2.194 Mitem/s │ 3.939 Mitem/s │ 3.848 Mitem/s │         │
│     ╰─ 6                     8.508 ms      │ 20.56 ms      │ 12.75 ms      │ 13.24 ms      │ 100     │ 100
│                              7.051 Mitem/s │ 2.918 Mitem/s │ 4.704 Mitem/s │ 4.53 Mitem/s  │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.1 ms        │ 14.06 ms      │ 7.553 ms      │ 8.569 ms      │ 100     │ 100
│     │                        1.408 Mitem/s │ 711.1 Kitem/s │ 1.323 Mitem/s │ 1.166 Mitem/s │         │
│     ├─ 2                     7.736 ms      │ 14.88 ms      │ 10.56 ms      │ 10.49 ms      │ 100     │ 100
│     │                        2.585 Mitem/s │ 1.343 Mitem/s │ 1.892 Mitem/s │ 1.905 Mitem/s │         │
│     ├─ 3                     8.414 ms      │ 16.28 ms      │ 11.77 ms      │ 12.23 ms      │ 100     │ 100
│     │                        3.565 Mitem/s │ 1.842 Mitem/s │ 2.548 Mitem/s │ 2.451 Mitem/s │         │
│     ├─ 4                     8.86 ms       │ 20.17 ms      │ 12.3 ms       │ 12.32 ms      │ 100     │ 100
│     │                        4.514 Mitem/s │ 1.982 Mitem/s │ 3.251 Mitem/s │ 3.245 Mitem/s │         │
│     ├─ 5                     8.951 ms      │ 22.62 ms      │ 13.13 ms      │ 13.27 ms      │ 100     │ 100
│     │                        5.585 Mitem/s │ 2.209 Mitem/s │ 3.805 Mitem/s │ 3.767 Mitem/s │         │
│     ╰─ 6                     9.183 ms      │ 22.3 ms       │ 13.8 ms       │ 14.72 ms      │ 100     │ 100
│                              6.533 Mitem/s │ 2.689 Mitem/s │ 4.345 Mitem/s │ 4.075 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.09 ms       │ 9.859 ms      │ 7.307 ms      │ 7.576 ms      │ 100     │ 100
│     │                        1.41 Mitem/s  │ 1.014 Mitem/s │ 1.368 Mitem/s │ 1.319 Mitem/s │         │
│     ├─ 2                     7.487 ms      │ 15.62 ms      │ 9.187 ms      │ 10.04 ms      │ 100     │ 100
│     │                        2.671 Mitem/s │ 1.279 Mitem/s │ 2.176 Mitem/s │ 1.991 Mitem/s │         │
│     ├─ 3                     7.789 ms      │ 18.64 ms      │ 11.47 ms      │ 11.43 ms      │ 100     │ 100
│     │                        3.851 Mitem/s │ 1.609 Mitem/s │ 2.615 Mitem/s │ 2.623 Mitem/s │         │
│     ├─ 4                     8.618 ms      │ 20.75 ms      │ 13.1 ms       │ 12.73 ms      │ 100     │ 100
│     │                        4.641 Mitem/s │ 1.927 Mitem/s │ 3.052 Mitem/s │ 3.139 Mitem/s │         │
│     ├─ 5                     9.142 ms      │ 24.79 ms      │ 13.54 ms      │ 13.75 ms      │ 100     │ 100
│     │                        5.468 Mitem/s │ 2.016 Mitem/s │ 3.691 Mitem/s │ 3.634 Mitem/s │         │
│     ╰─ 6                     9.237 ms      │ 22.3 ms       │ 13.53 ms      │ 14.31 ms      │ 100     │ 100
│                              6.495 Mitem/s │ 2.689 Mitem/s │ 4.432 Mitem/s │ 4.192 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.061 ms      │ 13.54 ms      │ 7.333 ms      │ 7.893 ms      │ 100     │ 100
│     │                        1.416 Mitem/s │ 738.4 Kitem/s │ 1.363 Mitem/s │ 1.266 Mitem/s │         │
│     ├─ 2                     7.451 ms      │ 15.75 ms      │ 10.82 ms      │ 10.69 ms      │ 100     │ 100
│     │                        2.684 Mitem/s │ 1.269 Mitem/s │ 1.847 Mitem/s │ 1.869 Mitem/s │         │
│     ├─ 3                     7.71 ms       │ 16.19 ms      │ 11.96 ms      │ 12.19 ms      │ 100     │ 100
│     │                        3.89 Mitem/s  │ 1.852 Mitem/s │ 2.507 Mitem/s │ 2.459 Mitem/s │         │
│     ├─ 4                     8.622 ms      │ 18.97 ms      │ 13.02 ms      │ 12.41 ms      │ 100     │ 100
│     │                        4.639 Mitem/s │ 2.107 Mitem/s │ 3.071 Mitem/s │ 3.221 Mitem/s │         │
│     ├─ 5                     9.111 ms      │ 23.32 ms      │ 13.24 ms      │ 13.33 ms      │ 100     │ 100
│     │                        5.487 Mitem/s │ 2.143 Mitem/s │ 3.775 Mitem/s │ 3.748 Mitem/s │         │
│     ╰─ 6                     9.5 ms        │ 22.61 ms      │ 14.08 ms      │ 15.47 ms      │ 100     │ 100
│                              6.315 Mitem/s │ 2.652 Mitem/s │ 4.259 Mitem/s │ 3.877 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.204 ms      │ 14.34 ms      │ 7.645 ms      │ 8.461 ms      │ 100     │ 100
│     │                        1.388 Mitem/s │ 697.2 Kitem/s │ 1.308 Mitem/s │ 1.181 Mitem/s │         │
│     ├─ 2                     7.935 ms      │ 16.36 ms      │ 11.06 ms      │ 11.14 ms      │ 100     │ 100
│     │                        2.52 Mitem/s  │ 1.221 Mitem/s │ 1.807 Mitem/s │ 1.794 Mitem/s │         │
│     ├─ 3                     7.899 ms      │ 16.17 ms      │ 11.87 ms      │ 12.27 ms      │ 100     │ 100
│     │                        3.797 Mitem/s │ 1.854 Mitem/s │ 2.526 Mitem/s │ 2.444 Mitem/s │         │
│     ├─ 4                     8.806 ms      │ 21.32 ms      │ 13.47 ms      │ 13.53 ms      │ 100     │ 100
│     │                        4.541 Mitem/s │ 1.875 Mitem/s │ 2.968 Mitem/s │ 2.954 Mitem/s │         │
│     ├─ 5                     9.133 ms      │ 22.19 ms      │ 13.45 ms      │ 13.49 ms      │ 100     │ 100
│     │                        5.474 Mitem/s │ 2.252 Mitem/s │ 3.716 Mitem/s │ 3.706 Mitem/s │         │
│     ╰─ 6                     9.056 ms      │ 22.16 ms      │ 13.75 ms      │ 13.84 ms      │ 100     │ 100
│                              6.624 Mitem/s │ 2.706 Mitem/s │ 4.361 Mitem/s │ 4.334 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 3                     8.963 ms      │ 15.37 ms      │ 11.35 ms      │ 11.6 ms       │ 100     │ 100
│     ├─ 4                     9.327 ms      │ 16.06 ms      │ 14.13 ms      │ 13.74 ms      │ 100     │ 100
│     ├─ 5                     11.26 ms      │ 24.51 ms      │ 14.06 ms      │ 14.23 ms      │ 100     │ 100
│     ╰─ 6                     11.59 ms      │ 24 ms         │ 14.38 ms      │ 14.91 ms      │ 100     │ 100
├─ 14_prefix_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     19.89 ms      │ 22.06 ms      │ 20.73 ms      │ 20.8 ms       │ 20      │ 20
│     │                        5.025 Kitem/s │ 4.531 Kitem/s │ 4.823 Kitem/s │ 4.805 Kitem/s │         │
│     ├─ 2                     23.34 ms      │ 52.24 ms      │ 45.76 ms      │ 41.75 ms      │ 20      │ 20
│     │                        8.566 Kitem/s │ 3.827 Kitem/s │ 4.37 Kitem/s  │ 4.789 Kitem/s │         │
│     ├─ 3                     22.15 ms      │ 59.26 ms      │ 53.89 ms      │ 53.84 ms      │ 20      │ 20
│     │                        13.54 Kitem/s │ 5.061 Kitem/s │ 5.566 Kitem/s │ 5.571 Kitem/s │         │
│     ├─ 4                     32.23 ms      │ 66.1 ms       │ 57.62 ms      │ 56.17 ms      │ 20      │ 20
│     │                        12.4 Kitem/s  │ 6.05 Kitem/s  │ 6.941 Kitem/s │ 7.12 Kitem/s  │         │
│     ├─ 5                     30.4 ms       │ 65.19 ms      │ 57.69 ms      │ 54.82 ms      │ 20      │ 20
│     │                        16.44 Kitem/s │ 7.669 Kitem/s │ 8.666 Kitem/s │ 9.12 Kitem/s  │         │
│     ╰─ 6                     31.98 ms      │ 104.8 ms      │ 64.47 ms      │ 61.89 ms      │ 20      │ 20
│                              18.75 Kitem/s │ 5.724 Kitem/s │ 9.306 Kitem/s │ 9.693 Kitem/s │         │
├─ 15_full_scan_aggregate                    │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     33.44 ms      │ 43.67 ms      │ 35.16 ms      │ 35.38 ms      │ 100     │ 100
│     │                        2.989 Kitem/s │ 2.289 Kitem/s │ 2.843 Kitem/s │ 2.826 Kitem/s │         │
│     ├─ 2                     34.16 ms      │ 60.49 ms      │ 35.31 ms      │ 36.36 ms      │ 100     │ 100
│     │                        5.853 Kitem/s │ 3.305 Kitem/s │ 5.663 Kitem/s │ 5.499 Kitem/s │         │
│     ├─ 3                     34.42 ms      │ 65.15 ms      │ 36.56 ms      │ 39.87 ms      │ 100     │ 100
│     │                        8.714 Kitem/s │ 4.604 Kitem/s │ 8.205 Kitem/s │ 7.522 Kitem/s │         │
│     ├─ 4                     34.94 ms      │ 76.11 ms      │ 45.73 ms      │ 47.82 ms      │ 100     │ 100
│     │                        11.44 Kitem/s │ 5.254 Kitem/s │ 8.745 Kitem/s │ 8.364 Kitem/s │         │
│     ├─ 5                     35.17 ms      │ 79.75 ms      │ 58.21 ms      │ 53.92 ms      │ 100     │ 100
│     │                        14.21 Kitem/s │ 6.269 Kitem/s │ 8.589 Kitem/s │ 9.272 Kitem/s │         │
│     ╰─ 6                     36.18 ms      │ 83.03 ms      │ 60.62 ms      │ 60.74 ms      │ 100     │ 100
│                              16.58 Kitem/s │ 7.226 Kitem/s │ 9.896 Kitem/s │ 9.877 Kitem/s │         │
├─ 16_insert_heavy                           │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     1.436 ms      │ 2.793 ms      │ 2.087 ms      │ 2.065 ms      │ 100     │ 100
│     │                        6.959 Mitem/s │ 3.579 Mitem/s │ 4.79 Mitem/s  │ 4.84 Mitem/s  │         │
│     ├─ 2                     1.642 ms      │ 3.702 ms      │ 2.484 ms      │ 2.487 ms      │ 100     │ 100
│     │                        12.17 Mitem/s │ 5.402 Mitem/s │ 8.051 Mitem/s │ 8.04 Mitem/s  │         │
│     ├─ 3                     1.978 ms      │ 4.384 ms      │ 3.066 ms      │ 3.031 ms      │ 100     │ 100
│     │                        15.15 Mitem/s │ 6.842 Mitem/s │ 9.783 Mitem/s │ 9.897 Mitem/s │         │
│     ├─ 4                     2.423 ms      │ 4.864 ms      │ 3.242 ms      │ 3.309 ms      │ 100     │ 100
│     │                        16.5 Mitem/s  │ 8.222 Mitem/s │ 12.33 Mitem/s │ 12.08 Mitem/s │         │
│     ├─ 5                     2.683 ms      │ 7.119 ms      │ 3.758 ms      │ 3.701 ms      │ 100     │ 100
│     │                        18.63 Mitem/s │ 7.022 Mitem/s │ 13.3 Mitem/s  │ 13.5 Mitem/s  │         │
│     ╰─ 6                     2.89 ms       │ 4.893 ms      │ 4.044 ms      │ 3.95 ms       │ 100     │ 100
│                              20.75 Mitem/s │ 12.26 Mitem/s │ 14.83 Mitem/s │ 15.18 Mitem/s │         │
├─ 17_hot_spot                               │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     1.015 ms      │ 2.209 ms      │ 1.727 ms      │ 1.656 ms      │ 100     │ 100
│     │                        9.851 Mitem/s │ 4.525 Mitem/s │ 5.789 Mitem/s │ 6.036 Mitem/s │         │
│     ├─ 2                     2.106 ms      │ 5.643 ms      │ 4.633 ms      │ 4.279 ms      │ 100     │ 100
│     │                        9.492 Mitem/s │ 3.543 Mitem/s │ 4.316 Mitem/s │ 4.673 Mitem/s │         │
│     ├─ 3                     4.165 ms      │ 7.611 ms      │ 6.583 ms      │ 6.515 ms      │ 100     │ 100
│     │                        7.202 Mitem/s │ 3.941 Mitem/s │ 4.557 Mitem/s │ 4.604 Mitem/s │         │
│     ├─ 4                     8.304 ms      │ 9.853 ms      │ 9.098 ms      │ 9.069 ms      │ 100     │ 100
│     │                        4.816 Mitem/s │ 4.059 Mitem/s │ 4.396 Mitem/s │ 4.41 Mitem/s  │         │
│     ├─ 5                     5.91 ms       │ 12.72 ms      │ 11.29 ms      │ 11.24 ms      │ 100     │ 100
│     │                        8.459 Mitem/s │ 3.929 Mitem/s │ 4.426 Mitem/s │ 4.447 Mitem/s │         │
│     ╰─ 6                     11.98 ms      │ 17.33 ms      │ 14.07 ms      │ 14.09 ms      │ 100     │ 100
│                              5.005 Mitem/s │ 3.46 Mitem/s  │ 4.261 Mitem/s │ 4.257 Mitem/s │         │
╰─ 18_split_inducing_scan                    │               │               │               │         │
   ╰─ masstree24                             │               │               │               │         │
      ├─ 3                     7.968 ms      │ 18.22 ms      │ 8.942 ms      │ 9.761 ms      │ 100     │ 100
      ├─ 4                     8.501 ms      │ 24.18 ms      │ 13.51 ms      │ 13.1 ms       │ 100     │ 100
      ├─ 5                     9.447 ms      │ 25.53 ms      │ 14.94 ms      │ 14.53 ms      │ 100     │ 100
      ╰─ 6                     10.89 ms      │ 26.49 ms      │ 15.18 ms      │ 16.34 ms      │ 100     │ 100
```
