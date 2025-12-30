```text
Timer precision: 30 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.443 ms      │ 7.464 ms      │ 4.587 ms      │ 4.718 ms      │ 100     │ 100
│  │  │                        2.25 Mitem/s  │ 1.339 Mitem/s │ 2.179 Mitem/s │ 2.119 Mitem/s │         │
│  │  ├─ 2                     10.46 ms      │ 20.72 ms      │ 15.88 ms      │ 16.08 ms      │ 100     │ 100
│  │  │                        1.91 Mitem/s  │ 964.7 Kitem/s │ 1.258 Mitem/s │ 1.243 Mitem/s │         │
│  │  ├─ 3                     22.76 ms      │ 34.79 ms      │ 30.16 ms      │ 29.43 ms      │ 100     │ 100
│  │  │                        1.317 Mitem/s │ 862 Kitem/s   │ 994.6 Kitem/s │ 1.019 Mitem/s │         │
│  │  ├─ 4                     32.54 ms      │ 60.14 ms      │ 43.22 ms      │ 43.25 ms      │ 100     │ 100
│  │  │                        1.228 Mitem/s │ 665.1 Kitem/s │ 925.3 Kitem/s │ 924.7 Kitem/s │         │
│  │  ├─ 5                     51.63 ms      │ 82.78 ms      │ 60.1 ms       │ 60.8 ms       │ 100     │ 100
│  │  │                        968.3 Kitem/s │ 604 Kitem/s   │ 831.9 Kitem/s │ 822.2 Kitem/s │         │
│  │  ╰─ 6                     81.16 ms      │ 112.5 ms      │ 95.22 ms      │ 95.82 ms      │ 100     │ 100
│  │                           739.2 Kitem/s │ 533 Kitem/s   │ 630 Kitem/s   │ 626.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.489 ms      │ 12.56 ms      │ 6.83 ms       │ 8.297 ms      │ 100     │ 100
│  │  │                        1.54 Mitem/s  │ 796 Kitem/s   │ 1.464 Mitem/s │ 1.205 Mitem/s │         │
│  │  ├─ 2                     6.862 ms      │ 15.66 ms      │ 10.46 ms      │ 10.83 ms      │ 100     │ 100
│  │  │                        2.914 Mitem/s │ 1.276 Mitem/s │ 1.911 Mitem/s │ 1.846 Mitem/s │         │
│  │  ├─ 3                     7.623 ms      │ 15.77 ms      │ 12.27 ms      │ 12.18 ms      │ 100     │ 100
│  │  │                        3.934 Mitem/s │ 1.902 Mitem/s │ 2.444 Mitem/s │ 2.461 Mitem/s │         │
│  │  ├─ 4                     7.68 ms       │ 23.26 ms      │ 12.34 ms      │ 12.14 ms      │ 100     │ 100
│  │  │                        5.208 Mitem/s │ 1.718 Mitem/s │ 3.241 Mitem/s │ 3.294 Mitem/s │         │
│  │  ├─ 5                     7.841 ms      │ 23.62 ms      │ 12.9 ms       │ 13.3 ms       │ 100     │ 100
│  │  │                        6.376 Mitem/s │ 2.116 Mitem/s │ 3.874 Mitem/s │ 3.759 Mitem/s │         │
│  │  ╰─ 6                     7.871 ms      │ 21.86 ms      │ 12.78 ms      │ 12.81 ms      │ 100     │ 100
│  │                           7.622 Mitem/s │ 2.743 Mitem/s │ 4.693 Mitem/s │ 4.682 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.349 ms      │ 12.08 ms      │ 9.728 ms      │ 9.923 ms      │ 100     │ 100
│     │                        1.069 Mitem/s │ 827.5 Kitem/s │ 1.027 Mitem/s │ 1.007 Mitem/s │         │
│     ├─ 2                     9.567 ms      │ 21.93 ms      │ 11.26 ms      │ 12.4 ms       │ 100     │ 100
│     │                        2.09 Mitem/s  │ 911.8 Kitem/s │ 1.775 Mitem/s │ 1.612 Mitem/s │         │
│     ├─ 3                     9.784 ms      │ 21.03 ms      │ 14.68 ms      │ 15 ms         │ 100     │ 100
│     │                        3.066 Mitem/s │ 1.426 Mitem/s │ 2.043 Mitem/s │ 1.998 Mitem/s │         │
│     ├─ 4                     10.01 ms      │ 21.32 ms      │ 17.09 ms      │ 16.4 ms       │ 100     │ 100
│     │                        3.995 Mitem/s │ 1.876 Mitem/s │ 2.339 Mitem/s │ 2.438 Mitem/s │         │
│     ├─ 5                     10.26 ms      │ 31.5 ms       │ 18.1 ms       │ 17.99 ms      │ 100     │ 100
│     │                        4.868 Mitem/s │ 1.587 Mitem/s │ 2.761 Mitem/s │ 2.778 Mitem/s │         │
│     ╰─ 6                     12.99 ms      │ 32.94 ms      │ 18.65 ms      │ 20.79 ms      │ 100     │ 100
│                              4.616 Mitem/s │ 1.821 Mitem/s │ 3.216 Mitem/s │ 2.885 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.718 ms      │ 7.698 ms      │ 4.829 ms      │ 5.057 ms      │ 100     │ 100
│  │  │                        2.119 Mitem/s │ 1.298 Mitem/s │ 2.07 Mitem/s  │ 1.977 Mitem/s │         │
│  │  ├─ 2                     11.89 ms      │ 21.03 ms      │ 16.08 ms      │ 15.89 ms      │ 100     │ 100
│  │  │                        1.681 Mitem/s │ 951 Kitem/s   │ 1.243 Mitem/s │ 1.258 Mitem/s │         │
│  │  ├─ 3                     23.12 ms      │ 37.22 ms      │ 29.94 ms      │ 30.08 ms      │ 100     │ 100
│  │  │                        1.297 Mitem/s │ 805.8 Kitem/s │ 1.001 Mitem/s │ 997 Kitem/s   │         │
│  │  ├─ 4                     35.21 ms      │ 55.45 ms      │ 43.2 ms       │ 43.51 ms      │ 100     │ 100
│  │  │                        1.135 Mitem/s │ 721.2 Kitem/s │ 925.8 Kitem/s │ 919.2 Kitem/s │         │
│  │  ├─ 5                     55.53 ms      │ 79.69 ms      │ 64.51 ms      │ 64.51 ms      │ 100     │ 100
│  │  │                        900.3 Kitem/s │ 627.3 Kitem/s │ 775 Kitem/s   │ 774.9 Kitem/s │         │
│  │  ╰─ 6                     88.05 ms      │ 117.7 ms      │ 99.39 ms      │ 99.5 ms       │ 100     │ 100
│  │                           681.3 Kitem/s │ 509.5 Kitem/s │ 603.6 Kitem/s │ 603 Kitem/s   │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.797 ms      │ 9.43 ms       │ 6.927 ms      │ 7.151 ms      │ 100     │ 100
│  │  │                        1.471 Mitem/s │ 1.06 Mitem/s  │ 1.443 Mitem/s │ 1.398 Mitem/s │         │
│  │  ├─ 2                     7.233 ms      │ 14.91 ms      │ 9.731 ms      │ 10.18 ms      │ 100     │ 100
│  │  │                        2.764 Mitem/s │ 1.34 Mitem/s  │ 2.055 Mitem/s │ 1.962 Mitem/s │         │
│  │  ├─ 3                     8.882 ms      │ 15.84 ms      │ 13.82 ms      │ 13.52 ms      │ 100     │ 100
│  │  │                        3.377 Mitem/s │ 1.893 Mitem/s │ 2.17 Mitem/s  │ 2.217 Mitem/s │         │
│  │  ├─ 4                     7.915 ms      │ 18.58 ms      │ 12.29 ms      │ 11.83 ms      │ 100     │ 100
│  │  │                        5.053 Mitem/s │ 2.152 Mitem/s │ 3.253 Mitem/s │ 3.381 Mitem/s │         │
│  │  ├─ 5                     8.15 ms       │ 23.77 ms      │ 12.94 ms      │ 13.29 ms      │ 100     │ 100
│  │  │                        6.134 Mitem/s │ 2.103 Mitem/s │ 3.861 Mitem/s │ 3.762 Mitem/s │         │
│  │  ╰─ 6                     8.104 ms      │ 22.04 ms      │ 13.12 ms      │ 13.53 ms      │ 100     │ 100
│  │                           7.403 Mitem/s │ 2.721 Mitem/s │ 4.572 Mitem/s │ 4.432 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.878 ms      │ 17.67 ms      │ 10.39 ms      │ 10.97 ms      │ 100     │ 100
│     │                        1.012 Mitem/s │ 565.7 Kitem/s │ 961.5 Kitem/s │ 911.5 Kitem/s │         │
│     ├─ 2                     10.02 ms      │ 20.91 ms      │ 13.47 ms      │ 13.19 ms      │ 100     │ 100
│     │                        1.994 Mitem/s │ 956.4 Kitem/s │ 1.484 Mitem/s │ 1.515 Mitem/s │         │
│     ├─ 3                     10.45 ms      │ 21.9 ms       │ 15.84 ms      │ 16.69 ms      │ 100     │ 100
│     │                        2.87 Mitem/s  │ 1.369 Mitem/s │ 1.893 Mitem/s │ 1.797 Mitem/s │         │
│     ├─ 4                     10.54 ms      │ 31.87 ms      │ 15.85 ms      │ 17.07 ms      │ 100     │ 100
│     │                        3.793 Mitem/s │ 1.254 Mitem/s │ 2.522 Mitem/s │ 2.343 Mitem/s │         │
│     ├─ 5                     10.64 ms      │ 34.57 ms      │ 18.24 ms      │ 18.61 ms      │ 100     │ 100
│     │                        4.698 Mitem/s │ 1.446 Mitem/s │ 2.74 Mitem/s  │ 2.685 Mitem/s │         │
│     ╰─ 6                     10.61 ms      │ 33.62 ms      │ 19.18 ms      │ 21.02 ms      │ 100     │ 100
│                              5.653 Mitem/s │ 1.784 Mitem/s │ 3.127 Mitem/s │ 2.853 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.545 ms      │ 9.167 ms      │ 4.706 ms      │ 5.289 ms      │ 100     │ 100
│  │  │                        2.199 Mitem/s │ 1.09 Mitem/s  │ 2.124 Mitem/s │ 1.89 Mitem/s  │         │
│  │  ├─ 2                     10.31 ms      │ 19.96 ms      │ 14.48 ms      │ 14.57 ms      │ 100     │ 100
│  │  │                        1.938 Mitem/s │ 1.001 Mitem/s │ 1.38 Mitem/s  │ 1.372 Mitem/s │         │
│  │  ├─ 3                     21.61 ms      │ 40.91 ms      │ 28.76 ms      │ 29.16 ms      │ 100     │ 100
│  │  │                        1.387 Mitem/s │ 733.2 Kitem/s │ 1.042 Mitem/s │ 1.028 Mitem/s │         │
│  │  ├─ 4                     34.42 ms      │ 58.79 ms      │ 42.76 ms      │ 43.27 ms      │ 100     │ 100
│  │  │                        1.161 Mitem/s │ 680.3 Kitem/s │ 935.3 Kitem/s │ 924.4 Kitem/s │         │
│  │  ├─ 5                     50.54 ms      │ 74.37 ms      │ 61.34 ms      │ 61.96 ms      │ 100     │ 100
│  │  │                        989.1 Kitem/s │ 672.2 Kitem/s │ 815 Kitem/s   │ 806.8 Kitem/s │         │
│  │  ╰─ 6                     85.08 ms      │ 123 ms        │ 98.27 ms      │ 98.6 ms       │ 100     │ 100
│  │                           705.2 Kitem/s │ 487.4 Kitem/s │ 610.5 Kitem/s │ 608.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.706 ms      │ 9.654 ms      │ 6.853 ms      │ 7.028 ms      │ 100     │ 100
│  │  │                        1.491 Mitem/s │ 1.035 Mitem/s │ 1.459 Mitem/s │ 1.422 Mitem/s │         │
│  │  ├─ 2                     7.038 ms      │ 14.7 ms       │ 10.58 ms      │ 10.35 ms      │ 100     │ 100
│  │  │                        2.841 Mitem/s │ 1.359 Mitem/s │ 1.889 Mitem/s │ 1.931 Mitem/s │         │
│  │  ├─ 3                     7.138 ms      │ 15 ms         │ 13.03 ms      │ 12.87 ms      │ 100     │ 100
│  │  │                        4.202 Mitem/s │ 1.998 Mitem/s │ 2.301 Mitem/s │ 2.33 Mitem/s  │         │
│  │  ├─ 4                     7.915 ms      │ 22.11 ms      │ 12.68 ms      │ 12.39 ms      │ 100     │ 100
│  │  │                        5.053 Mitem/s │ 1.808 Mitem/s │ 3.152 Mitem/s │ 3.227 Mitem/s │         │
│  │  ├─ 5                     7.882 ms      │ 23.08 ms      │ 12.93 ms      │ 13.47 ms      │ 100     │ 100
│  │  │                        6.343 Mitem/s │ 2.166 Mitem/s │ 3.866 Mitem/s │ 3.711 Mitem/s │         │
│  │  ╰─ 6                     8.083 ms      │ 22.57 ms      │ 13.13 ms      │ 13.55 ms      │ 100     │ 100
│  │                           7.422 Mitem/s │ 2.657 Mitem/s │ 4.568 Mitem/s │ 4.427 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.402 ms      │ 11.79 ms      │ 9.745 ms      │ 9.88 ms       │ 100     │ 100
│     │                        1.063 Mitem/s │ 848.1 Kitem/s │ 1.026 Mitem/s │ 1.012 Mitem/s │         │
│     ├─ 2                     9.696 ms      │ 19.32 ms      │ 10.84 ms      │ 11.72 ms      │ 100     │ 100
│     │                        2.062 Mitem/s │ 1.034 Mitem/s │ 1.844 Mitem/s │ 1.705 Mitem/s │         │
│     ├─ 3                     10.56 ms      │ 20.34 ms      │ 14.65 ms      │ 15.32 ms      │ 100     │ 100
│     │                        2.838 Mitem/s │ 1.474 Mitem/s │ 2.047 Mitem/s │ 1.957 Mitem/s │         │
│     ├─ 4                     9.903 ms      │ 27.59 ms      │ 15.18 ms      │ 15.88 ms      │ 100     │ 100
│     │                        4.039 Mitem/s │ 1.449 Mitem/s │ 2.633 Mitem/s │ 2.518 Mitem/s │         │
│     ├─ 5                     9.946 ms      │ 35.62 ms      │ 17.61 ms      │ 18.56 ms      │ 100     │ 100
│     │                        5.026 Mitem/s │ 1.403 Mitem/s │ 2.839 Mitem/s │ 2.693 Mitem/s │         │
│     ╰─ 6                     11.66 ms      │ 32.17 ms      │ 18.57 ms      │ 19.87 ms      │ 100     │ 100
│                              5.143 Mitem/s │ 1.864 Mitem/s │ 3.23 Mitem/s  │ 3.019 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.517 ms      │ 6.619 ms      │ 4.598 ms      │ 4.667 ms      │ 100     │ 100
│  │  │                        2.213 Mitem/s │ 1.51 Mitem/s  │ 2.174 Mitem/s │ 2.142 Mitem/s │         │
│  │  ├─ 2                     10.35 ms      │ 20.4 ms       │ 16.27 ms      │ 16.06 ms      │ 100     │ 100
│  │  │                        1.931 Mitem/s │ 980.1 Kitem/s │ 1.228 Mitem/s │ 1.244 Mitem/s │         │
│  │  ├─ 3                     20.44 ms      │ 39.38 ms      │ 29.56 ms      │ 29.32 ms      │ 100     │ 100
│  │  │                        1.467 Mitem/s │ 761.7 Kitem/s │ 1.014 Mitem/s │ 1.022 Mitem/s │         │
│  │  ├─ 4                     33.22 ms      │ 56.43 ms      │ 42.18 ms      │ 42.62 ms      │ 100     │ 100
│  │  │                        1.203 Mitem/s │ 708.7 Kitem/s │ 948.2 Kitem/s │ 938.3 Kitem/s │         │
│  │  ├─ 5                     51.51 ms      │ 80.39 ms      │ 60.14 ms      │ 60.93 ms      │ 100     │ 100
│  │  │                        970.5 Kitem/s │ 621.9 Kitem/s │ 831.2 Kitem/s │ 820.5 Kitem/s │         │
│  │  ╰─ 6                     79.79 ms      │ 127.9 ms      │ 96.6 ms       │ 96.88 ms      │ 100     │ 100
│  │                           751.9 Kitem/s │ 468.9 Kitem/s │ 621.1 Kitem/s │ 619.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.564 ms      │ 9.627 ms      │ 6.998 ms      │ 7.799 ms      │ 100     │ 100
│  │  │                        1.523 Mitem/s │ 1.038 Mitem/s │ 1.428 Mitem/s │ 1.282 Mitem/s │         │
│  │  ├─ 2                     6.907 ms      │ 14.72 ms      │ 12.31 ms      │ 11.08 ms      │ 100     │ 100
│  │  │                        2.895 Mitem/s │ 1.358 Mitem/s │ 1.624 Mitem/s │ 1.803 Mitem/s │         │
│  │  ├─ 3                     7.185 ms      │ 14.63 ms      │ 12.3 ms       │ 11.7 ms       │ 100     │ 100
│  │  │                        4.175 Mitem/s │ 2.049 Mitem/s │ 2.438 Mitem/s │ 2.562 Mitem/s │         │
│  │  ├─ 4                     7.619 ms      │ 20.65 ms      │ 12.81 ms      │ 12.54 ms      │ 100     │ 100
│  │  │                        5.249 Mitem/s │ 1.936 Mitem/s │ 3.12 Mitem/s  │ 3.187 Mitem/s │         │
│  │  ├─ 5                     7.791 ms      │ 22.31 ms      │ 12.67 ms      │ 12.64 ms      │ 100     │ 100
│  │  │                        6.417 Mitem/s │ 2.24 Mitem/s  │ 3.945 Mitem/s │ 3.954 Mitem/s │         │
│  │  ╰─ 6                     7.818 ms      │ 21.6 ms       │ 12.9 ms       │ 13.51 ms      │ 100     │ 100
│  │                           7.673 Mitem/s │ 2.777 Mitem/s │ 4.647 Mitem/s │ 4.438 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.315 ms      │ 14.26 ms      │ 9.665 ms      │ 10.07 ms      │ 100     │ 100
│     │                        1.073 Mitem/s │ 700.9 Kitem/s │ 1.034 Mitem/s │ 992.4 Kitem/s │         │
│     ├─ 2                     9.714 ms      │ 20.3 ms       │ 13.75 ms      │ 13.76 ms      │ 100     │ 100
│     │                        2.058 Mitem/s │ 985 Kitem/s   │ 1.454 Mitem/s │ 1.453 Mitem/s │         │
│     ├─ 3                     9.821 ms      │ 20.52 ms      │ 14.65 ms      │ 15.03 ms      │ 100     │ 100
│     │                        3.054 Mitem/s │ 1.461 Mitem/s │ 2.047 Mitem/s │ 1.994 Mitem/s │         │
│     ├─ 4                     9.929 ms      │ 32.06 ms      │ 15.04 ms      │ 16.13 ms      │ 100     │ 100
│     │                        4.028 Mitem/s │ 1.247 Mitem/s │ 2.658 Mitem/s │ 2.478 Mitem/s │         │
│     ├─ 5                     10.74 ms      │ 31.38 ms      │ 17.45 ms      │ 17.98 ms      │ 100     │ 100
│     │                        4.651 Mitem/s │ 1.593 Mitem/s │ 2.864 Mitem/s │ 2.779 Mitem/s │         │
│     ╰─ 6                     10.67 ms      │ 32.94 ms      │ 18.61 ms      │ 20.15 ms      │ 100     │ 100
│                              5.62 Mitem/s  │ 1.821 Mitem/s │ 3.223 Mitem/s │ 2.976 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.919 ms      │ 6.602 ms      │ 3.974 ms      │ 4.119 ms      │ 100     │ 100
│  │  │                        2.551 Mitem/s │ 1.514 Mitem/s │ 2.515 Mitem/s │ 2.427 Mitem/s │         │
│  │  ├─ 2                     8.696 ms      │ 17.15 ms      │ 14.26 ms      │ 13.7 ms       │ 100     │ 100
│  │  │                        2.299 Mitem/s │ 1.166 Mitem/s │ 1.401 Mitem/s │ 1.459 Mitem/s │         │
│  │  ├─ 3                     17.37 ms      │ 38.8 ms       │ 24.63 ms      │ 24.28 ms      │ 100     │ 100
│  │  │                        1.726 Mitem/s │ 773.1 Kitem/s │ 1.217 Mitem/s │ 1.235 Mitem/s │         │
│  │  ├─ 4                     30 ms         │ 50.97 ms      │ 38.47 ms      │ 39.37 ms      │ 100     │ 100
│  │  │                        1.333 Mitem/s │ 784.7 Kitem/s │ 1.039 Mitem/s │ 1.015 Mitem/s │         │
│  │  ├─ 5                     46.86 ms      │ 73.11 ms      │ 54.64 ms      │ 55.4 ms       │ 100     │ 100
│  │  │                        1.066 Mitem/s │ 683.8 Kitem/s │ 914.9 Kitem/s │ 902.4 Kitem/s │         │
│  │  ╰─ 6                     75.72 ms      │ 104.3 ms      │ 89.25 ms      │ 88.96 ms      │ 100     │ 100
│  │                           792.3 Kitem/s │ 575 Kitem/s   │ 672.2 Kitem/s │ 674.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.637 ms      │ 9.142 ms      │ 6.822 ms      │ 7.024 ms      │ 100     │ 100
│  │  │                        1.506 Mitem/s │ 1.093 Mitem/s │ 1.465 Mitem/s │ 1.423 Mitem/s │         │
│  │  ├─ 2                     7.056 ms      │ 14.83 ms      │ 12.49 ms      │ 12.07 ms      │ 100     │ 100
│  │  │                        2.834 Mitem/s │ 1.348 Mitem/s │ 1.6 Mitem/s   │ 1.656 Mitem/s │         │
│  │  ├─ 3                     7.023 ms      │ 15.47 ms      │ 12.52 ms      │ 11.77 ms      │ 100     │ 100
│  │  │                        4.271 Mitem/s │ 1.939 Mitem/s │ 2.394 Mitem/s │ 2.547 Mitem/s │         │
│  │  ├─ 4                     7.738 ms      │ 18.14 ms      │ 12.23 ms      │ 11.92 ms      │ 100     │ 100
│  │  │                        5.169 Mitem/s │ 2.205 Mitem/s │ 3.268 Mitem/s │ 3.355 Mitem/s │         │
│  │  ├─ 5                     8.01 ms       │ 21.74 ms      │ 12.4 ms       │ 12.82 ms      │ 100     │ 100
│  │  │                        6.241 Mitem/s │ 2.299 Mitem/s │ 4.029 Mitem/s │ 3.897 Mitem/s │         │
│  │  ╰─ 6                     7.768 ms      │ 21.48 ms      │ 12.46 ms      │ 12.23 ms      │ 100     │ 100
│  │                           7.723 Mitem/s │ 2.792 Mitem/s │ 4.812 Mitem/s │ 4.903 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.947 ms      │ 11.87 ms      │ 9.22 ms       │ 9.422 ms      │ 100     │ 100
│     │                        1.117 Mitem/s │ 842.3 Kitem/s │ 1.084 Mitem/s │ 1.061 Mitem/s │         │
│     ├─ 2                     9.09 ms       │ 19.27 ms      │ 12.93 ms      │ 13.56 ms      │ 100     │ 100
│     │                        2.2 Mitem/s   │ 1.037 Mitem/s │ 1.545 Mitem/s │ 1.473 Mitem/s │         │
│     ├─ 3                     9.68 ms       │ 21.03 ms      │ 15.41 ms      │ 15.75 ms      │ 100     │ 100
│     │                        3.098 Mitem/s │ 1.426 Mitem/s │ 1.945 Mitem/s │ 1.903 Mitem/s │         │
│     ├─ 4                     9.416 ms      │ 27.29 ms      │ 15.5 ms       │ 15.36 ms      │ 100     │ 100
│     │                        4.247 Mitem/s │ 1.465 Mitem/s │ 2.579 Mitem/s │ 2.603 Mitem/s │         │
│     ├─ 5                     9.439 ms      │ 30.57 ms      │ 16.38 ms      │ 16.79 ms      │ 100     │ 100
│     │                        5.296 Mitem/s │ 1.635 Mitem/s │ 3.051 Mitem/s │ 2.976 Mitem/s │         │
│     ╰─ 6                     9.646 ms      │ 27.05 ms      │ 16.29 ms      │ 16.42 ms      │ 100     │ 100
│                              6.219 Mitem/s │ 2.218 Mitem/s │ 3.68 Mitem/s  │ 3.654 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.622 ms      │ 6.387 ms      │ 3.71 ms       │ 3.853 ms      │ 100     │ 100
│  │  │                        2.76 Mitem/s  │ 1.565 Mitem/s │ 2.695 Mitem/s │ 2.594 Mitem/s │         │
│  │  ├─ 2                     8.619 ms      │ 15.57 ms      │ 11.7 ms       │ 11.84 ms      │ 100     │ 100
│  │  │                        2.32 Mitem/s  │ 1.283 Mitem/s │ 1.708 Mitem/s │ 1.689 Mitem/s │         │
│  │  ├─ 3                     17.69 ms      │ 30.41 ms      │ 24.28 ms      │ 24.32 ms      │ 100     │ 100
│  │  │                        1.695 Mitem/s │ 986.5 Kitem/s │ 1.235 Mitem/s │ 1.233 Mitem/s │         │
│  │  ├─ 4                     28.7 ms       │ 43.3 ms       │ 33.98 ms      │ 34.31 ms      │ 100     │ 100
│  │  │                        1.393 Mitem/s │ 923.6 Kitem/s │ 1.176 Mitem/s │ 1.165 Mitem/s │         │
│  │  ├─ 5                     39.96 ms      │ 59.35 ms      │ 49.29 ms      │ 49.62 ms      │ 100     │ 100
│  │  │                        1.251 Mitem/s │ 842.3 Kitem/s │ 1.014 Mitem/s │ 1.007 Mitem/s │         │
│  │  ╰─ 6                     62.49 ms      │ 101.5 ms      │ 76.53 ms      │ 77 ms         │ 100     │ 100
│  │                           960.1 Kitem/s │ 590.8 Kitem/s │ 784 Kitem/s   │ 779.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.633 ms      │ 13.84 ms      │ 9.904 ms      │ 10.68 ms      │ 100     │ 100
│  │  │                        1.038 Mitem/s │ 722.5 Kitem/s │ 1.009 Mitem/s │ 935.5 Kitem/s │         │
│  │  ├─ 2                     9.775 ms      │ 19.76 ms      │ 11.29 ms      │ 12.17 ms      │ 100     │ 100
│  │  │                        2.045 Mitem/s │ 1.011 Mitem/s │ 1.77 Mitem/s  │ 1.642 Mitem/s │         │
│  │  ├─ 3                     9.879 ms      │ 20.67 ms      │ 14.49 ms      │ 14.13 ms      │ 100     │ 100
│  │  │                        3.036 Mitem/s │ 1.451 Mitem/s │ 2.069 Mitem/s │ 2.122 Mitem/s │         │
│  │  ├─ 4                     9.919 ms      │ 27.57 ms      │ 15.34 ms      │ 15.93 ms      │ 100     │ 100
│  │  │                        4.032 Mitem/s │ 1.45 Mitem/s  │ 2.606 Mitem/s │ 2.509 Mitem/s │         │
│  │  ├─ 5                     10.27 ms      │ 32.5 ms       │ 16.26 ms      │ 17.38 ms      │ 100     │ 100
│  │  │                        4.868 Mitem/s │ 1.538 Mitem/s │ 3.073 Mitem/s │ 2.876 Mitem/s │         │
│  │  ╰─ 6                     10.14 ms      │ 32.38 ms      │ 18.66 ms      │ 20.24 ms      │ 100     │ 100
│  │                           5.916 Mitem/s │ 1.852 Mitem/s │ 3.214 Mitem/s │ 2.964 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     10.04 ms      │ 13.17 ms      │ 10.76 ms      │ 10.89 ms      │ 100     │ 100
│     │                        995.5 Kitem/s │ 758.9 Kitem/s │ 929.2 Kitem/s │ 917.9 Kitem/s │         │
│     ├─ 2                     10.39 ms      │ 21.07 ms      │ 15.27 ms      │ 14.78 ms      │ 100     │ 100
│     │                        1.923 Mitem/s │ 948.7 Kitem/s │ 1.309 Mitem/s │ 1.352 Mitem/s │         │
│     ├─ 3                     11.03 ms      │ 21.73 ms      │ 16.06 ms      │ 16.19 ms      │ 100     │ 100
│     │                        2.718 Mitem/s │ 1.38 Mitem/s  │ 1.867 Mitem/s │ 1.851 Mitem/s │         │
│     ├─ 4                     10.79 ms      │ 28.27 ms      │ 16.61 ms      │ 17.62 ms      │ 100     │ 100
│     │                        3.705 Mitem/s │ 1.414 Mitem/s │ 2.408 Mitem/s │ 2.269 Mitem/s │         │
│     ├─ 5                     10.88 ms      │ 30.94 ms      │ 16.51 ms      │ 17.58 ms      │ 100     │ 100
│     │                        4.591 Mitem/s │ 1.615 Mitem/s │ 3.026 Mitem/s │ 2.844 Mitem/s │         │
│     ╰─ 6                     11.38 ms      │ 33.5 ms       │ 19.17 ms      │ 20.92 ms      │ 100     │ 100
│                              5.27 Mitem/s  │ 1.79 Mitem/s  │ 3.129 Mitem/s │ 2.867 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.614 ms      │ 5.84 ms       │ 3.689 ms      │ 3.795 ms      │ 100     │ 100
│  │  │                        2.766 Mitem/s │ 1.712 Mitem/s │ 2.71 Mitem/s  │ 2.635 Mitem/s │         │
│  │  ├─ 2                     8.54 ms       │ 17.52 ms      │ 12.74 ms      │ 12.95 ms      │ 100     │ 100
│  │  │                        2.341 Mitem/s │ 1.141 Mitem/s │ 1.568 Mitem/s │ 1.543 Mitem/s │         │
│  │  ├─ 3                     17.72 ms      │ 28.51 ms      │ 23.87 ms      │ 23.39 ms      │ 100     │ 100
│  │  │                        1.692 Mitem/s │ 1.052 Mitem/s │ 1.256 Mitem/s │ 1.282 Mitem/s │         │
│  │  ├─ 4                     29.86 ms      │ 42.28 ms      │ 35.35 ms      │ 35.72 ms      │ 100     │ 100
│  │  │                        1.339 Mitem/s │ 946 Kitem/s   │ 1.131 Mitem/s │ 1.119 Mitem/s │         │
│  │  ├─ 5                     38.92 ms      │ 58.38 ms      │ 49.81 ms      │ 49.88 ms      │ 100     │ 100
│  │  │                        1.284 Mitem/s │ 856.3 Kitem/s │ 1.003 Mitem/s │ 1.002 Mitem/s │         │
│  │  ╰─ 6                     65.26 ms      │ 89.36 ms      │ 75.6 ms       │ 75.61 ms      │ 100     │ 100
│  │                           919.3 Kitem/s │ 671.4 Kitem/s │ 793.6 Kitem/s │ 793.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.077 ms      │ 10.13 ms      │ 7.199 ms      │ 7.476 ms      │ 100     │ 100
│  │  │                        1.412 Mitem/s │ 986.4 Kitem/s │ 1.388 Mitem/s │ 1.337 Mitem/s │         │
│  │  ├─ 2                     7.364 ms      │ 15.67 ms      │ 11.01 ms      │ 10.88 ms      │ 100     │ 100
│  │  │                        2.715 Mitem/s │ 1.275 Mitem/s │ 1.816 Mitem/s │ 1.836 Mitem/s │         │
│  │  ├─ 3                     7.894 ms      │ 16.09 ms      │ 12.12 ms      │ 12.25 ms      │ 100     │ 100
│  │  │                        3.799 Mitem/s │ 1.863 Mitem/s │ 2.473 Mitem/s │ 2.447 Mitem/s │         │
│  │  ├─ 4                     8.215 ms      │ 24.08 ms      │ 13.5 ms       │ 13.59 ms      │ 100     │ 100
│  │  │                        4.868 Mitem/s │ 1.661 Mitem/s │ 2.962 Mitem/s │ 2.941 Mitem/s │         │
│  │  ├─ 5                     9.203 ms      │ 22.63 ms      │ 13.42 ms      │ 13.67 ms      │ 100     │ 100
│  │  │                        5.432 Mitem/s │ 2.208 Mitem/s │ 3.724 Mitem/s │ 3.656 Mitem/s │         │
│  │  ╰─ 6                     8.412 ms      │ 23.17 ms      │ 13.58 ms      │ 14.03 ms      │ 100     │ 100
│  │                           7.132 Mitem/s │ 2.589 Mitem/s │ 4.416 Mitem/s │ 4.273 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.73 ms       │ 15.15 ms      │ 10.08 ms      │ 10.46 ms      │ 100     │ 100
│     │                        1.027 Mitem/s │ 659.8 Kitem/s │ 991.1 Kitem/s │ 955.2 Kitem/s │         │
│     ├─ 2                     9.894 ms      │ 30.69 ms      │ 11.94 ms      │ 12.75 ms      │ 100     │ 100
│     │                        2.021 Mitem/s │ 651.6 Kitem/s │ 1.675 Mitem/s │ 1.567 Mitem/s │         │
│     ├─ 3                     10.22 ms      │ 21.13 ms      │ 17.52 ms      │ 16.71 ms      │ 100     │ 100
│     │                        2.933 Mitem/s │ 1.419 Mitem/s │ 1.711 Mitem/s │ 1.794 Mitem/s │         │
│     ├─ 4                     10.23 ms      │ 37.15 ms      │ 15.46 ms      │ 16.52 ms      │ 100     │ 100
│     │                        3.908 Mitem/s │ 1.076 Mitem/s │ 2.587 Mitem/s │ 2.42 Mitem/s  │         │
│     ├─ 5                     10.65 ms      │ 35.29 ms      │ 18.38 ms      │ 19.79 ms      │ 100     │ 100
│     │                        4.694 Mitem/s │ 1.416 Mitem/s │ 2.719 Mitem/s │ 2.525 Mitem/s │         │
│     ╰─ 6                     10.26 ms      │ 39.88 ms      │ 21.01 ms      │ 22.74 ms      │ 100     │ 100
│                              5.847 Mitem/s │ 1.504 Mitem/s │ 2.854 Mitem/s │ 2.637 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.536 ms      │ 8.324 ms      │ 4.704 ms      │ 5.228 ms      │ 100     │ 100
│  │  │                        2.204 Mitem/s │ 1.201 Mitem/s │ 2.125 Mitem/s │ 1.912 Mitem/s │         │
│  │  ├─ 2                     11.95 ms      │ 20.37 ms      │ 16.29 ms      │ 15.71 ms      │ 100     │ 100
│  │  │                        1.673 Mitem/s │ 981.3 Kitem/s │ 1.227 Mitem/s │ 1.272 Mitem/s │         │
│  │  ├─ 3                     21.59 ms      │ 41.15 ms      │ 29.29 ms      │ 29.41 ms      │ 100     │ 100
│  │  │                        1.389 Mitem/s │ 728.9 Kitem/s │ 1.024 Mitem/s │ 1.019 Mitem/s │         │
│  │  ├─ 4                     33.42 ms      │ 55.09 ms      │ 40.09 ms      │ 40.15 ms      │ 100     │ 100
│  │  │                        1.196 Mitem/s │ 725.9 Kitem/s │ 997.5 Kitem/s │ 996.1 Kitem/s │         │
│  │  ├─ 5                     53.23 ms      │ 84.55 ms      │ 62.64 ms      │ 63.6 ms       │ 100     │ 100
│  │  │                        939.2 Kitem/s │ 591.3 Kitem/s │ 798.1 Kitem/s │ 786.1 Kitem/s │         │
│  │  ╰─ 6                     85.33 ms      │ 114.7 ms      │ 99.93 ms      │ 99.74 ms      │ 100     │ 100
│  │                           703 Kitem/s   │ 522.9 Kitem/s │ 600.3 Kitem/s │ 601.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.583 ms      │ 8.849 ms      │ 6.705 ms      │ 6.879 ms      │ 100     │ 100
│  │  │                        1.519 Mitem/s │ 1.129 Mitem/s │ 1.491 Mitem/s │ 1.453 Mitem/s │         │
│  │  ├─ 2                     6.972 ms      │ 14.36 ms      │ 11.26 ms      │ 10.7 ms       │ 100     │ 100
│  │  │                        2.868 Mitem/s │ 1.392 Mitem/s │ 1.775 Mitem/s │ 1.867 Mitem/s │         │
│  │  ├─ 3                     7.079 ms      │ 14.64 ms      │ 9.574 ms      │ 9.896 ms      │ 100     │ 100
│  │  │                        4.237 Mitem/s │ 2.047 Mitem/s │ 3.133 Mitem/s │ 3.031 Mitem/s │         │
│  │  ├─ 4                     7.731 ms      │ 22.09 ms      │ 12.37 ms      │ 12.13 ms      │ 100     │ 100
│  │  │                        5.173 Mitem/s │ 1.81 Mitem/s  │ 3.232 Mitem/s │ 3.297 Mitem/s │         │
│  │  ├─ 5                     7.808 ms      │ 23.24 ms      │ 12.04 ms      │ 12.09 ms      │ 100     │ 100
│  │  │                        6.403 Mitem/s │ 2.15 Mitem/s  │ 4.15 Mitem/s  │ 4.132 Mitem/s │         │
│  │  ╰─ 6                     8.741 ms      │ 22.72 ms      │ 12.59 ms      │ 12.91 ms      │ 100     │ 100
│  │                           6.864 Mitem/s │ 2.639 Mitem/s │ 4.763 Mitem/s │ 4.646 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.557 ms      │ 14.13 ms      │ 10.07 ms      │ 10.5 ms       │ 100     │ 100
│     │                        1.046 Mitem/s │ 707.3 Kitem/s │ 992.7 Kitem/s │ 951.8 Kitem/s │         │
│     ├─ 2                     9.821 ms      │ 20.24 ms      │ 14.12 ms      │ 13.7 ms       │ 100     │ 100
│     │                        2.036 Mitem/s │ 987.8 Kitem/s │ 1.416 Mitem/s │ 1.459 Mitem/s │         │
│     ├─ 3                     10.1 ms       │ 20.93 ms      │ 14.8 ms       │ 14.7 ms       │ 100     │ 100
│     │                        2.969 Mitem/s │ 1.433 Mitem/s │ 2.026 Mitem/s │ 2.04 Mitem/s  │         │
│     ├─ 4                     10.12 ms      │ 29.46 ms      │ 15.35 ms      │ 16.02 ms      │ 100     │ 100
│     │                        3.949 Mitem/s │ 1.357 Mitem/s │ 2.604 Mitem/s │ 2.495 Mitem/s │         │
│     ├─ 5                     10.13 ms      │ 28.29 ms      │ 16.87 ms      │ 17.1 ms       │ 100     │ 100
│     │                        4.931 Mitem/s │ 1.766 Mitem/s │ 2.963 Mitem/s │ 2.923 Mitem/s │         │
│     ╰─ 6                     10.23 ms      │ 31.68 ms      │ 17.26 ms      │ 18.29 ms      │ 100     │ 100
│                              5.863 Mitem/s │ 1.893 Mitem/s │ 3.475 Mitem/s │ 3.279 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.607 ms      │ 6.884 ms      │ 4.698 ms      │ 4.882 ms      │ 100     │ 100
│  │  │                        2.17 Mitem/s  │ 1.452 Mitem/s │ 2.128 Mitem/s │ 2.048 Mitem/s │         │
│  │  ├─ 2                     10.08 ms      │ 20.36 ms      │ 16.37 ms      │ 15.91 ms      │ 100     │ 100
│  │  │                        1.982 Mitem/s │ 982 Kitem/s   │ 1.221 Mitem/s │ 1.256 Mitem/s │         │
│  │  ├─ 3                     22.27 ms      │ 36.95 ms      │ 28.5 ms       │ 28.88 ms      │ 100     │ 100
│  │  │                        1.346 Mitem/s │ 811.8 Kitem/s │ 1.052 Mitem/s │ 1.038 Mitem/s │         │
│  │  ├─ 4                     35.82 ms      │ 58.51 ms      │ 44.03 ms      │ 44.45 ms      │ 100     │ 100
│  │  │                        1.116 Mitem/s │ 683.6 Kitem/s │ 908.3 Kitem/s │ 899.8 Kitem/s │         │
│  │  ├─ 5                     52.78 ms      │ 77.42 ms      │ 62.84 ms      │ 63.45 ms      │ 100     │ 100
│  │  │                        947.1 Kitem/s │ 645.7 Kitem/s │ 795.5 Kitem/s │ 788 Kitem/s   │         │
│  │  ╰─ 6                     86.4 ms       │ 115.3 ms      │ 99.95 ms      │ 98.96 ms      │ 100     │ 100
│  │                           694.3 Kitem/s │ 520 Kitem/s   │ 600.2 Kitem/s │ 606.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.61 ms       │ 9.823 ms      │ 6.864 ms      │ 7.26 ms       │ 100     │ 100
│  │  │                        1.512 Mitem/s │ 1.018 Mitem/s │ 1.456 Mitem/s │ 1.377 Mitem/s │         │
│  │  ├─ 2                     7.018 ms      │ 15.05 ms      │ 9.316 ms      │ 10.23 ms      │ 100     │ 100
│  │  │                        2.849 Mitem/s │ 1.328 Mitem/s │ 2.146 Mitem/s │ 1.953 Mitem/s │         │
│  │  ├─ 3                     7.386 ms      │ 14.75 ms      │ 12.03 ms      │ 11.37 ms      │ 100     │ 100
│  │  │                        4.061 Mitem/s │ 2.033 Mitem/s │ 2.492 Mitem/s │ 2.636 Mitem/s │         │
│  │  ├─ 4                     7.884 ms      │ 21.21 ms      │ 12.36 ms      │ 12.06 ms      │ 100     │ 100
│  │  │                        5.073 Mitem/s │ 1.885 Mitem/s │ 3.234 Mitem/s │ 3.316 Mitem/s │         │
│  │  ├─ 5                     7.825 ms      │ 22.57 ms      │ 12.75 ms      │ 13.05 ms      │ 100     │ 100
│  │  │                        6.389 Mitem/s │ 2.215 Mitem/s │ 3.919 Mitem/s │ 3.829 Mitem/s │         │
│  │  ╰─ 6                     7.816 ms      │ 19.55 ms      │ 13.06 ms      │ 13.32 ms      │ 100     │ 100
│  │                           7.676 Mitem/s │ 3.068 Mitem/s │ 4.592 Mitem/s │ 4.502 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.513 ms      │ 13.74 ms      │ 9.68 ms       │ 9.907 ms      │ 100     │ 100
│     │                        1.051 Mitem/s │ 727.6 Kitem/s │ 1.033 Mitem/s │ 1.009 Mitem/s │         │
│     ├─ 2                     9.651 ms      │ 17.47 ms      │ 10.17 ms      │ 11.01 ms      │ 100     │ 100
│     │                        2.072 Mitem/s │ 1.144 Mitem/s │ 1.965 Mitem/s │ 1.816 Mitem/s │         │
│     ├─ 3                     9.775 ms      │ 22.6 ms       │ 13.84 ms      │ 13.93 ms      │ 100     │ 100
│     │                        3.068 Mitem/s │ 1.327 Mitem/s │ 2.166 Mitem/s │ 2.152 Mitem/s │         │
│     ├─ 4                     10.18 ms      │ 29.31 ms      │ 16.77 ms      │ 16.62 ms      │ 100     │ 100
│     │                        3.925 Mitem/s │ 1.364 Mitem/s │ 2.385 Mitem/s │ 2.405 Mitem/s │         │
│     ├─ 5                     10.2 ms       │ 31.24 ms      │ 18.43 ms      │ 18.72 ms      │ 100     │ 100
│     │                        4.9 Mitem/s   │ 1.6 Mitem/s   │ 2.712 Mitem/s │ 2.669 Mitem/s │         │
│     ╰─ 6                     10.08 ms      │ 32.34 ms      │ 18.73 ms      │ 20.36 ms      │ 100     │ 100
│                              5.948 Mitem/s │ 1.854 Mitem/s │ 3.202 Mitem/s │ 2.945 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.526 ms      │ 9.037 ms      │ 4.641 ms      │ 5.37 ms       │ 100     │ 100
│  │  │                        2.209 Mitem/s │ 1.106 Mitem/s │ 2.154 Mitem/s │ 1.861 Mitem/s │         │
│  │  ├─ 2                     10.73 ms      │ 20.57 ms      │ 16.13 ms      │ 15.86 ms      │ 100     │ 100
│  │  │                        1.863 Mitem/s │ 972.1 Kitem/s │ 1.239 Mitem/s │ 1.26 Mitem/s  │         │
│  │  ├─ 3                     22.76 ms      │ 39.01 ms      │ 28.37 ms      │ 28.7 ms       │ 100     │ 100
│  │  │                        1.317 Mitem/s │ 769 Kitem/s   │ 1.057 Mitem/s │ 1.045 Mitem/s │         │
│  │  ├─ 4                     34.85 ms      │ 59.45 ms      │ 42.77 ms      │ 43.4 ms       │ 100     │ 100
│  │  │                        1.147 Mitem/s │ 672.8 Kitem/s │ 935.1 Kitem/s │ 921.5 Kitem/s │         │
│  │  ├─ 5                     51.4 ms       │ 79.97 ms      │ 61.75 ms      │ 62.69 ms      │ 100     │ 100
│  │  │                        972.6 Kitem/s │ 625.2 Kitem/s │ 809.6 Kitem/s │ 797.5 Kitem/s │         │
│  │  ╰─ 6                     82.41 ms      │ 112.6 ms      │ 97.96 ms      │ 97.84 ms      │ 100     │ 100
│  │                           728 Kitem/s   │ 532.5 Kitem/s │ 612.4 Kitem/s │ 613.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.588 ms      │ 9.249 ms      │ 6.691 ms      │ 6.987 ms      │ 100     │ 100
│  │  │                        1.517 Mitem/s │ 1.081 Mitem/s │ 1.494 Mitem/s │ 1.431 Mitem/s │         │
│  │  ├─ 2                     7.586 ms      │ 14.54 ms      │ 12.37 ms      │ 11.36 ms      │ 100     │ 100
│  │  │                        2.636 Mitem/s │ 1.375 Mitem/s │ 1.615 Mitem/s │ 1.759 Mitem/s │         │
│  │  ├─ 3                     7.72 ms       │ 15.25 ms      │ 12.53 ms      │ 12.04 ms      │ 100     │ 100
│  │  │                        3.885 Mitem/s │ 1.965 Mitem/s │ 2.393 Mitem/s │ 2.49 Mitem/s  │         │
│  │  ├─ 4                     7.85 ms       │ 18.21 ms      │ 12.57 ms      │ 12.35 ms      │ 100     │ 100
│  │  │                        5.095 Mitem/s │ 2.196 Mitem/s │ 3.181 Mitem/s │ 3.237 Mitem/s │         │
│  │  ├─ 5                     7.849 ms      │ 21.97 ms      │ 12.93 ms      │ 13.26 ms      │ 100     │ 100
│  │  │                        6.37 Mitem/s  │ 2.274 Mitem/s │ 3.865 Mitem/s │ 3.769 Mitem/s │         │
│  │  ╰─ 6                     8.805 ms      │ 22.21 ms      │ 13.46 ms      │ 15.36 ms      │ 100     │ 100
│  │                           6.813 Mitem/s │ 2.701 Mitem/s │ 4.454 Mitem/s │ 3.904 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.472 ms      │ 12.07 ms      │ 9.763 ms      │ 9.914 ms      │ 100     │ 100
│     │                        1.055 Mitem/s │ 827.8 Kitem/s │ 1.024 Mitem/s │ 1.008 Mitem/s │         │
│     ├─ 2                     9.774 ms      │ 21.01 ms      │ 13.47 ms      │ 13.69 ms      │ 100     │ 100
│     │                        2.046 Mitem/s │ 951.7 Kitem/s │ 1.484 Mitem/s │ 1.46 Mitem/s  │         │
│     ├─ 3                     10.04 ms      │ 20.7 ms       │ 15.78 ms      │ 16.22 ms      │ 100     │ 100
│     │                        2.986 Mitem/s │ 1.449 Mitem/s │ 1.9 Mitem/s   │ 1.848 Mitem/s │         │
│     ├─ 4                     9.92 ms       │ 30.8 ms       │ 16.37 ms      │ 16.57 ms      │ 100     │ 100
│     │                        4.032 Mitem/s │ 1.298 Mitem/s │ 2.443 Mitem/s │ 2.412 Mitem/s │         │
│     ├─ 5                     9.912 ms      │ 31.62 ms      │ 18.47 ms      │ 18.66 ms      │ 100     │ 100
│     │                        5.044 Mitem/s │ 1.581 Mitem/s │ 2.706 Mitem/s │ 2.679 Mitem/s │         │
│     ╰─ 6                     10.49 ms      │ 30.99 ms      │ 18.6 ms       │ 19.78 ms      │ 100     │ 100
│                              5.717 Mitem/s │ 1.936 Mitem/s │ 3.224 Mitem/s │ 3.033 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.471 ms      │ 8.846 ms      │ 4.646 ms      │ 5.061 ms      │ 100     │ 100
│  │  │                        2.236 Mitem/s │ 1.13 Mitem/s  │ 2.152 Mitem/s │ 1.975 Mitem/s │         │
│  │  ├─ 2                     11.03 ms      │ 20.39 ms      │ 15.67 ms      │ 15.69 ms      │ 100     │ 100
│  │  │                        1.812 Mitem/s │ 980.6 Kitem/s │ 1.275 Mitem/s │ 1.274 Mitem/s │         │
│  │  ├─ 3                     20.13 ms      │ 33.65 ms      │ 27.58 ms      │ 27.67 ms      │ 100     │ 100
│  │  │                        1.49 Mitem/s  │ 891.3 Kitem/s │ 1.087 Mitem/s │ 1.083 Mitem/s │         │
│  │  ├─ 4                     35.05 ms      │ 60.92 ms      │ 41.71 ms      │ 43.08 ms      │ 100     │ 100
│  │  │                        1.141 Mitem/s │ 656.5 Kitem/s │ 958.8 Kitem/s │ 928.3 Kitem/s │         │
│  │  ├─ 5                     51.15 ms      │ 80.85 ms      │ 59.13 ms      │ 59.7 ms       │ 100     │ 100
│  │  │                        977.3 Kitem/s │ 618.3 Kitem/s │ 845.5 Kitem/s │ 837.3 Kitem/s │         │
│  │  ╰─ 6                     79.04 ms      │ 110.7 ms      │ 96.77 ms      │ 96.73 ms      │ 100     │ 100
│  │                           759 Kitem/s   │ 541.5 Kitem/s │ 620 Kitem/s   │ 620.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.608 ms      │ 11.32 ms      │ 6.691 ms      │ 6.925 ms      │ 100     │ 100
│  │  │                        1.513 Mitem/s │ 882.8 Kitem/s │ 1.494 Mitem/s │ 1.444 Mitem/s │         │
│  │  ├─ 2                     6.965 ms      │ 17.86 ms      │ 12.49 ms      │ 11.9 ms       │ 100     │ 100
│  │  │                        2.871 Mitem/s │ 1.119 Mitem/s │ 1.6 Mitem/s   │ 1.68 Mitem/s  │         │
│  │  ├─ 3                     7.64 ms       │ 14.9 ms       │ 12.82 ms      │ 12.5 ms       │ 100     │ 100
│  │  │                        3.926 Mitem/s │ 2.012 Mitem/s │ 2.34 Mitem/s  │ 2.398 Mitem/s │         │
│  │  ├─ 4                     7.714 ms      │ 15.29 ms      │ 12.16 ms      │ 11.34 ms      │ 100     │ 100
│  │  │                        5.185 Mitem/s │ 2.614 Mitem/s │ 3.286 Mitem/s │ 3.525 Mitem/s │         │
│  │  ├─ 5                     7.715 ms      │ 22.15 ms      │ 12.75 ms      │ 13.35 ms      │ 100     │ 100
│  │  │                        6.48 Mitem/s  │ 2.256 Mitem/s │ 3.92 Mitem/s  │ 3.744 Mitem/s │         │
│  │  ╰─ 6                     7.829 ms      │ 21.73 ms      │ 12.97 ms      │ 13.6 ms       │ 100     │ 100
│  │                           7.663 Mitem/s │ 2.76 Mitem/s  │ 4.622 Mitem/s │ 4.411 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.453 ms      │ 13.55 ms      │ 9.686 ms      │ 9.875 ms      │ 100     │ 100
│     │                        1.057 Mitem/s │ 737.6 Kitem/s │ 1.032 Mitem/s │ 1.012 Mitem/s │         │
│     ├─ 2                     9.702 ms      │ 21 ms         │ 12.77 ms      │ 12.74 ms      │ 100     │ 100
│     │                        2.061 Mitem/s │ 952.3 Kitem/s │ 1.565 Mitem/s │ 1.569 Mitem/s │         │
│     ├─ 3                     9.9 ms        │ 21.71 ms      │ 14.83 ms      │ 15.34 ms      │ 100     │ 100
│     │                        3.03 Mitem/s  │ 1.381 Mitem/s │ 2.021 Mitem/s │ 1.954 Mitem/s │         │
│     ├─ 4                     9.92 ms       │ 31.9 ms       │ 17.01 ms      │ 16.73 ms      │ 100     │ 100
│     │                        4.032 Mitem/s │ 1.253 Mitem/s │ 2.35 Mitem/s  │ 2.389 Mitem/s │         │
│     ├─ 5                     9.918 ms      │ 29.75 ms      │ 17.1 ms       │ 17.45 ms      │ 100     │ 100
│     │                        5.041 Mitem/s │ 1.68 Mitem/s  │ 2.923 Mitem/s │ 2.864 Mitem/s │         │
│     ╰─ 6                     10.15 ms      │ 35.09 ms      │ 18.68 ms      │ 19.79 ms      │ 100     │ 100
│                              5.906 Mitem/s │ 1.709 Mitem/s │ 3.211 Mitem/s │ 3.031 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     5.17 ms       │ 7.286 ms      │ 5.288 ms      │ 5.406 ms      │ 100     │ 100
│  │  │                        1.933 Mitem/s │ 1.372 Mitem/s │ 1.89 Mitem/s  │ 1.849 Mitem/s │         │
│  │  ├─ 2                     12.77 ms      │ 25.15 ms      │ 17.06 ms      │ 17.06 ms      │ 100     │ 100
│  │  │                        1.565 Mitem/s │ 795 Kitem/s   │ 1.171 Mitem/s │ 1.172 Mitem/s │         │
│  │  ├─ 3                     25.08 ms      │ 41.22 ms      │ 33.6 ms       │ 33.4 ms       │ 100     │ 100
│  │  │                        1.195 Mitem/s │ 727.7 Kitem/s │ 892.8 Kitem/s │ 898.1 Kitem/s │         │
│  │  ├─ 4                     38.04 ms      │ 62.34 ms      │ 47.71 ms      │ 48.08 ms      │ 100     │ 100
│  │  │                        1.051 Mitem/s │ 641.5 Kitem/s │ 838.2 Kitem/s │ 831.8 Kitem/s │         │
│  │  ├─ 5                     57.74 ms      │ 88.31 ms      │ 66.47 ms      │ 67.53 ms      │ 100     │ 100
│  │  │                        865.8 Kitem/s │ 566.1 Kitem/s │ 752.1 Kitem/s │ 740.3 Kitem/s │         │
│  │  ╰─ 6                     87.9 ms       │ 128.4 ms      │ 104.7 ms      │ 104.3 ms      │ 100     │ 100
│  │                           682.5 Kitem/s │ 467 Kitem/s   │ 572.9 Kitem/s │ 575 Kitem/s   │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.687 ms      │ 12.61 ms      │ 6.969 ms      │ 7.582 ms      │ 100     │ 100
│  │  │                        1.495 Mitem/s │ 792.5 Kitem/s │ 1.434 Mitem/s │ 1.318 Mitem/s │         │
│  │  ├─ 2                     6.994 ms      │ 14.29 ms      │ 8.094 ms      │ 8.954 ms      │ 100     │ 100
│  │  │                        2.859 Mitem/s │ 1.398 Mitem/s │ 2.47 Mitem/s  │ 2.233 Mitem/s │         │
│  │  ├─ 3                     7.725 ms      │ 18.86 ms      │ 12.48 ms      │ 12.37 ms      │ 100     │ 100
│  │  │                        3.883 Mitem/s │ 1.59 Mitem/s  │ 2.402 Mitem/s │ 2.423 Mitem/s │         │
│  │  ├─ 4                     7.636 ms      │ 21.75 ms      │ 12.2 ms       │ 11.53 ms      │ 100     │ 100
│  │  │                        5.237 Mitem/s │ 1.838 Mitem/s │ 3.277 Mitem/s │ 3.468 Mitem/s │         │
│  │  ├─ 5                     7.734 ms      │ 20.66 ms      │ 12.39 ms      │ 11.99 ms      │ 100     │ 100
│  │  │                        6.464 Mitem/s │ 2.419 Mitem/s │ 4.032 Mitem/s │ 4.167 Mitem/s │         │
│  │  ╰─ 6                     7.764 ms      │ 25.67 ms      │ 13.04 ms      │ 13.29 ms      │ 100     │ 100
│  │                           7.727 Mitem/s │ 2.336 Mitem/s │ 4.6 Mitem/s   │ 4.514 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.635 ms      │ 12.18 ms      │ 9.895 ms      │ 10.06 ms      │ 100     │ 100
│     │                        1.037 Mitem/s │ 820.4 Kitem/s │ 1.01 Mitem/s  │ 993.2 Kitem/s │         │
│     ├─ 2                     9.918 ms      │ 20.69 ms      │ 11.16 ms      │ 11.97 ms      │ 100     │ 100
│     │                        2.016 Mitem/s │ 966.3 Kitem/s │ 1.791 Mitem/s │ 1.67 Mitem/s  │         │
│     ├─ 3                     10.22 ms      │ 22.2 ms       │ 15.22 ms      │ 14.6 ms       │ 100     │ 100
│     │                        2.933 Mitem/s │ 1.351 Mitem/s │ 1.971 Mitem/s │ 2.053 Mitem/s │         │
│     ├─ 4                     10.29 ms      │ 31.6 ms       │ 15.89 ms      │ 16.14 ms      │ 100     │ 100
│     │                        3.885 Mitem/s │ 1.265 Mitem/s │ 2.515 Mitem/s │ 2.477 Mitem/s │         │
│     ├─ 5                     10.66 ms      │ 32.93 ms      │ 19.13 ms      │ 19.4 ms       │ 100     │ 100
│     │                        4.688 Mitem/s │ 1.518 Mitem/s │ 2.612 Mitem/s │ 2.576 Mitem/s │         │
│     ╰─ 6                     10.28 ms      │ 32.13 ms      │ 19.26 ms      │ 20.16 ms      │ 100     │ 100
│                              5.835 Mitem/s │ 1.867 Mitem/s │ 3.114 Mitem/s │ 2.974 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 3                     8.818 ms      │ 15.99 ms      │ 12.69 ms      │ 12.7 ms       │ 100     │ 100
│  │  ├─ 4                     9.948 ms      │ 17.7 ms       │ 13.76 ms      │ 13.78 ms      │ 100     │ 100
│  │  ├─ 5                     12.21 ms      │ 24.43 ms      │ 14.16 ms      │ 14.42 ms      │ 100     │ 100
│  │  ╰─ 6                     13.24 ms      │ 22.42 ms      │ 14.66 ms      │ 14.88 ms      │ 100     │ 100
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 3                     12.83 ms      │ 24.38 ms      │ 14.62 ms      │ 14.98 ms      │ 100     │ 100
│     ├─ 4                     12.75 ms      │ 24.26 ms      │ 19.37 ms      │ 18.44 ms      │ 100     │ 100
│     ├─ 5                     13.24 ms      │ 27.7 ms       │ 18.25 ms      │ 17.69 ms      │ 100     │ 100
│     ╰─ 6                     14.42 ms      │ 31.91 ms      │ 19.43 ms      │ 19.84 ms      │ 100     │ 100
├─ 14_prefix_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     22.8 ms       │ 25.85 ms      │ 23.21 ms      │ 23.47 ms      │ 20      │ 20
│     │                        4.385 Kitem/s │ 3.868 Kitem/s │ 4.308 Kitem/s │ 4.26 Kitem/s  │         │
│     ├─ 2                     24.01 ms      │ 39.88 ms      │ 39.06 ms      │ 37.32 ms      │ 20      │ 20
│     │                        8.329 Kitem/s │ 5.014 Kitem/s │ 5.119 Kitem/s │ 5.358 Kitem/s │         │
│     ├─ 3                     23.21 ms      │ 43.07 ms      │ 34.81 ms      │ 35.95 ms      │ 20      │ 20
│     │                        12.92 Kitem/s │ 6.964 Kitem/s │ 8.615 Kitem/s │ 8.344 Kitem/s │         │
│     ├─ 4                     32.32 ms      │ 42.59 ms      │ 33.81 ms      │ 34.96 ms      │ 20      │ 20
│     │                        12.37 Kitem/s │ 9.391 Kitem/s │ 11.82 Kitem/s │ 11.44 Kitem/s │         │
│     ├─ 5                     33.85 ms      │ 45.78 ms      │ 39.93 ms      │ 39.35 ms      │ 20      │ 20
│     │                        14.77 Kitem/s │ 10.91 Kitem/s │ 12.52 Kitem/s │ 12.7 Kitem/s  │         │
│     ╰─ 6                     35.07 ms      │ 53.98 ms      │ 42.55 ms      │ 44.57 ms      │ 20      │ 20
│                              17.1 Kitem/s  │ 11.11 Kitem/s │ 14.09 Kitem/s │ 13.46 Kitem/s │         │
├─ 15_full_scan_aggregate                    │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     14.99 ms      │ 18.28 ms      │ 15.38 ms      │ 15.66 ms      │ 100     │ 100
│  │  │                        6.669 Kitem/s │ 5.468 Kitem/s │ 6.499 Kitem/s │ 6.383 Kitem/s │         │
│  │  ├─ 2                     34.03 ms      │ 55.84 ms      │ 49.74 ms      │ 48.14 ms      │ 100     │ 100
│  │  │                        5.876 Kitem/s │ 3.581 Kitem/s │ 4.02 Kitem/s  │ 4.153 Kitem/s │         │
│  │  ├─ 3                     50.78 ms      │ 87.93 ms      │ 76.3 ms       │ 74.43 ms      │ 100     │ 100
│  │  │                        5.906 Kitem/s │ 3.411 Kitem/s │ 3.931 Kitem/s │ 4.03 Kitem/s  │         │
│  │  ├─ 4                     66.3 ms       │ 114.1 ms      │ 91.78 ms      │ 91.46 ms      │ 100     │ 100
│  │  │                        6.032 Kitem/s │ 3.504 Kitem/s │ 4.358 Kitem/s │ 4.373 Kitem/s │         │
│  │  ├─ 5                     91.5 ms       │ 142.7 ms      │ 126.8 ms      │ 124.5 ms      │ 100     │ 100
│  │  │                        5.464 Kitem/s │ 3.502 Kitem/s │ 3.94 Kitem/s  │ 4.015 Kitem/s │         │
│  │  ╰─ 6                     102.8 ms      │ 166.5 ms      │ 153.9 ms      │ 150.4 ms      │ 100     │ 100
│  │                           5.83 Kitem/s  │ 3.603 Kitem/s │ 3.897 Kitem/s │ 3.987 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     33.48 ms      │ 48.83 ms      │ 35.56 ms      │ 35.69 ms      │ 100     │ 100
│  │  │                        2.986 Kitem/s │ 2.047 Kitem/s │ 2.811 Kitem/s │ 2.801 Kitem/s │         │
│  │  ├─ 2                     34.39 ms      │ 59.72 ms      │ 36.38 ms      │ 37.15 ms      │ 100     │ 100
│  │  │                        5.814 Kitem/s │ 3.348 Kitem/s │ 5.497 Kitem/s │ 5.383 Kitem/s │         │
│  │  ├─ 3                     35.32 ms      │ 59.16 ms      │ 37.37 ms      │ 41.69 ms      │ 100     │ 100
│  │  │                        8.493 Kitem/s │ 5.07 Kitem/s  │ 8.027 Kitem/s │ 7.194 Kitem/s │         │
│  │  ├─ 4                     35.16 ms      │ 61.35 ms      │ 40.28 ms      │ 43.22 ms      │ 100     │ 100
│  │  │                        11.37 Kitem/s │ 6.519 Kitem/s │ 9.929 Kitem/s │ 9.253 Kitem/s │         │
│  │  ├─ 5                     35.88 ms      │ 80.71 ms      │ 55.43 ms      │ 55.27 ms      │ 100     │ 100
│  │  │                        13.93 Kitem/s │ 6.194 Kitem/s │ 9.019 Kitem/s │ 9.045 Kitem/s │         │
│  │  ╰─ 6                     39.4 ms       │ 82.01 ms      │ 60.51 ms      │ 62.33 ms      │ 100     │ 100
│  │                           15.22 Kitem/s │ 7.316 Kitem/s │ 9.914 Kitem/s │ 9.626 Kitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     48.68 ms      │ 53.53 ms      │ 49.45 ms      │ 49.73 ms      │ 100     │ 100
│     │                        2.053 Kitem/s │ 1.867 Kitem/s │ 2.022 Kitem/s │ 2.01 Kitem/s  │         │
│     ├─ 2                     49.43 ms      │ 74.47 ms      │ 51.35 ms      │ 51.69 ms      │ 100     │ 100
│     │                        4.046 Kitem/s │ 2.685 Kitem/s │ 3.894 Kitem/s │ 3.868 Kitem/s │         │
│     ├─ 3                     50.21 ms      │ 84.22 ms      │ 61.66 ms      │ 61.69 ms      │ 100     │ 100
│     │                        5.974 Kitem/s │ 3.561 Kitem/s │ 4.864 Kitem/s │ 4.862 Kitem/s │         │
│     ├─ 4                     50.28 ms      │ 99.34 ms      │ 62.51 ms      │ 63.91 ms      │ 100     │ 100
│     │                        7.954 Kitem/s │ 4.026 Kitem/s │ 6.398 Kitem/s │ 6.258 Kitem/s │         │
│     ├─ 5                     51.35 ms      │ 103.2 ms      │ 78.42 ms      │ 76.13 ms      │ 100     │ 100
│     │                        9.736 Kitem/s │ 4.843 Kitem/s │ 6.375 Kitem/s │ 6.567 Kitem/s │         │
│     ╰─ 6                     52.93 ms      │ 114.9 ms      │ 91.39 ms      │ 89.51 ms      │ 100     │ 100
│                              11.33 Kitem/s │ 5.218 Kitem/s │ 6.565 Kitem/s │ 6.702 Kitem/s │         │
├─ 16_insert_heavy                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.277 ms      │ 7.091 ms      │ 5.378 ms      │ 5.491 ms      │ 100     │ 100
│  │  │                        2.337 Mitem/s │ 1.41 Mitem/s  │ 1.859 Mitem/s │ 1.82 Mitem/s  │         │
│  │  ├─ 2                     6.429 ms      │ 12.69 ms      │ 10.24 ms      │ 10.04 ms      │ 100     │ 100
│  │  │                        3.11 Mitem/s  │ 1.575 Mitem/s │ 1.952 Mitem/s │ 1.99 Mitem/s  │         │
│  │  ├─ 3                     8.912 ms      │ 17.71 ms      │ 12.86 ms      │ 12.72 ms      │ 100     │ 100
│  │  │                        3.365 Mitem/s │ 1.693 Mitem/s │ 2.332 Mitem/s │ 2.356 Mitem/s │         │
│  │  ├─ 4                     9.591 ms      │ 25.01 ms      │ 15.27 ms      │ 15.3 ms       │ 100     │ 100
│  │  │                        4.17 Mitem/s  │ 1.598 Mitem/s │ 2.618 Mitem/s │ 2.614 Mitem/s │         │
│  │  ├─ 5                     11.8 ms       │ 28.04 ms      │ 17.8 ms       │ 18.35 ms      │ 100     │ 100
│  │  │                        4.236 Mitem/s │ 1.782 Mitem/s │ 2.807 Mitem/s │ 2.723 Mitem/s │         │
│  │  ╰─ 6                     12.18 ms      │ 28.75 ms      │ 18.41 ms      │ 18.94 ms      │ 100     │ 100
│  │                           4.924 Mitem/s │ 2.086 Mitem/s │ 3.258 Mitem/s │ 3.166 Mitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     1.752 ms      │ 3.237 ms      │ 2.564 ms      │ 2.461 ms      │ 100     │ 100
│  │  │                        5.705 Mitem/s │ 3.089 Mitem/s │ 3.899 Mitem/s │ 4.062 Mitem/s │         │
│  │  ├─ 2                     1.961 ms      │ 3.768 ms      │ 3.207 ms      │ 3.077 ms      │ 100     │ 100
│  │  │                        10.19 Mitem/s │ 5.307 Mitem/s │ 6.236 Mitem/s │ 6.498 Mitem/s │         │
│  │  ├─ 3                     2.7 ms        │ 4.322 ms      │ 3.541 ms      │ 3.542 ms      │ 100     │ 100
│  │  │                        11.11 Mitem/s │ 6.939 Mitem/s │ 8.471 Mitem/s │ 8.468 Mitem/s │         │
│  │  ├─ 4                     2.872 ms      │ 5.053 ms      │ 3.82 ms       │ 3.84 ms       │ 100     │ 100
│  │  │                        13.92 Mitem/s │ 7.915 Mitem/s │ 10.47 Mitem/s │ 10.41 Mitem/s │         │
│  │  ├─ 5                     3.009 ms      │ 5.602 ms      │ 4.582 ms      │ 4.524 ms      │ 100     │ 100
│  │  │                        16.61 Mitem/s │ 8.924 Mitem/s │ 10.91 Mitem/s │ 11.05 Mitem/s │         │
│  │  ╰─ 6                     3.433 ms      │ 6.562 ms      │ 5.247 ms      │ 5.216 ms      │ 100     │ 100
│  │                           17.47 Mitem/s │ 9.142 Mitem/s │ 11.43 Mitem/s │ 11.5 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     2.108 ms      │ 4.242 ms      │ 3.112 ms      │ 3.01 ms       │ 100     │ 100
│     │                        4.742 Mitem/s │ 2.357 Mitem/s │ 3.212 Mitem/s │ 3.321 Mitem/s │         │
│     ├─ 2                     2.406 ms      │ 5.212 ms      │ 4.184 ms      │ 3.969 ms      │ 100     │ 100
│     │                        8.31 Mitem/s  │ 3.836 Mitem/s │ 4.779 Mitem/s │ 5.037 Mitem/s │         │
│     ├─ 3                     3.047 ms      │ 6.668 ms      │ 4.576 ms      │ 4.539 ms      │ 100     │ 100
│     │                        9.844 Mitem/s │ 4.498 Mitem/s │ 6.554 Mitem/s │ 6.608 Mitem/s │         │
│     ├─ 4                     3.61 ms       │ 6.741 ms      │ 4.783 ms      │ 4.828 ms      │ 100     │ 100
│     │                        11.07 Mitem/s │ 5.933 Mitem/s │ 8.362 Mitem/s │ 8.284 Mitem/s │         │
│     ├─ 5                     4.106 ms      │ 7.27 ms       │ 6.49 ms       │ 6.103 ms      │ 100     │ 100
│     │                        12.17 Mitem/s │ 6.876 Mitem/s │ 7.703 Mitem/s │ 8.192 Mitem/s │         │
│     ╰─ 6                     4.005 ms      │ 7.613 ms      │ 6.898 ms      │ 6.515 ms      │ 100     │ 100
│                              14.97 Mitem/s │ 7.88 Mitem/s  │ 8.697 Mitem/s │ 9.208 Mitem/s │         │
├─ 17_hot_spot                               │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     770.7 µs      │ 1.919 ms      │ 1.312 ms      │ 1.347 ms      │ 100     │ 100
│  │  │                        12.97 Mitem/s │ 5.209 Mitem/s │ 7.619 Mitem/s │ 7.42 Mitem/s  │         │
│  │  ├─ 2                     2.498 ms      │ 4.571 ms      │ 3.739 ms      │ 3.745 ms      │ 100     │ 100
│  │  │                        8.003 Mitem/s │ 4.375 Mitem/s │ 5.348 Mitem/s │ 5.339 Mitem/s │         │
│  │  ├─ 3                     4.239 ms      │ 7.739 ms      │ 6.001 ms      │ 6.161 ms      │ 100     │ 100
│  │  │                        7.076 Mitem/s │ 3.876 Mitem/s │ 4.999 Mitem/s │ 4.868 Mitem/s │         │
│  │  ├─ 4                     6.301 ms      │ 11.3 ms       │ 8.955 ms      │ 9.079 ms      │ 100     │ 100
│  │  │                        6.347 Mitem/s │ 3.537 Mitem/s │ 4.466 Mitem/s │ 4.405 Mitem/s │         │
│  │  ├─ 5                     10.05 ms      │ 13.44 ms      │ 11.86 ms      │ 11.86 ms      │ 100     │ 100
│  │  │                        4.972 Mitem/s │ 3.718 Mitem/s │ 4.215 Mitem/s │ 4.215 Mitem/s │         │
│  │  ╰─ 6                     12.4 ms       │ 17.64 ms      │ 15.54 ms      │ 15.32 ms      │ 100     │ 100
│  │                           4.835 Mitem/s │ 3.4 Mitem/s   │ 3.86 Mitem/s  │ 3.916 Mitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     986.5 µs      │ 1.885 ms      │ 1.637 ms      │ 1.597 ms      │ 100     │ 100
│  │  │                        10.13 Mitem/s │ 5.302 Mitem/s │ 6.107 Mitem/s │ 6.261 Mitem/s │         │
│  │  ├─ 2                     2.452 ms      │ 4.853 ms      │ 4.139 ms      │ 4.046 ms      │ 100     │ 100
│  │  │                        8.156 Mitem/s │ 4.12 Mitem/s  │ 4.831 Mitem/s │ 4.942 Mitem/s │         │
│  │  ├─ 3                     2.962 ms      │ 6.885 ms      │ 5.884 ms      │ 5.696 ms      │ 100     │ 100
│  │  │                        10.12 Mitem/s │ 4.356 Mitem/s │ 5.098 Mitem/s │ 5.266 Mitem/s │         │
│  │  ├─ 4                     6.597 ms      │ 9.676 ms      │ 8.386 ms      │ 8.317 ms      │ 100     │ 100
│  │  │                        6.062 Mitem/s │ 4.133 Mitem/s │ 4.769 Mitem/s │ 4.809 Mitem/s │         │
│  │  ├─ 5                     9.083 ms      │ 12.54 ms      │ 10.36 ms      │ 10.41 ms      │ 100     │ 100
│  │  │                        5.504 Mitem/s │ 3.985 Mitem/s │ 4.823 Mitem/s │ 4.802 Mitem/s │         │
│  │  ╰─ 6                     8.06 ms       │ 13.87 ms      │ 12.13 ms      │ 12.12 ms      │ 100     │ 100
│  │                           7.443 Mitem/s │ 4.324 Mitem/s │ 4.942 Mitem/s │ 4.948 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     1.349 ms      │ 2.755 ms      │ 2.121 ms      │ 2.109 ms      │ 100     │ 100
│     │                        7.411 Mitem/s │ 3.628 Mitem/s │ 4.714 Mitem/s │ 4.74 Mitem/s  │         │
│     ├─ 2                     2.485 ms      │ 3.815 ms      │ 3.288 ms      │ 3.27 ms       │ 100     │ 100
│     │                        8.047 Mitem/s │ 5.241 Mitem/s │ 6.081 Mitem/s │ 6.115 Mitem/s │         │
│     ├─ 3                     2.423 ms      │ 4.095 ms      │ 3.433 ms      │ 3.43 ms       │ 100     │ 100
│     │                        12.37 Mitem/s │ 7.324 Mitem/s │ 8.736 Mitem/s │ 8.745 Mitem/s │         │
│     ├─ 4                     3.134 ms      │ 4.647 ms      │ 3.838 ms      │ 3.859 ms      │ 100     │ 100
│     │                        12.75 Mitem/s │ 8.607 Mitem/s │ 10.41 Mitem/s │ 10.36 Mitem/s │         │
│     ├─ 5                     2.751 ms      │ 4.994 ms      │ 4.2 ms        │ 4.2 ms        │ 100     │ 100
│     │                        18.17 Mitem/s │ 10.01 Mitem/s │ 11.9 Mitem/s  │ 11.9 Mitem/s  │         │
│     ╰─ 6                     3.601 ms      │ 5.489 ms      │ 4.44 ms       │ 4.445 ms      │ 100     │ 100
│                              16.65 Mitem/s │ 10.92 Mitem/s │ 13.51 Mitem/s │ 13.49 Mitem/s │         │
╰─ 18_split_inducing_scan                    │               │               │               │         │
   ├─ indexset                               │               │               │               │         │
   │  ├─ 3                     22.98 ms      │ 44.49 ms      │ 29.06 ms      │ 29.81 ms      │ 100     │ 100
   │  ├─ 4                     31.85 ms      │ 53.28 ms      │ 39 ms         │ 39.64 ms      │ 100     │ 100
   │  ├─ 5                     40.86 ms      │ 71.16 ms      │ 51.34 ms      │ 52.05 ms      │ 100     │ 100
   │  ╰─ 6                     58.3 ms       │ 89.2 ms       │ 69.91 ms      │ 70.67 ms      │ 100     │ 100
   ├─ masstree24                             │               │               │               │         │
   │  ├─ 3                     7.891 ms      │ 16.3 ms       │ 9.77 ms       │ 11.01 ms      │ 100     │ 100
   │  ├─ 4                     9.355 ms      │ 22.7 ms       │ 14.79 ms      │ 14 ms         │ 100     │ 100
   │  ├─ 5                     9.598 ms      │ 23.92 ms      │ 13.91 ms      │ 13.73 ms      │ 100     │ 100
   │  ╰─ 6                     10.24 ms      │ 24.63 ms      │ 14.36 ms      │ 15.05 ms      │ 100     │ 100
   ╰─ tree_index                             │               │               │               │         │
      ├─ 3                     9.721 ms      │ 20.92 ms      │ 11.22 ms      │ 12.83 ms      │ 100     │ 100
      ├─ 4                     10.29 ms      │ 21.38 ms      │ 17.53 ms      │ 16.13 ms      │ 100     │ 100
      ├─ 5                     10.34 ms      │ 31.12 ms      │ 18.3 ms       │ 17.61 ms      │ 100     │ 100
      ╰─ 6                     12.09 ms      │ 34.66 ms      │ 19.13 ms      │ 19.91 ms      │ 100     │ 100
```
