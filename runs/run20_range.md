```text
Timer precision: 20 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.501 ms      │ 7.325 ms      │ 4.633 ms      │ 4.897 ms      │ 100     │ 100
│  │  │                        2.221 Mitem/s │ 1.365 Mitem/s │ 2.157 Mitem/s │ 2.041 Mitem/s │         │
│  │  ├─ 2                     11.16 ms      │ 20.02 ms      │ 13.4 ms       │ 14.06 ms      │ 100     │ 100
│  │  │                        1.792 Mitem/s │ 998.6 Kitem/s │ 1.491 Mitem/s │ 1.421 Mitem/s │         │
│  │  ├─ 3                     21.93 ms      │ 37.87 ms      │ 27.5 ms       │ 28.05 ms      │ 100     │ 100
│  │  │                        1.367 Mitem/s │ 792 Kitem/s   │ 1.09 Mitem/s  │ 1.069 Mitem/s │         │
│  │  ├─ 4                     34.07 ms      │ 54.77 ms      │ 39.71 ms      │ 39.94 ms      │ 100     │ 100
│  │  │                        1.173 Mitem/s │ 730.2 Kitem/s │ 1.007 Mitem/s │ 1.001 Mitem/s │         │
│  │  ├─ 5                     51.29 ms      │ 69.92 ms      │ 58.81 ms      │ 58.69 ms      │ 100     │ 100
│  │  │                        974.7 Kitem/s │ 715 Kitem/s   │ 850.1 Kitem/s │ 851.7 Kitem/s │         │
│  │  ╰─ 6                     85.91 ms      │ 121.5 ms      │ 98.65 ms      │ 99.04 ms      │ 100     │ 100
│  │                           698.3 Kitem/s │ 493.7 Kitem/s │ 608.1 Kitem/s │ 605.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.344 ms      │ 11.33 ms      │ 7.662 ms      │ 7.991 ms      │ 100     │ 100
│  │  │                        1.361 Mitem/s │ 881.9 Kitem/s │ 1.305 Mitem/s │ 1.251 Mitem/s │         │
│  │  ├─ 2                     7.85 ms       │ 12.09 ms      │ 8.882 ms      │ 8.993 ms      │ 100     │ 100
│  │  │                        2.547 Mitem/s │ 1.653 Mitem/s │ 2.251 Mitem/s │ 2.223 Mitem/s │         │
│  │  ├─ 3                     7.865 ms      │ 13.88 ms      │ 9.434 ms      │ 10.02 ms      │ 100     │ 100
│  │  │                        3.814 Mitem/s │ 2.161 Mitem/s │ 3.179 Mitem/s │ 2.991 Mitem/s │         │
│  │  ├─ 4                     8.559 ms      │ 23.38 ms      │ 11.95 ms      │ 11.78 ms      │ 100     │ 100
│  │  │                        4.673 Mitem/s │ 1.71 Mitem/s  │ 3.345 Mitem/s │ 3.394 Mitem/s │         │
│  │  ├─ 5                     8.642 ms      │ 21.96 ms      │ 13.3 ms       │ 12.53 ms      │ 100     │ 100
│  │  │                        5.785 Mitem/s │ 2.276 Mitem/s │ 3.759 Mitem/s │ 3.987 Mitem/s │         │
│  │  ╰─ 6                     9.637 ms      │ 22.75 ms      │ 13.43 ms      │ 13.61 ms      │ 100     │ 100
│  │                           6.225 Mitem/s │ 2.636 Mitem/s │ 4.464 Mitem/s │ 4.406 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.65 ms       │ 11.18 ms      │ 9.178 ms      │ 9.361 ms      │ 100     │ 100
│     │                        1.155 Mitem/s │ 893.9 Kitem/s │ 1.089 Mitem/s │ 1.068 Mitem/s │         │
│     ├─ 2                     8.862 ms      │ 18.5 ms       │ 12.35 ms      │ 12.25 ms      │ 100     │ 100
│     │                        2.256 Mitem/s │ 1.08 Mitem/s  │ 1.619 Mitem/s │ 1.632 Mitem/s │         │
│     ├─ 3                     9.065 ms      │ 22.21 ms      │ 15.8 ms       │ 15.32 ms      │ 100     │ 100
│     │                        3.309 Mitem/s │ 1.35 Mitem/s  │ 1.898 Mitem/s │ 1.957 Mitem/s │         │
│     ├─ 4                     9.031 ms      │ 24.84 ms      │ 13.92 ms      │ 14.56 ms      │ 100     │ 100
│     │                        4.429 Mitem/s │ 1.61 Mitem/s  │ 2.871 Mitem/s │ 2.745 Mitem/s │         │
│     ├─ 5                     9.27 ms       │ 29.65 ms      │ 15.6 ms       │ 15.29 ms      │ 100     │ 100
│     │                        5.393 Mitem/s │ 1.686 Mitem/s │ 3.204 Mitem/s │ 3.269 Mitem/s │         │
│     ╰─ 6                     11.42 ms      │ 28.34 ms      │ 16.84 ms      │ 18.39 ms      │ 100     │ 100
│                              5.25 Mitem/s  │ 2.116 Mitem/s │ 3.562 Mitem/s │ 3.261 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.682 ms      │ 7.262 ms      │ 4.842 ms      │ 5.051 ms      │ 100     │ 100
│  │  │                        2.135 Mitem/s │ 1.376 Mitem/s │ 2.065 Mitem/s │ 1.979 Mitem/s │         │
│  │  ├─ 2                     11.98 ms      │ 21.25 ms      │ 15.44 ms      │ 15.49 ms      │ 100     │ 100
│  │  │                        1.668 Mitem/s │ 940.9 Kitem/s │ 1.294 Mitem/s │ 1.29 Mitem/s  │         │
│  │  ├─ 3                     23.83 ms      │ 39.3 ms       │ 30.45 ms      │ 30.94 ms      │ 100     │ 100
│  │  │                        1.258 Mitem/s │ 763.1 Kitem/s │ 985.2 Kitem/s │ 969.6 Kitem/s │         │
│  │  ├─ 4                     36.96 ms      │ 54.68 ms      │ 43.39 ms      │ 43.94 ms      │ 100     │ 100
│  │  │                        1.081 Mitem/s │ 731.4 Kitem/s │ 921.6 Kitem/s │ 910.2 Kitem/s │         │
│  │  ├─ 5                     56.31 ms      │ 71.76 ms      │ 61.91 ms      │ 62.06 ms      │ 100     │ 100
│  │  │                        887.9 Kitem/s │ 696.6 Kitem/s │ 807.4 Kitem/s │ 805.5 Kitem/s │         │
│  │  ╰─ 6                     88.15 ms      │ 120.5 ms      │ 100.7 ms      │ 100.3 ms      │ 100     │ 100
│  │                           680.6 Kitem/s │ 497.6 Kitem/s │ 595.7 Kitem/s │ 597.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.29 ms       │ 10.78 ms      │ 7.545 ms      │ 7.998 ms      │ 100     │ 100
│  │  │                        1.371 Mitem/s │ 926.8 Kitem/s │ 1.325 Mitem/s │ 1.25 Mitem/s  │         │
│  │  ├─ 2                     7.643 ms      │ 16.06 ms      │ 9.397 ms      │ 10.43 ms      │ 100     │ 100
│  │  │                        2.616 Mitem/s │ 1.245 Mitem/s │ 2.128 Mitem/s │ 1.917 Mitem/s │         │
│  │  ├─ 3                     8.011 ms      │ 15.26 ms      │ 9.384 ms      │ 10.38 ms      │ 100     │ 100
│  │  │                        3.744 Mitem/s │ 1.965 Mitem/s │ 3.196 Mitem/s │ 2.89 Mitem/s  │         │
│  │  ├─ 4                     8.44 ms       │ 21.21 ms      │ 12 ms         │ 11.85 ms      │ 100     │ 100
│  │  │                        4.738 Mitem/s │ 1.885 Mitem/s │ 3.332 Mitem/s │ 3.373 Mitem/s │         │
│  │  ├─ 5                     8.652 ms      │ 19.46 ms      │ 12.45 ms      │ 12.12 ms      │ 100     │ 100
│  │  │                        5.778 Mitem/s │ 2.568 Mitem/s │ 4.015 Mitem/s │ 4.124 Mitem/s │         │
│  │  ╰─ 6                     8.69 ms       │ 25.22 ms      │ 13.45 ms      │ 14.11 ms      │ 100     │ 100
│  │                           6.904 Mitem/s │ 2.378 Mitem/s │ 4.46 Mitem/s  │ 4.25 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.543 ms      │ 13.66 ms      │ 10.06 ms      │ 10.41 ms      │ 100     │ 100
│     │                        1.047 Mitem/s │ 731.6 Kitem/s │ 993.4 Kitem/s │ 959.8 Kitem/s │         │
│     ├─ 2                     9.631 ms      │ 19.55 ms      │ 11.25 ms      │ 12.29 ms      │ 100     │ 100
│     │                        2.076 Mitem/s │ 1.022 Mitem/s │ 1.776 Mitem/s │ 1.626 Mitem/s │         │
│     ├─ 3                     9.947 ms      │ 21.54 ms      │ 14.5 ms       │ 14.27 ms      │ 100     │ 100
│     │                        3.015 Mitem/s │ 1.392 Mitem/s │ 2.068 Mitem/s │ 2.1 Mitem/s   │         │
│     ├─ 4                     10.07 ms      │ 29.14 ms      │ 14.73 ms      │ 14.57 ms      │ 100     │ 100
│     │                        3.971 Mitem/s │ 1.372 Mitem/s │ 2.715 Mitem/s │ 2.743 Mitem/s │         │
│     ├─ 5                     10.28 ms      │ 31.37 ms      │ 17.5 ms       │ 17.4 ms       │ 100     │ 100
│     │                        4.863 Mitem/s │ 1.593 Mitem/s │ 2.856 Mitem/s │ 2.872 Mitem/s │         │
│     ╰─ 6                     10.78 ms      │ 29.89 ms      │ 17.57 ms      │ 18.35 ms      │ 100     │ 100
│                              5.563 Mitem/s │ 2.007 Mitem/s │ 3.413 Mitem/s │ 3.268 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.577 ms      │ 6.609 ms      │ 4.827 ms      │ 4.985 ms      │ 100     │ 100
│  │  │                        2.184 Mitem/s │ 1.512 Mitem/s │ 2.071 Mitem/s │ 2.005 Mitem/s │         │
│  │  ├─ 2                     11.57 ms      │ 20.74 ms      │ 13.25 ms      │ 14.13 ms      │ 100     │ 100
│  │  │                        1.728 Mitem/s │ 964 Kitem/s   │ 1.508 Mitem/s │ 1.415 Mitem/s │         │
│  │  ├─ 3                     21.02 ms      │ 37.2 ms       │ 28.19 ms      │ 28.03 ms      │ 100     │ 100
│  │  │                        1.426 Mitem/s │ 806.2 Kitem/s │ 1.064 Mitem/s │ 1.07 Mitem/s  │         │
│  │  ├─ 4                     33.48 ms      │ 49.64 ms      │ 40.98 ms      │ 41.1 ms       │ 100     │ 100
│  │  │                        1.194 Mitem/s │ 805.6 Kitem/s │ 975.8 Kitem/s │ 973.1 Kitem/s │         │
│  │  ├─ 5                     53.41 ms      │ 71.43 ms      │ 60.43 ms      │ 60.18 ms      │ 100     │ 100
│  │  │                        936.1 Kitem/s │ 699.8 Kitem/s │ 827.2 Kitem/s │ 830.7 Kitem/s │         │
│  │  ╰─ 6                     83.42 ms      │ 111.3 ms      │ 100.6 ms      │ 100.4 ms      │ 100     │ 100
│  │                           719.2 Kitem/s │ 538.7 Kitem/s │ 596.2 Kitem/s │ 597.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.323 ms      │ 10.96 ms      │ 7.64 ms       │ 8.142 ms      │ 100     │ 100
│  │  │                        1.365 Mitem/s │ 912 Kitem/s   │ 1.308 Mitem/s │ 1.228 Mitem/s │         │
│  │  ├─ 2                     7.751 ms      │ 16.56 ms      │ 9.398 ms      │ 10.35 ms      │ 100     │ 100
│  │  │                        2.58 Mitem/s  │ 1.207 Mitem/s │ 2.128 Mitem/s │ 1.931 Mitem/s │         │
│  │  ├─ 3                     8.046 ms      │ 22.83 ms      │ 11.92 ms      │ 11.74 ms      │ 100     │ 100
│  │  │                        3.728 Mitem/s │ 1.313 Mitem/s │ 2.515 Mitem/s │ 2.553 Mitem/s │         │
│  │  ├─ 4                     8.59 ms       │ 21.95 ms      │ 12.13 ms      │ 12.35 ms      │ 100     │ 100
│  │  │                        4.656 Mitem/s │ 1.821 Mitem/s │ 3.295 Mitem/s │ 3.237 Mitem/s │         │
│  │  ├─ 5                     8.838 ms      │ 23.12 ms      │ 13.31 ms      │ 13.47 ms      │ 100     │ 100
│  │  │                        5.656 Mitem/s │ 2.161 Mitem/s │ 3.754 Mitem/s │ 3.71 Mitem/s  │         │
│  │  ╰─ 6                     8.777 ms      │ 26.1 ms       │ 13.61 ms      │ 13.72 ms      │ 100     │ 100
│  │                           6.835 Mitem/s │ 2.298 Mitem/s │ 4.408 Mitem/s │ 4.371 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.504 ms      │ 12.94 ms      │ 9.212 ms      │ 9.87 ms       │ 100     │ 100
│     │                        1.175 Mitem/s │ 772.4 Kitem/s │ 1.085 Mitem/s │ 1.013 Mitem/s │         │
│     ├─ 2                     8.933 ms      │ 18.82 ms      │ 11.96 ms      │ 12.19 ms      │ 100     │ 100
│     │                        2.238 Mitem/s │ 1.062 Mitem/s │ 1.671 Mitem/s │ 1.639 Mitem/s │         │
│     ├─ 3                     9.101 ms      │ 18.06 ms      │ 13.5 ms       │ 13.49 ms      │ 100     │ 100
│     │                        3.296 Mitem/s │ 1.66 Mitem/s  │ 2.221 Mitem/s │ 2.222 Mitem/s │         │
│     ├─ 4                     9.245 ms      │ 28.05 ms      │ 13.78 ms      │ 14.13 ms      │ 100     │ 100
│     │                        4.326 Mitem/s │ 1.425 Mitem/s │ 2.9 Mitem/s   │ 2.829 Mitem/s │         │
│     ├─ 5                     9.504 ms      │ 28.81 ms      │ 16.44 ms      │ 16.26 ms      │ 100     │ 100
│     │                        5.26 Mitem/s  │ 1.735 Mitem/s │ 3.04 Mitem/s  │ 3.073 Mitem/s │         │
│     ╰─ 6                     9.615 ms      │ 31.15 ms      │ 16.79 ms      │ 17.64 ms      │ 100     │ 100
│                              6.239 Mitem/s │ 1.925 Mitem/s │ 3.572 Mitem/s │ 3.4 Mitem/s   │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.537 ms      │ 7.264 ms      │ 4.674 ms      │ 4.922 ms      │ 100     │ 100
│  │  │                        2.203 Mitem/s │ 1.376 Mitem/s │ 2.139 Mitem/s │ 2.031 Mitem/s │         │
│  │  ├─ 2                     11.45 ms      │ 21.1 ms       │ 14.19 ms      │ 14.88 ms      │ 100     │ 100
│  │  │                        1.745 Mitem/s │ 947.4 Kitem/s │ 1.408 Mitem/s │ 1.343 Mitem/s │         │
│  │  ├─ 3                     21.05 ms      │ 37.58 ms      │ 26.05 ms      │ 26.87 ms      │ 100     │ 100
│  │  │                        1.425 Mitem/s │ 798 Kitem/s   │ 1.151 Mitem/s │ 1.116 Mitem/s │         │
│  │  ├─ 4                     34.42 ms      │ 47.98 ms      │ 40.75 ms      │ 41.13 ms      │ 100     │ 100
│  │  │                        1.161 Mitem/s │ 833.5 Kitem/s │ 981.5 Kitem/s │ 972.5 Kitem/s │         │
│  │  ├─ 5                     53.44 ms      │ 65.12 ms      │ 59.14 ms      │ 59.07 ms      │ 100     │ 100
│  │  │                        935.4 Kitem/s │ 767.7 Kitem/s │ 845.3 Kitem/s │ 846.3 Kitem/s │         │
│  │  ╰─ 6                     85.58 ms      │ 114.4 ms      │ 100.5 ms      │ 100.2 ms      │ 100     │ 100
│  │                           701 Kitem/s   │ 524 Kitem/s   │ 596.8 Kitem/s │ 598.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.404 ms      │ 11.63 ms      │ 7.739 ms      │ 8.062 ms      │ 100     │ 100
│  │  │                        1.35 Mitem/s  │ 859.7 Kitem/s │ 1.292 Mitem/s │ 1.24 Mitem/s  │         │
│  │  ├─ 2                     8.112 ms      │ 15.54 ms      │ 9.38 ms       │ 10.05 ms      │ 100     │ 100
│  │  │                        2.465 Mitem/s │ 1.286 Mitem/s │ 2.132 Mitem/s │ 1.988 Mitem/s │         │
│  │  ├─ 3                     7.971 ms      │ 15.62 ms      │ 11.63 ms      │ 11.02 ms      │ 100     │ 100
│  │  │                        3.763 Mitem/s │ 1.919 Mitem/s │ 2.577 Mitem/s │ 2.721 Mitem/s │         │
│  │  ├─ 4                     8.747 ms      │ 19.35 ms      │ 12.75 ms      │ 12.81 ms      │ 100     │ 100
│  │  │                        4.572 Mitem/s │ 2.066 Mitem/s │ 3.136 Mitem/s │ 3.12 Mitem/s  │         │
│  │  ├─ 5                     8.546 ms      │ 24.39 ms      │ 13.55 ms      │ 13.47 ms      │ 100     │ 100
│  │  │                        5.85 Mitem/s  │ 2.049 Mitem/s │ 3.687 Mitem/s │ 3.711 Mitem/s │         │
│  │  ╰─ 6                     8.789 ms      │ 23.66 ms      │ 13.78 ms      │ 13.95 ms      │ 100     │ 100
│  │                           6.825 Mitem/s │ 2.535 Mitem/s │ 4.354 Mitem/s │ 4.298 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.579 ms      │ 11.32 ms      │ 8.967 ms      │ 9.08 ms       │ 100     │ 100
│     │                        1.165 Mitem/s │ 882.9 Kitem/s │ 1.115 Mitem/s │ 1.101 Mitem/s │         │
│     ├─ 2                     9.068 ms      │ 19.36 ms      │ 12.7 ms       │ 13.05 ms      │ 100     │ 100
│     │                        2.205 Mitem/s │ 1.032 Mitem/s │ 1.574 Mitem/s │ 1.531 Mitem/s │         │
│     ├─ 3                     9.02 ms       │ 20.25 ms      │ 13.65 ms      │ 14.62 ms      │ 100     │ 100
│     │                        3.325 Mitem/s │ 1.48 Mitem/s  │ 2.197 Mitem/s │ 2.05 Mitem/s  │         │
│     ├─ 4                     9.066 ms      │ 27 ms         │ 13.88 ms      │ 14.07 ms      │ 100     │ 100
│     │                        4.412 Mitem/s │ 1.481 Mitem/s │ 2.88 Mitem/s  │ 2.841 Mitem/s │         │
│     ├─ 5                     9.127 ms      │ 28.9 ms       │ 16.1 ms       │ 16.05 ms      │ 100     │ 100
│     │                        5.478 Mitem/s │ 1.729 Mitem/s │ 3.105 Mitem/s │ 3.115 Mitem/s │         │
│     ╰─ 6                     9.053 ms      │ 29.94 ms      │ 16.84 ms      │ 17.85 ms      │ 100     │ 100
│                              6.627 Mitem/s │ 2.003 Mitem/s │ 3.562 Mitem/s │ 3.36 Mitem/s  │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.171 ms      │ 5.23 ms       │ 3.397 ms      │ 3.514 ms      │ 100     │ 100
│  │  │                        3.152 Mitem/s │ 1.911 Mitem/s │ 2.943 Mitem/s │ 2.845 Mitem/s │         │
│  │  ├─ 2                     7.127 ms      │ 14.84 ms      │ 10.34 ms      │ 10.22 ms      │ 100     │ 100
│  │  │                        2.806 Mitem/s │ 1.347 Mitem/s │ 1.933 Mitem/s │ 1.955 Mitem/s │         │
│  │  ├─ 3                     14.57 ms      │ 28.24 ms      │ 20.23 ms      │ 20.05 ms      │ 100     │ 100
│  │  │                        2.058 Mitem/s │ 1.062 Mitem/s │ 1.482 Mitem/s │ 1.495 Mitem/s │         │
│  │  ├─ 4                     26.12 ms      │ 40.96 ms      │ 30.2 ms       │ 30.34 ms      │ 100     │ 100
│  │  │                        1.531 Mitem/s │ 976.3 Kitem/s │ 1.324 Mitem/s │ 1.318 Mitem/s │         │
│  │  ├─ 5                     37.01 ms      │ 49.87 ms      │ 41.93 ms      │ 41.82 ms      │ 100     │ 100
│  │  │                        1.35 Mitem/s  │ 1.002 Mitem/s │ 1.192 Mitem/s │ 1.195 Mitem/s │         │
│  │  ╰─ 6                     63.84 ms      │ 87.3 ms       │ 74.38 ms      │ 74.11 ms      │ 100     │ 100
│  │                           939.8 Kitem/s │ 687.2 Kitem/s │ 806.6 Kitem/s │ 809.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.719 ms      │ 9.093 ms      │ 6.922 ms      │ 7.108 ms      │ 100     │ 100
│  │  │                        1.488 Mitem/s │ 1.099 Mitem/s │ 1.444 Mitem/s │ 1.406 Mitem/s │         │
│  │  ├─ 2                     7.035 ms      │ 14.53 ms      │ 8.015 ms      │ 8.461 ms      │ 100     │ 100
│  │  │                        2.842 Mitem/s │ 1.376 Mitem/s │ 2.495 Mitem/s │ 2.363 Mitem/s │         │
│  │  ├─ 3                     7.289 ms      │ 15.08 ms      │ 11.33 ms      │ 11.01 ms      │ 100     │ 100
│  │  │                        4.115 Mitem/s │ 1.988 Mitem/s │ 2.646 Mitem/s │ 2.722 Mitem/s │         │
│  │  ├─ 4                     7.817 ms      │ 17.88 ms      │ 11.06 ms      │ 11.19 ms      │ 100     │ 100
│  │  │                        5.116 Mitem/s │ 2.236 Mitem/s │ 3.615 Mitem/s │ 3.572 Mitem/s │         │
│  │  ├─ 5                     7.838 ms      │ 20.93 ms      │ 12.05 ms      │ 12.2 ms       │ 100     │ 100
│  │  │                        6.379 Mitem/s │ 2.388 Mitem/s │ 4.148 Mitem/s │ 4.096 Mitem/s │         │
│  │  ╰─ 6                     8.04 ms       │ 18.33 ms      │ 12.44 ms      │ 11.79 ms      │ 100     │ 100
│  │                           7.461 Mitem/s │ 3.271 Mitem/s │ 4.82 Mitem/s  │ 5.085 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.557 ms      │ 12.79 ms      │ 9.195 ms      │ 9.5 ms        │ 100     │ 100
│     │                        1.168 Mitem/s │ 781.6 Kitem/s │ 1.087 Mitem/s │ 1.052 Mitem/s │         │
│     ├─ 2                     8.872 ms      │ 17.25 ms      │ 10.28 ms      │ 10.84 ms      │ 100     │ 100
│     │                        2.254 Mitem/s │ 1.159 Mitem/s │ 1.944 Mitem/s │ 1.844 Mitem/s │         │
│     ├─ 3                     9.274 ms      │ 17.66 ms      │ 12.96 ms      │ 12.43 ms      │ 100     │ 100
│     │                        3.234 Mitem/s │ 1.697 Mitem/s │ 2.314 Mitem/s │ 2.412 Mitem/s │         │
│     ├─ 4                     9.137 ms      │ 29.26 ms      │ 14.71 ms      │ 15.17 ms      │ 100     │ 100
│     │                        4.377 Mitem/s │ 1.367 Mitem/s │ 2.718 Mitem/s │ 2.636 Mitem/s │         │
│     ├─ 5                     9.498 ms      │ 25.95 ms      │ 15.62 ms      │ 15.43 ms      │ 100     │ 100
│     │                        5.264 Mitem/s │ 1.926 Mitem/s │ 3.199 Mitem/s │ 3.24 Mitem/s  │         │
│     ╰─ 6                     10.26 ms      │ 29.16 ms      │ 16.31 ms      │ 17.82 ms      │ 100     │ 100
│                              5.846 Mitem/s │ 2.057 Mitem/s │ 3.678 Mitem/s │ 3.366 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.287 ms      │ 5.054 ms      │ 3.379 ms      │ 3.494 ms      │ 100     │ 100
│  │  │                        3.042 Mitem/s │ 1.978 Mitem/s │ 2.958 Mitem/s │ 2.861 Mitem/s │         │
│  │  ├─ 2                     7.58 ms       │ 14.75 ms      │ 9.12 ms       │ 9.469 ms      │ 100     │ 100
│  │  │                        2.638 Mitem/s │ 1.355 Mitem/s │ 2.192 Mitem/s │ 2.112 Mitem/s │         │
│  │  ├─ 3                     13.72 ms      │ 26.74 ms      │ 19.6 ms       │ 19.42 ms      │ 100     │ 100
│  │  │                        2.185 Mitem/s │ 1.121 Mitem/s │ 1.53 Mitem/s  │ 1.544 Mitem/s │         │
│  │  ├─ 4                     24 ms         │ 34.93 ms      │ 28.14 ms      │ 28.08 ms      │ 100     │ 100
│  │  │                        1.666 Mitem/s │ 1.144 Mitem/s │ 1.421 Mitem/s │ 1.424 Mitem/s │         │
│  │  ├─ 5                     35.46 ms      │ 50.66 ms      │ 43.64 ms      │ 43.01 ms      │ 100     │ 100
│  │  │                        1.409 Mitem/s │ 986.9 Kitem/s │ 1.145 Mitem/s │ 1.162 Mitem/s │         │
│  │  ╰─ 6                     60.9 ms       │ 85.98 ms      │ 75.1 ms       │ 74.75 ms      │ 100     │ 100
│  │                           985.1 Kitem/s │ 697.7 Kitem/s │ 798.8 Kitem/s │ 802.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.702 ms      │ 18.31 ms      │ 10.16 ms      │ 10.57 ms      │ 100     │ 100
│  │  │                        1.03 Mitem/s  │ 545.9 Kitem/s │ 984.1 Kitem/s │ 946 Kitem/s   │         │
│  │  ├─ 2                     9.816 ms      │ 18.63 ms      │ 10.77 ms      │ 11.36 ms      │ 100     │ 100
│  │  │                        2.037 Mitem/s │ 1.073 Mitem/s │ 1.856 Mitem/s │ 1.759 Mitem/s │         │
│  │  ├─ 3                     9.983 ms      │ 25.79 ms      │ 11.59 ms      │ 13.33 ms      │ 100     │ 100
│  │  │                        3.004 Mitem/s │ 1.162 Mitem/s │ 2.587 Mitem/s │ 2.25 Mitem/s  │         │
│  │  ├─ 4                     10.02 ms      │ 31.48 ms      │ 17.66 ms      │ 16.43 ms      │ 100     │ 100
│  │  │                        3.99 Mitem/s  │ 1.27 Mitem/s  │ 2.263 Mitem/s │ 2.433 Mitem/s │         │
│  │  ├─ 5                     10.96 ms      │ 31.84 ms      │ 18.27 ms      │ 18.07 ms      │ 100     │ 100
│  │  │                        4.559 Mitem/s │ 1.57 Mitem/s  │ 2.736 Mitem/s │ 2.765 Mitem/s │         │
│  │  ╰─ 6                     10.69 ms      │ 31.74 ms      │ 18.89 ms      │ 19.75 ms      │ 100     │ 100
│  │                           5.612 Mitem/s │ 1.89 Mitem/s  │ 3.175 Mitem/s │ 3.037 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.264 ms      │ 15.3 ms       │ 9.957 ms      │ 10.31 ms      │ 100     │ 100
│     │                        1.079 Mitem/s │ 653.3 Kitem/s │ 1.004 Mitem/s │ 969.8 Kitem/s │         │
│     ├─ 2                     9.601 ms      │ 20.47 ms      │ 11.44 ms      │ 12.48 ms      │ 100     │ 100
│     │                        2.083 Mitem/s │ 976.5 Kitem/s │ 1.747 Mitem/s │ 1.601 Mitem/s │         │
│     ├─ 3                     9.606 ms      │ 21.05 ms      │ 14.43 ms      │ 13.96 ms      │ 100     │ 100
│     │                        3.122 Mitem/s │ 1.424 Mitem/s │ 2.077 Mitem/s │ 2.148 Mitem/s │         │
│     ├─ 4                     10.16 ms      │ 30.89 ms      │ 15.07 ms      │ 16.09 ms      │ 100     │ 100
│     │                        3.933 Mitem/s │ 1.294 Mitem/s │ 2.654 Mitem/s │ 2.484 Mitem/s │         │
│     ├─ 5                     10.05 ms      │ 31.91 ms      │ 18.31 ms      │ 18.14 ms      │ 100     │ 100
│     │                        4.973 Mitem/s │ 1.566 Mitem/s │ 2.73 Mitem/s  │ 2.755 Mitem/s │         │
│     ╰─ 6                     10.19 ms      │ 30.85 ms      │ 18.62 ms      │ 18.77 ms      │ 100     │ 100
│                              5.883 Mitem/s │ 1.944 Mitem/s │ 3.221 Mitem/s │ 3.195 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.285 ms      │ 6.715 ms      │ 3.393 ms      │ 3.583 ms      │ 100     │ 100
│  │  │                        3.043 Mitem/s │ 1.489 Mitem/s │ 2.946 Mitem/s │ 2.79 Mitem/s  │         │
│  │  ├─ 2                     7.884 ms      │ 15.04 ms      │ 9.579 ms      │ 9.947 ms      │ 100     │ 100
│  │  │                        2.536 Mitem/s │ 1.329 Mitem/s │ 2.087 Mitem/s │ 2.01 Mitem/s  │         │
│  │  ├─ 3                     12.82 ms      │ 27.92 ms      │ 18.57 ms      │ 18.76 ms      │ 100     │ 100
│  │  │                        2.338 Mitem/s │ 1.074 Mitem/s │ 1.615 Mitem/s │ 1.598 Mitem/s │         │
│  │  ├─ 4                     24.65 ms      │ 37.02 ms      │ 30.38 ms      │ 30.28 ms      │ 100     │ 100
│  │  │                        1.622 Mitem/s │ 1.08 Mitem/s  │ 1.316 Mitem/s │ 1.32 Mitem/s  │         │
│  │  ├─ 5                     36.71 ms      │ 51.58 ms      │ 43.86 ms      │ 44 ms         │ 100     │ 100
│  │  │                        1.362 Mitem/s │ 969.1 Kitem/s │ 1.139 Mitem/s │ 1.136 Mitem/s │         │
│  │  ╰─ 6                     63.34 ms      │ 84.93 ms      │ 75.2 ms       │ 74.93 ms      │ 100     │ 100
│  │                           947.2 Kitem/s │ 706.4 Kitem/s │ 797.8 Kitem/s │ 800.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.311 ms      │ 11.44 ms      │ 7.613 ms      │ 8.443 ms      │ 100     │ 100
│  │  │                        1.367 Mitem/s │ 873.6 Kitem/s │ 1.313 Mitem/s │ 1.184 Mitem/s │         │
│  │  ├─ 2                     7.751 ms      │ 15.6 ms       │ 10.86 ms      │ 10.84 ms      │ 100     │ 100
│  │  │                        2.58 Mitem/s  │ 1.281 Mitem/s │ 1.84 Mitem/s  │ 1.844 Mitem/s │         │
│  │  ├─ 3                     7.877 ms      │ 16.31 ms      │ 11.48 ms      │ 11.32 ms      │ 100     │ 100
│  │  │                        3.808 Mitem/s │ 1.838 Mitem/s │ 2.612 Mitem/s │ 2.648 Mitem/s │         │
│  │  ├─ 4                     8.365 ms      │ 19.48 ms      │ 11.68 ms      │ 11.4 ms       │ 100     │ 100
│  │  │                        4.781 Mitem/s │ 2.052 Mitem/s │ 3.422 Mitem/s │ 3.508 Mitem/s │         │
│  │  ├─ 5                     8.597 ms      │ 22.99 ms      │ 13.4 ms       │ 13.31 ms      │ 100     │ 100
│  │  │                        5.815 Mitem/s │ 2.174 Mitem/s │ 3.73 Mitem/s  │ 3.754 Mitem/s │         │
│  │  ╰─ 6                     8.671 ms      │ 20.04 ms      │ 13.54 ms      │ 13.19 ms      │ 100     │ 100
│  │                           6.919 Mitem/s │ 2.992 Mitem/s │ 4.428 Mitem/s │ 4.548 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.172 ms      │ 11.41 ms      │ 9.68 ms       │ 9.764 ms      │ 100     │ 100
│     │                        1.09 Mitem/s  │ 875.8 Kitem/s │ 1.032 Mitem/s │ 1.024 Mitem/s │         │
│     ├─ 2                     9.46 ms       │ 17.99 ms      │ 11.18 ms      │ 11.76 ms      │ 100     │ 100
│     │                        2.114 Mitem/s │ 1.111 Mitem/s │ 1.788 Mitem/s │ 1.699 Mitem/s │         │
│     ├─ 3                     9.62 ms       │ 22.32 ms      │ 14.2 ms       │ 14.07 ms      │ 100     │ 100
│     │                        3.118 Mitem/s │ 1.343 Mitem/s │ 2.112 Mitem/s │ 2.132 Mitem/s │         │
│     ├─ 4                     9.815 ms      │ 28.34 ms      │ 14.58 ms      │ 14.5 ms       │ 100     │ 100
│     │                        4.075 Mitem/s │ 1.411 Mitem/s │ 2.742 Mitem/s │ 2.757 Mitem/s │         │
│     ├─ 5                     10.16 ms      │ 29.17 ms      │ 17.69 ms      │ 16.76 ms      │ 100     │ 100
│     │                        4.917 Mitem/s │ 1.714 Mitem/s │ 2.825 Mitem/s │ 2.982 Mitem/s │         │
│     ╰─ 6                     10.14 ms      │ 31.32 ms      │ 18.68 ms      │ 18.59 ms      │ 100     │ 100
│                              5.912 Mitem/s │ 1.915 Mitem/s │ 3.211 Mitem/s │ 3.227 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.551 ms      │ 6.732 ms      │ 4.655 ms      │ 4.804 ms      │ 100     │ 100
│  │  │                        2.197 Mitem/s │ 1.485 Mitem/s │ 2.148 Mitem/s │ 2.081 Mitem/s │         │
│  │  ├─ 2                     11.38 ms      │ 22.79 ms      │ 14.8 ms       │ 14.71 ms      │ 100     │ 100
│  │  │                        1.756 Mitem/s │ 877.3 Kitem/s │ 1.35 Mitem/s  │ 1.359 Mitem/s │         │
│  │  ├─ 3                     21.81 ms      │ 37.23 ms      │ 26.64 ms      │ 27.01 ms      │ 100     │ 100
│  │  │                        1.375 Mitem/s │ 805.5 Kitem/s │ 1.125 Mitem/s │ 1.11 Mitem/s  │         │
│  │  ├─ 4                     34.56 ms      │ 54.33 ms      │ 41.37 ms      │ 41.45 ms      │ 100     │ 100
│  │  │                        1.157 Mitem/s │ 736.2 Kitem/s │ 966.6 Kitem/s │ 964.9 Kitem/s │         │
│  │  ├─ 5                     53.42 ms      │ 69.19 ms      │ 60.8 ms       │ 60.82 ms      │ 100     │ 100
│  │  │                        935.8 Kitem/s │ 722.5 Kitem/s │ 822.2 Kitem/s │ 822 Kitem/s   │         │
│  │  ╰─ 6                     86.79 ms      │ 120.5 ms      │ 100.9 ms      │ 100.3 ms      │ 100     │ 100
│  │                           691.3 Kitem/s │ 497.7 Kitem/s │ 594.2 Kitem/s │ 597.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.723 ms      │ 12.3 ms       │ 7.111 ms      │ 7.681 ms      │ 100     │ 100
│  │  │                        1.487 Mitem/s │ 812.6 Kitem/s │ 1.406 Mitem/s │ 1.301 Mitem/s │         │
│  │  ├─ 2                     7.167 ms      │ 13.99 ms      │ 8.105 ms      │ 8.472 ms      │ 100     │ 100
│  │  │                        2.79 Mitem/s  │ 1.429 Mitem/s │ 2.467 Mitem/s │ 2.36 Mitem/s  │         │
│  │  ├─ 3                     7.153 ms      │ 15.27 ms      │ 9.759 ms      │ 10.53 ms      │ 100     │ 100
│  │  │                        4.193 Mitem/s │ 1.964 Mitem/s │ 3.074 Mitem/s │ 2.846 Mitem/s │         │
│  │  ├─ 4                     8.161 ms      │ 18.42 ms      │ 11.11 ms      │ 11.3 ms       │ 100     │ 100
│  │  │                        4.9 Mitem/s   │ 2.17 Mitem/s  │ 3.599 Mitem/s │ 3.538 Mitem/s │         │
│  │  ├─ 5                     8.003 ms      │ 21.63 ms      │ 12.36 ms      │ 11.9 ms       │ 100     │ 100
│  │  │                        6.247 Mitem/s │ 2.31 Mitem/s  │ 4.043 Mitem/s │ 4.199 Mitem/s │         │
│  │  ╰─ 6                     8.099 ms      │ 23.16 ms      │ 12.81 ms      │ 13.65 ms      │ 100     │ 100
│  │                           7.407 Mitem/s │ 2.59 Mitem/s  │ 4.683 Mitem/s │ 4.392 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.318 ms      │ 10.9 ms       │ 8.671 ms      │ 8.926 ms      │ 100     │ 100
│     │                        1.202 Mitem/s │ 916.8 Kitem/s │ 1.153 Mitem/s │ 1.12 Mitem/s  │         │
│     ├─ 2                     8.5 ms        │ 18.33 ms      │ 9.862 ms      │ 10.46 ms      │ 100     │ 100
│     │                        2.352 Mitem/s │ 1.09 Mitem/s  │ 2.027 Mitem/s │ 1.91 Mitem/s  │         │
│     ├─ 3                     8.606 ms      │ 18.99 ms      │ 12.04 ms      │ 12.09 ms      │ 100     │ 100
│     │                        3.485 Mitem/s │ 1.579 Mitem/s │ 2.49 Mitem/s  │ 2.48 Mitem/s  │         │
│     ├─ 4                     9.22 ms       │ 24.66 ms      │ 13.56 ms      │ 13.83 ms      │ 100     │ 100
│     │                        4.337 Mitem/s │ 1.622 Mitem/s │ 2.949 Mitem/s │ 2.891 Mitem/s │         │
│     ├─ 5                     8.769 ms      │ 23.53 ms      │ 14.85 ms      │ 14.14 ms      │ 100     │ 100
│     │                        5.701 Mitem/s │ 2.124 Mitem/s │ 3.364 Mitem/s │ 3.536 Mitem/s │         │
│     ╰─ 6                     8.986 ms      │ 25.05 ms      │ 15.28 ms      │ 16.23 ms      │ 100     │ 100
│                              6.676 Mitem/s │ 2.394 Mitem/s │ 3.924 Mitem/s │ 3.695 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.624 ms      │ 7.296 ms      │ 4.773 ms      │ 5.004 ms      │ 100     │ 100
│  │  │                        2.162 Mitem/s │ 1.37 Mitem/s  │ 2.094 Mitem/s │ 1.998 Mitem/s │         │
│  │  ├─ 2                     11.78 ms      │ 20.71 ms      │ 16.09 ms      │ 15.67 ms      │ 100     │ 100
│  │  │                        1.696 Mitem/s │ 965.5 Kitem/s │ 1.242 Mitem/s │ 1.275 Mitem/s │         │
│  │  ├─ 3                     20.89 ms      │ 38.93 ms      │ 25.42 ms      │ 25.93 ms      │ 100     │ 100
│  │  │                        1.435 Mitem/s │ 770.5 Kitem/s │ 1.18 Mitem/s  │ 1.156 Mitem/s │         │
│  │  ├─ 4                     35.32 ms      │ 49.04 ms      │ 40.81 ms      │ 40.86 ms      │ 100     │ 100
│  │  │                        1.132 Mitem/s │ 815.5 Kitem/s │ 979.9 Kitem/s │ 978.8 Kitem/s │         │
│  │  ├─ 5                     54.11 ms      │ 68.41 ms      │ 60.68 ms      │ 60.54 ms      │ 100     │ 100
│  │  │                        923.8 Kitem/s │ 730.8 Kitem/s │ 823.9 Kitem/s │ 825.8 Kitem/s │         │
│  │  ╰─ 6                     83.79 ms      │ 113.1 ms      │ 101.8 ms      │ 101.1 ms      │ 100     │ 100
│  │                           716 Kitem/s   │ 530.2 Kitem/s │ 589.3 Kitem/s │ 592.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.375 ms      │ 10.77 ms      │ 7.673 ms      │ 8.201 ms      │ 100     │ 100
│  │  │                        1.355 Mitem/s │ 927.6 Kitem/s │ 1.303 Mitem/s │ 1.219 Mitem/s │         │
│  │  ├─ 2                     7.755 ms      │ 14.92 ms      │ 9.32 ms       │ 10.06 ms      │ 100     │ 100
│  │  │                        2.578 Mitem/s │ 1.34 Mitem/s  │ 2.145 Mitem/s │ 1.987 Mitem/s │         │
│  │  ├─ 3                     7.824 ms      │ 16.37 ms      │ 12.18 ms      │ 11.91 ms      │ 100     │ 100
│  │  │                        3.833 Mitem/s │ 1.832 Mitem/s │ 2.462 Mitem/s │ 2.517 Mitem/s │         │
│  │  ├─ 4                     8.565 ms      │ 18.19 ms      │ 12.19 ms      │ 12.12 ms      │ 100     │ 100
│  │  │                        4.67 Mitem/s  │ 2.198 Mitem/s │ 3.279 Mitem/s │ 3.298 Mitem/s │         │
│  │  ├─ 5                     8.621 ms      │ 22.02 ms      │ 13.37 ms      │ 13.43 ms      │ 100     │ 100
│  │  │                        5.799 Mitem/s │ 2.269 Mitem/s │ 3.737 Mitem/s │ 3.72 Mitem/s  │         │
│  │  ╰─ 6                     8.704 ms      │ 22.53 ms      │ 13.92 ms      │ 14.27 ms      │ 100     │ 100
│  │                           6.892 Mitem/s │ 2.662 Mitem/s │ 4.308 Mitem/s │ 4.203 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.66 ms       │ 10.64 ms      │ 9.036 ms      │ 9.218 ms      │ 100     │ 100
│     │                        1.154 Mitem/s │ 939.4 Kitem/s │ 1.106 Mitem/s │ 1.084 Mitem/s │         │
│     ├─ 2                     8.981 ms      │ 17.06 ms      │ 10.83 ms      │ 11.6 ms       │ 100     │ 100
│     │                        2.226 Mitem/s │ 1.171 Mitem/s │ 1.845 Mitem/s │ 1.723 Mitem/s │         │
│     ├─ 3                     9.319 ms      │ 19.67 ms      │ 13.75 ms      │ 13.98 ms      │ 100     │ 100
│     │                        3.218 Mitem/s │ 1.524 Mitem/s │ 2.18 Mitem/s  │ 2.145 Mitem/s │         │
│     ├─ 4                     9.224 ms      │ 25.57 ms      │ 14.18 ms      │ 14.43 ms      │ 100     │ 100
│     │                        4.336 Mitem/s │ 1.564 Mitem/s │ 2.819 Mitem/s │ 2.77 Mitem/s  │         │
│     ├─ 5                     9.42 ms       │ 27.43 ms      │ 16.51 ms      │ 16.42 ms      │ 100     │ 100
│     │                        5.307 Mitem/s │ 1.822 Mitem/s │ 3.027 Mitem/s │ 3.044 Mitem/s │         │
│     ╰─ 6                     9.814 ms      │ 29.22 ms      │ 17.25 ms      │ 18.28 ms      │ 100     │ 100
│                              6.113 Mitem/s │ 2.052 Mitem/s │ 3.476 Mitem/s │ 3.281 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.557 ms      │ 9.333 ms      │ 4.861 ms      │ 5.412 ms      │ 100     │ 100
│  │  │                        2.194 Mitem/s │ 1.071 Mitem/s │ 2.057 Mitem/s │ 1.847 Mitem/s │         │
│  │  ├─ 2                     11.9 ms       │ 23.74 ms      │ 15.27 ms      │ 15.23 ms      │ 100     │ 100
│  │  │                        1.68 Mitem/s  │ 842.2 Kitem/s │ 1.309 Mitem/s │ 1.313 Mitem/s │         │
│  │  ├─ 3                     21.64 ms      │ 38.17 ms      │ 26.97 ms      │ 27.08 ms      │ 100     │ 100
│  │  │                        1.385 Mitem/s │ 785.8 Kitem/s │ 1.112 Mitem/s │ 1.107 Mitem/s │         │
│  │  ├─ 4                     35.46 ms      │ 57.41 ms      │ 41.31 ms      │ 41.97 ms      │ 100     │ 100
│  │  │                        1.127 Mitem/s │ 696.6 Kitem/s │ 968.2 Kitem/s │ 952.9 Kitem/s │         │
│  │  ├─ 5                     52.84 ms      │ 82.66 ms      │ 59.79 ms      │ 59.96 ms      │ 100     │ 100
│  │  │                        946.1 Kitem/s │ 604.8 Kitem/s │ 836.1 Kitem/s │ 833.7 Kitem/s │         │
│  │  ╰─ 6                     83.45 ms      │ 110.8 ms      │ 98.58 ms      │ 98.13 ms      │ 100     │ 100
│  │                           718.9 Kitem/s │ 541 Kitem/s   │ 608.6 Kitem/s │ 611.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.381 ms      │ 14.16 ms      │ 7.633 ms      │ 8.302 ms      │ 100     │ 100
│  │  │                        1.354 Mitem/s │ 705.8 Kitem/s │ 1.31 Mitem/s  │ 1.204 Mitem/s │         │
│  │  ├─ 2                     7.76 ms       │ 14.42 ms      │ 9.166 ms      │ 9.676 ms      │ 100     │ 100
│  │  │                        2.577 Mitem/s │ 1.386 Mitem/s │ 2.181 Mitem/s │ 2.066 Mitem/s │         │
│  │  ├─ 3                     8.508 ms      │ 16.11 ms      │ 11.52 ms      │ 11.38 ms      │ 100     │ 100
│  │  │                        3.526 Mitem/s │ 1.861 Mitem/s │ 2.603 Mitem/s │ 2.634 Mitem/s │         │
│  │  ├─ 4                     8.637 ms      │ 23.47 ms      │ 12.24 ms      │ 12.64 ms      │ 100     │ 100
│  │  │                        4.631 Mitem/s │ 1.704 Mitem/s │ 3.267 Mitem/s │ 3.162 Mitem/s │         │
│  │  ├─ 5                     8.779 ms      │ 22.43 ms      │ 13.8 ms       │ 13.65 ms      │ 100     │ 100
│  │  │                        5.694 Mitem/s │ 2.228 Mitem/s │ 3.622 Mitem/s │ 3.66 Mitem/s  │         │
│  │  ╰─ 6                     8.799 ms      │ 21.7 ms       │ 13.88 ms      │ 13.53 ms      │ 100     │ 100
│  │                           6.818 Mitem/s │ 2.763 Mitem/s │ 4.32 Mitem/s  │ 4.432 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.633 ms      │ 11.79 ms      │ 9.129 ms      │ 9.417 ms      │ 100     │ 100
│     │                        1.158 Mitem/s │ 847.7 Kitem/s │ 1.095 Mitem/s │ 1.061 Mitem/s │         │
│     ├─ 2                     8.781 ms      │ 16.66 ms      │ 9.853 ms      │ 10.52 ms      │ 100     │ 100
│     │                        2.277 Mitem/s │ 1.2 Mitem/s   │ 2.029 Mitem/s │ 1.9 Mitem/s   │         │
│     ├─ 3                     9.004 ms      │ 18.47 ms      │ 13.61 ms      │ 13.23 ms      │ 100     │ 100
│     │                        3.331 Mitem/s │ 1.623 Mitem/s │ 2.203 Mitem/s │ 2.267 Mitem/s │         │
│     ├─ 4                     9.23 ms       │ 24.54 ms      │ 14.06 ms      │ 14.07 ms      │ 100     │ 100
│     │                        4.333 Mitem/s │ 1.629 Mitem/s │ 2.844 Mitem/s │ 2.842 Mitem/s │         │
│     ├─ 5                     9.418 ms      │ 25.57 ms      │ 16.11 ms      │ 15.82 ms      │ 100     │ 100
│     │                        5.308 Mitem/s │ 1.954 Mitem/s │ 3.102 Mitem/s │ 3.158 Mitem/s │         │
│     ╰─ 6                     9.582 ms      │ 31.02 ms      │ 17.12 ms      │ 19.23 ms      │ 100     │ 100
│                              6.261 Mitem/s │ 1.933 Mitem/s │ 3.503 Mitem/s │ 3.119 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.62 ms       │ 7.52 ms       │ 4.797 ms      │ 5.13 ms       │ 100     │ 100
│  │  │                        2.164 Mitem/s │ 1.329 Mitem/s │ 2.084 Mitem/s │ 1.949 Mitem/s │         │
│  │  ├─ 2                     12.01 ms      │ 19.91 ms      │ 14.19 ms      │ 14.82 ms      │ 100     │ 100
│  │  │                        1.664 Mitem/s │ 1.004 Mitem/s │ 1.409 Mitem/s │ 1.348 Mitem/s │         │
│  │  ├─ 3                     22.98 ms      │ 36.13 ms      │ 27.39 ms      │ 27.77 ms      │ 100     │ 100
│  │  │                        1.305 Mitem/s │ 830.1 Kitem/s │ 1.095 Mitem/s │ 1.08 Mitem/s  │         │
│  │  ├─ 4                     35.8 ms       │ 52.32 ms      │ 42.43 ms      │ 42.35 ms      │ 100     │ 100
│  │  │                        1.117 Mitem/s │ 764.4 Kitem/s │ 942.6 Kitem/s │ 944.4 Kitem/s │         │
│  │  ├─ 5                     53.49 ms      │ 67.94 ms      │ 59.78 ms      │ 59.35 ms      │ 100     │ 100
│  │  │                        934.6 Kitem/s │ 735.8 Kitem/s │ 836.3 Kitem/s │ 842.3 Kitem/s │         │
│  │  ╰─ 6                     88.88 ms      │ 123.6 ms      │ 103.1 ms      │ 103.4 ms      │ 100     │ 100
│  │                           675 Kitem/s   │ 485.3 Kitem/s │ 581.4 Kitem/s │ 579.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.343 ms      │ 10.46 ms      │ 7.582 ms      │ 7.836 ms      │ 100     │ 100
│  │  │                        1.361 Mitem/s │ 955.3 Kitem/s │ 1.318 Mitem/s │ 1.276 Mitem/s │         │
│  │  ├─ 2                     7.757 ms      │ 15.98 ms      │ 9.753 ms      │ 10.18 ms      │ 100     │ 100
│  │  │                        2.578 Mitem/s │ 1.251 Mitem/s │ 2.05 Mitem/s  │ 1.962 Mitem/s │         │
│  │  ├─ 3                     8.247 ms      │ 16.07 ms      │ 12.01 ms      │ 11.7 ms       │ 100     │ 100
│  │  │                        3.637 Mitem/s │ 1.865 Mitem/s │ 2.497 Mitem/s │ 2.562 Mitem/s │         │
│  │  ├─ 4                     8.549 ms      │ 19.14 ms      │ 12.31 ms      │ 12.11 ms      │ 100     │ 100
│  │  │                        4.678 Mitem/s │ 2.089 Mitem/s │ 3.249 Mitem/s │ 3.302 Mitem/s │         │
│  │  ├─ 5                     8.726 ms      │ 22.11 ms      │ 13.51 ms      │ 12.97 ms      │ 100     │ 100
│  │  │                        5.729 Mitem/s │ 2.26 Mitem/s  │ 3.698 Mitem/s │ 3.853 Mitem/s │         │
│  │  ╰─ 6                     8.868 ms      │ 22.37 ms      │ 13.98 ms      │ 14.23 ms      │ 100     │ 100
│  │                           6.765 Mitem/s │ 2.681 Mitem/s │ 4.29 Mitem/s  │ 4.216 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.627 ms      │ 12.47 ms      │ 9.031 ms      │ 9.442 ms      │ 100     │ 100
│     │                        1.159 Mitem/s │ 801.5 Kitem/s │ 1.107 Mitem/s │ 1.059 Mitem/s │         │
│     ├─ 2                     8.859 ms      │ 19 ms         │ 11.28 ms      │ 11.94 ms      │ 100     │ 100
│     │                        2.257 Mitem/s │ 1.052 Mitem/s │ 1.772 Mitem/s │ 1.674 Mitem/s │         │
│     ├─ 3                     9.096 ms      │ 19.36 ms      │ 13.7 ms       │ 13.63 ms      │ 100     │ 100
│     │                        3.297 Mitem/s │ 1.549 Mitem/s │ 2.189 Mitem/s │ 2.2 Mitem/s   │         │
│     ├─ 4                     9.27 ms       │ 25.44 ms      │ 14.44 ms      │ 14.26 ms      │ 100     │ 100
│     │                        4.314 Mitem/s │ 1.571 Mitem/s │ 2.768 Mitem/s │ 2.804 Mitem/s │         │
│     ├─ 5                     9.487 ms      │ 30.39 ms      │ 16.52 ms      │ 15.92 ms      │ 100     │ 100
│     │                        5.27 Mitem/s  │ 1.645 Mitem/s │ 3.025 Mitem/s │ 3.139 Mitem/s │         │
│     ╰─ 6                     9.829 ms      │ 27.4 ms       │ 17.11 ms      │ 18.57 ms      │ 100     │ 100
│                              6.103 Mitem/s │ 2.189 Mitem/s │ 3.506 Mitem/s │ 3.229 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.144 ms      │ 7.11 ms       │ 4.294 ms      │ 4.534 ms      │ 100     │ 100
│  │  │                        2.412 Mitem/s │ 1.406 Mitem/s │ 2.328 Mitem/s │ 2.205 Mitem/s │         │
│  │  ├─ 2                     9.747 ms      │ 18.34 ms      │ 12.18 ms      │ 12.74 ms      │ 100     │ 100
│  │  │                        2.051 Mitem/s │ 1.09 Mitem/s  │ 1.641 Mitem/s │ 1.569 Mitem/s │         │
│  │  ├─ 3                     19.75 ms      │ 31.79 ms      │ 23.16 ms      │ 23.99 ms      │ 100     │ 100
│  │  │                        1.518 Mitem/s │ 943.4 Kitem/s │ 1.294 Mitem/s │ 1.25 Mitem/s  │         │
│  │  ├─ 4                     31.18 ms      │ 44.62 ms      │ 35.64 ms      │ 35.81 ms      │ 100     │ 100
│  │  │                        1.282 Mitem/s │ 896.3 Kitem/s │ 1.122 Mitem/s │ 1.116 Mitem/s │         │
│  │  ├─ 5                     45.69 ms      │ 58.82 ms      │ 52.62 ms      │ 52.3 ms       │ 100     │ 100
│  │  │                        1.094 Mitem/s │ 849.9 Kitem/s │ 950.2 Kitem/s │ 955.8 Kitem/s │         │
│  │  ╰─ 6                     76.9 ms       │ 99.56 ms      │ 90.4 ms       │ 90.48 ms      │ 100     │ 100
│  │                           780.1 Kitem/s │ 602.6 Kitem/s │ 663.7 Kitem/s │ 663.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.427 ms      │ 10.83 ms      │ 7.779 ms      │ 8.278 ms      │ 100     │ 100
│  │  │                        1.346 Mitem/s │ 922.9 Kitem/s │ 1.285 Mitem/s │ 1.207 Mitem/s │         │
│  │  ├─ 2                     7.754 ms      │ 15.8 ms       │ 8.782 ms      │ 9.701 ms      │ 100     │ 100
│  │  │                        2.579 Mitem/s │ 1.265 Mitem/s │ 2.277 Mitem/s │ 2.061 Mitem/s │         │
│  │  ├─ 3                     7.88 ms       │ 16.21 ms      │ 9.686 ms      │ 10.57 ms      │ 100     │ 100
│  │  │                        3.807 Mitem/s │ 1.849 Mitem/s │ 3.097 Mitem/s │ 2.836 Mitem/s │         │
│  │  ├─ 4                     8.512 ms      │ 21.3 ms       │ 12.1 ms       │ 11.75 ms      │ 100     │ 100
│  │  │                        4.698 Mitem/s │ 1.877 Mitem/s │ 3.304 Mitem/s │ 3.402 Mitem/s │         │
│  │  ├─ 5                     8.725 ms      │ 24.26 ms      │ 13.75 ms      │ 13.35 ms      │ 100     │ 100
│  │  │                        5.73 Mitem/s  │ 2.06 Mitem/s  │ 3.635 Mitem/s │ 3.745 Mitem/s │         │
│  │  ╰─ 6                     8.86 ms       │ 23.64 ms      │ 14 ms         │ 14.48 ms      │ 100     │ 100
│  │                           6.771 Mitem/s │ 2.537 Mitem/s │ 4.285 Mitem/s │ 4.143 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.731 ms      │ 13.08 ms      │ 9.273 ms      │ 9.604 ms      │ 100     │ 100
│     │                        1.145 Mitem/s │ 764.1 Kitem/s │ 1.078 Mitem/s │ 1.041 Mitem/s │         │
│     ├─ 2                     8.876 ms      │ 17.86 ms      │ 10.54 ms      │ 11.37 ms      │ 100     │ 100
│     │                        2.253 Mitem/s │ 1.119 Mitem/s │ 1.895 Mitem/s │ 1.757 Mitem/s │         │
│     ├─ 3                     9.429 ms      │ 19.04 ms      │ 13.42 ms      │ 13.63 ms      │ 100     │ 100
│     │                        3.181 Mitem/s │ 1.575 Mitem/s │ 2.234 Mitem/s │ 2.2 Mitem/s   │         │
│     ├─ 4                     9.187 ms      │ 29.51 ms      │ 13.76 ms      │ 14.23 ms      │ 100     │ 100
│     │                        4.353 Mitem/s │ 1.355 Mitem/s │ 2.906 Mitem/s │ 2.809 Mitem/s │         │
│     ├─ 5                     9.216 ms      │ 30.39 ms      │ 17.08 ms      │ 17.24 ms      │ 100     │ 100
│     │                        5.425 Mitem/s │ 1.644 Mitem/s │ 2.927 Mitem/s │ 2.9 Mitem/s   │         │
│     ╰─ 6                     12.36 ms      │ 26.68 ms      │ 17.57 ms      │ 18.71 ms      │ 100     │ 100
│                              4.852 Mitem/s │ 2.248 Mitem/s │ 3.413 Mitem/s │ 3.206 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 2                     6.597 ms      │ 12.39 ms      │ 10.44 ms      │ 10.24 ms      │ 100     │ 100
│  │  ├─ 4                     10.14 ms      │ 18.92 ms      │ 14.09 ms      │ 13.58 ms      │ 100     │ 100
│  │  ├─ 8                     13.71 ms      │ 25.79 ms      │ 15.58 ms      │ 17.82 ms      │ 100     │ 100
│  │  ├─ 16                    21.47 ms      │ 29.7 ms       │ 23.66 ms      │ 23.84 ms      │ 100     │ 100
│  │  ╰─ 32                    40.48 ms      │ 51 ms         │ 42.78 ms      │ 43.11 ms      │ 100     │ 100
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 2                     12.77 ms      │ 23.87 ms      │ 14.32 ms      │ 15.15 ms      │ 100     │ 100
│     ├─ 4                     12.74 ms      │ 22.32 ms      │ 16.96 ms      │ 16.77 ms      │ 100     │ 100
│     ├─ 8                     16.99 ms      │ 35.68 ms      │ 18.44 ms      │ 20.66 ms      │ 100     │ 100
│     ├─ 16                    25.07 ms      │ 63.14 ms      │ 28.03 ms      │ 28.69 ms      │ 100     │ 100
│     ╰─ 32                    48.45 ms      │ 58.39 ms      │ 51.16 ms      │ 51.65 ms      │ 100     │ 100
├─ 14_prefix_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     23.12 ms      │ 25.89 ms      │ 23.75 ms      │ 24.03 ms      │ 20      │ 20
│     │                        4.324 Kitem/s │ 3.862 Kitem/s │ 4.209 Kitem/s │ 4.16 Kitem/s  │         │
│     ├─ 2                     24.49 ms      │ 40.52 ms      │ 39.35 ms      │ 37.68 ms      │ 20      │ 20
│     │                        8.164 Kitem/s │ 4.934 Kitem/s │ 5.082 Kitem/s │ 5.307 Kitem/s │         │
│     ├─ 3                     24.87 ms      │ 36.99 ms      │ 35.13 ms      │ 34.5 ms       │ 20      │ 20
│     │                        12.05 Kitem/s │ 8.109 Kitem/s │ 8.538 Kitem/s │ 8.695 Kitem/s │         │
│     ├─ 4                     33.65 ms      │ 42.5 ms       │ 34.97 ms      │ 36.55 ms      │ 20      │ 20
│     │                        11.88 Kitem/s │ 9.411 Kitem/s │ 11.43 Kitem/s │ 10.94 Kitem/s │         │
│     ├─ 5                     33.78 ms      │ 54.14 ms      │ 40.14 ms      │ 39.83 ms      │ 20      │ 20
│     │                        14.79 Kitem/s │ 9.234 Kitem/s │ 12.45 Kitem/s │ 12.55 Kitem/s │         │
│     ╰─ 6                     36.58 ms      │ 51.94 ms      │ 42.32 ms      │ 43.15 ms      │ 20      │ 20
│                              16.39 Kitem/s │ 11.55 Kitem/s │ 14.17 Kitem/s │ 13.9 Kitem/s  │         │
╰─ 15_full_scan_aggregate                    │               │               │               │         │
   ├─ indexset                               │               │               │               │         │
   │  ├─ 1                     302.9 ms      │ 335.8 ms      │ 307.4 ms      │ 308.7 ms      │ 100     │ 100
   │  │                        3.3 Kitem/s   │ 2.977 Kitem/s │ 3.252 Kitem/s │ 3.239 Kitem/s │         │
   │  ├─ 2                     688.8 ms      │ 1.139 s       │ 993.7 ms      │ 976.9 ms      │ 100     │ 100
   │  │                        2.903 Kitem/s │ 1.755 Kitem/s │ 2.012 Kitem/s │ 2.047 Kitem/s │         │
   │  ├─ 3                     1.443 s       │ 1.746 s       │ 1.647 s       │ 1.634 s       │ 100     │ 100
   │  │                        2.078 Kitem/s │ 1.717 Kitem/s │ 1.821 Kitem/s │ 1.835 Kitem/s │         │
   │  ├─ 4                     1.71 s        │ 2.339 s       │ 2.207 s       │ 2.179 s       │ 100     │ 100
   │  │                        2.338 Kitem/s │ 1.709 Kitem/s │ 1.812 Kitem/s │ 1.835 Kitem/s │         │
   │  ├─ 5                     2.538 s       │ 2.909 s       │ 2.775 s       │ 2.765 s       │ 100     │ 100
   │  │                        1.969 Kitem/s │ 1.718 Kitem/s │ 1.801 Kitem/s │ 1.807 Kitem/s │         │
   │  ╰─ 6                     2.585 s       │ 3.613 s       │ 3.354 s       │ 3.326 s       │ 100     │ 100
   │                           2.32 Kitem/s  │ 1.66 Kitem/s  │ 1.788 Kitem/s │ 1.803 Kitem/s │         │
   ├─ masstree24                             │               │               │               │         │
   │  ├─ 1                     811.8 ms      │ 990.9 ms      │ 880.3 ms      │ 882 ms        │ 100     │ 100
   │  │                        1.231 Kitem/s │ 1.009 Kitem/s │ 1.135 Kitem/s │ 1.133 Kitem/s │         │
   │  ├─ 2                     821.5 ms      │ 1.096 s       │ 867.6 ms      │ 882.8 ms      │ 100     │ 100
   │  │                        2.434 Kitem/s │ 1.823 Kitem/s │ 2.305 Kitem/s │ 2.265 Kitem/s │         │
   │  ├─ 3                     882.7 ms      │ 1.139 s       │ 938.6 ms      │ 954.8 ms      │ 100     │ 100
   │  │                        3.398 Kitem/s │ 2.631 Kitem/s │ 3.196 Kitem/s │ 3.141 Kitem/s │         │
   │  ├─ 4                     895 ms        │ 1.311 s       │ 975.3 ms      │ 996.1 ms      │ 100     │ 100
   │  │                        4.469 Kitem/s │ 3.05 Kitem/s  │ 4.1 Kitem/s   │ 4.015 Kitem/s │         │
   │  ├─ 5                     930.8 ms      │ 1.324 s       │ 991.1 ms      │ 1.037 s       │ 100     │ 100
   │  │                        5.371 Kitem/s │ 3.775 Kitem/s │ 5.044 Kitem/s │ 4.818 Kitem/s │         │
   │  ╰─ 6                     900 ms        │ 1.506 s       │ 1.049 s       │ 1.109 s       │ 100     │ 100
   │                           6.666 Kitem/s │ 3.981 Kitem/s │ 5.717 Kitem/s │ 5.406 Kitem/s │         │
   ╰─ tree_index                             │               │               │               │         │
      ├─ 1                     864.1 ms      │ 993.8 ms      │ 883.8 ms      │ 886.7 ms      │ 100     │ 100
      │                        1.157 Kitem/s │ 1.006 Kitem/s │ 1.131 Kitem/s │ 1.127 Kitem/s │         │
      ├─ 2                     875.3 ms      │ 1.037 s       │ 890.7 ms      │ 895.9 ms      │ 100     │ 100
      │                        2.284 Kitem/s │ 1.927 Kitem/s │ 2.245 Kitem/s │ 2.232 Kitem/s │         │
      ├─ 3                     893.5 ms      │ 1.091 s       │ 929.6 ms      │ 942.4 ms      │ 100     │ 100
      │                        3.357 Kitem/s │ 2.749 Kitem/s │ 3.226 Kitem/s │ 3.183 Kitem/s │         │
      ├─ 4                     913.6 ms      │ 1.028 s       │ 946.3 ms      │ 953.2 ms      │ 100     │ 100
      │                        4.378 Kitem/s │ 3.89 Kitem/s  │ 4.226 Kitem/s │ 4.196 Kitem/s │         │
      ├─ 5                     922.4 ms      │ 1.121 s       │ 951.1 ms      │ 965.6 ms      │ 100     │ 100
      │                        5.42 Kitem/s  │ 4.457 Kitem/s │ 5.256 Kitem/s │ 5.177 Kitem/s │         │
      ╰─ 6                     932.1 ms      │ 1.16 s        │ 954.8 ms      │ 964.5 ms      │ 100     │ 100
                               6.436 Kitem/s │ 5.171 Kitem/s │ 6.283 Kitem/s │ 6.22 Kitem/s  │         │
```
