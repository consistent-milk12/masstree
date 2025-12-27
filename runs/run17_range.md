```text
Timer precision: 20 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.486 ms      │ 6.928 ms      │ 4.552 ms      │ 4.625 ms      │ 100     │ 100
│  │  │                        2.228 Mitem/s │ 1.443 Mitem/s │ 2.196 Mitem/s │ 2.162 Mitem/s │         │
│  │  ├─ 2                     10.72 ms      │ 19.94 ms      │ 15.8 ms       │ 15.63 ms      │ 100     │ 100
│  │  │                        1.864 Mitem/s │ 1.002 Mitem/s │ 1.265 Mitem/s │ 1.278 Mitem/s │         │
│  │  ├─ 3                     21.82 ms      │ 37.18 ms      │ 30.02 ms      │ 29.48 ms      │ 100     │ 100
│  │  │                        1.374 Mitem/s │ 806.7 Kitem/s │ 999 Kitem/s   │ 1.017 Mitem/s │         │
│  │  ├─ 4                     36.42 ms      │ 57.28 ms      │ 45.8 ms       │ 46 ms         │ 100     │ 100
│  │  │                        1.098 Mitem/s │ 698.2 Kitem/s │ 873.3 Kitem/s │ 869.5 Kitem/s │         │
│  │  ├─ 5                     52.35 ms      │ 89.49 ms      │ 71.26 ms      │ 71.15 ms      │ 100     │ 100
│  │  │                        954.9 Kitem/s │ 558.6 Kitem/s │ 701.5 Kitem/s │ 702.7 Kitem/s │         │
│  │  ╰─ 6                     83.67 ms      │ 151.7 ms      │ 103.9 ms      │ 106 ms        │ 100     │ 100
│  │                           717 Kitem/s   │ 395.3 Kitem/s │ 577.4 Kitem/s │ 565.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     8.94 ms       │ 16.14 ms      │ 9.247 ms      │ 10.6 ms       │ 100     │ 100
│  │  │                        1.118 Mitem/s │ 619.2 Kitem/s │ 1.081 Mitem/s │ 943.3 Kitem/s │         │
│  │  ├─ 2                     9.404 ms      │ 19.84 ms      │ 13.95 ms      │ 14.24 ms      │ 100     │ 100
│  │  │                        2.126 Mitem/s │ 1.007 Mitem/s │ 1.432 Mitem/s │ 1.403 Mitem/s │         │
│  │  ├─ 3                     11.27 ms      │ 20.22 ms      │ 14.54 ms      │ 16.09 ms      │ 100     │ 100
│  │  │                        2.66 Mitem/s  │ 1.483 Mitem/s │ 2.062 Mitem/s │ 1.863 Mitem/s │         │
│  │  ├─ 4                     10.16 ms      │ 23.52 ms      │ 17.5 ms       │ 16.99 ms      │ 100     │ 100
│  │  │                        3.935 Mitem/s │ 1.7 Mitem/s   │ 2.285 Mitem/s │ 2.353 Mitem/s │         │
│  │  ├─ 5                     10.62 ms      │ 30.77 ms      │ 18.74 ms      │ 19 ms         │ 100     │ 100
│  │  │                        4.704 Mitem/s │ 1.624 Mitem/s │ 2.667 Mitem/s │ 2.631 Mitem/s │         │
│  │  ╰─ 6                     10.5 ms       │ 32.22 ms      │ 17.69 ms      │ 18.84 ms      │ 100     │ 100
│  │                           5.709 Mitem/s │ 1.861 Mitem/s │ 3.39 Mitem/s  │ 3.183 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.392 ms      │ 12.79 ms      │ 8.826 ms      │ 9.075 ms      │ 100     │ 100
│     │                        1.191 Mitem/s │ 781.6 Kitem/s │ 1.132 Mitem/s │ 1.101 Mitem/s │         │
│     ├─ 2                     8.656 ms      │ 19.9 ms       │ 13.07 ms      │ 13.82 ms      │ 100     │ 100
│     │                        2.31 Mitem/s  │ 1.004 Mitem/s │ 1.529 Mitem/s │ 1.446 Mitem/s │         │
│     ├─ 3                     8.709 ms      │ 21.11 ms      │ 17.11 ms      │ 15.97 ms      │ 100     │ 100
│     │                        3.444 Mitem/s │ 1.42 Mitem/s  │ 1.753 Mitem/s │ 1.877 Mitem/s │         │
│     ├─ 4                     8.996 ms      │ 27.4 ms       │ 17.35 ms      │ 16.48 ms      │ 100     │ 100
│     │                        4.446 Mitem/s │ 1.459 Mitem/s │ 2.305 Mitem/s │ 2.427 Mitem/s │         │
│     ├─ 5                     8.834 ms      │ 33.55 ms      │ 16.47 ms      │ 16.87 ms      │ 100     │ 100
│     │                        5.659 Mitem/s │ 1.489 Mitem/s │ 3.034 Mitem/s │ 2.963 Mitem/s │         │
│     ╰─ 6                     10.9 ms       │ 30.35 ms      │ 17.41 ms      │ 18.52 ms      │ 100     │ 100
│                              5.502 Mitem/s │ 1.976 Mitem/s │ 3.445 Mitem/s │ 3.238 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.579 ms      │ 7.716 ms      │ 4.649 ms      │ 4.889 ms      │ 100     │ 100
│  │  │                        2.183 Mitem/s │ 1.295 Mitem/s │ 2.15 Mitem/s  │ 2.045 Mitem/s │         │
│  │  ├─ 2                     11.09 ms      │ 22.23 ms      │ 15.82 ms      │ 16.04 ms      │ 100     │ 100
│  │  │                        1.802 Mitem/s │ 899.5 Kitem/s │ 1.263 Mitem/s │ 1.246 Mitem/s │         │
│  │  ├─ 3                     22.67 ms      │ 38.42 ms      │ 31.53 ms      │ 31.33 ms      │ 100     │ 100
│  │  │                        1.323 Mitem/s │ 780.7 Kitem/s │ 951.3 Kitem/s │ 957.3 Kitem/s │         │
│  │  ├─ 4                     37.98 ms      │ 58.47 ms      │ 47.89 ms      │ 47.54 ms      │ 100     │ 100
│  │  │                        1.053 Mitem/s │ 684.1 Kitem/s │ 835 Kitem/s   │ 841.2 Kitem/s │         │
│  │  ├─ 5                     59.28 ms      │ 94.27 ms      │ 78.06 ms      │ 77.75 ms      │ 100     │ 100
│  │  │                        843.3 Kitem/s │ 530.3 Kitem/s │ 640.5 Kitem/s │ 643 Kitem/s   │         │
│  │  ╰─ 6                     88.13 ms      │ 148.7 ms      │ 110.8 ms      │ 111.3 ms      │ 100     │ 100
│  │                           680.7 Kitem/s │ 403.3 Kitem/s │ 541.3 Kitem/s │ 538.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     8.955 ms      │ 11.69 ms      │ 9.105 ms      │ 9.303 ms      │ 100     │ 100
│  │  │                        1.116 Mitem/s │ 855 Kitem/s   │ 1.098 Mitem/s │ 1.074 Mitem/s │         │
│  │  ├─ 2                     9.34 ms       │ 19.27 ms      │ 13.38 ms      │ 13.66 ms      │ 100     │ 100
│  │  │                        2.141 Mitem/s │ 1.037 Mitem/s │ 1.494 Mitem/s │ 1.463 Mitem/s │         │
│  │  ├─ 3                     10.12 ms      │ 22.12 ms      │ 15.12 ms      │ 15.77 ms      │ 100     │ 100
│  │  │                        2.964 Mitem/s │ 1.355 Mitem/s │ 1.983 Mitem/s │ 1.901 Mitem/s │         │
│  │  ├─ 4                     10.16 ms      │ 28.92 ms      │ 14.83 ms      │ 15.97 ms      │ 100     │ 100
│  │  │                        3.934 Mitem/s │ 1.383 Mitem/s │ 2.695 Mitem/s │ 2.504 Mitem/s │         │
│  │  ├─ 5                     10.21 ms      │ 29.98 ms      │ 16.88 ms      │ 17.83 ms      │ 100     │ 100
│  │  │                        4.895 Mitem/s │ 1.667 Mitem/s │ 2.962 Mitem/s │ 2.803 Mitem/s │         │
│  │  ╰─ 6                     14.13 ms      │ 30.09 ms      │ 22.31 ms      │ 21.34 ms      │ 100     │ 100
│  │                           4.246 Mitem/s │ 1.993 Mitem/s │ 2.688 Mitem/s │ 2.81 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.814 ms      │ 13.78 ms      │ 9.079 ms      │ 9.85 ms       │ 100     │ 100
│     │                        1.134 Mitem/s │ 725.6 Kitem/s │ 1.101 Mitem/s │ 1.015 Mitem/s │         │
│     ├─ 2                     9.062 ms      │ 19.55 ms      │ 13.33 ms      │ 14.03 ms      │ 100     │ 100
│     │                        2.206 Mitem/s │ 1.022 Mitem/s │ 1.499 Mitem/s │ 1.424 Mitem/s │         │
│     ├─ 3                     11.22 ms      │ 21.8 ms       │ 16.1 ms       │ 15.88 ms      │ 100     │ 100
│     │                        2.673 Mitem/s │ 1.375 Mitem/s │ 1.862 Mitem/s │ 1.888 Mitem/s │         │
│     ├─ 4                     9.136 ms      │ 21.71 ms      │ 16.52 ms      │ 16.12 ms      │ 100     │ 100
│     │                        4.378 Mitem/s │ 1.842 Mitem/s │ 2.42 Mitem/s  │ 2.48 Mitem/s  │         │
│     ├─ 5                     10.44 ms      │ 33.57 ms      │ 17.52 ms      │ 18.27 ms      │ 100     │ 100
│     │                        4.788 Mitem/s │ 1.489 Mitem/s │ 2.853 Mitem/s │ 2.735 Mitem/s │         │
│     ╰─ 6                     9.541 ms      │ 32.66 ms      │ 17.03 ms      │ 18.57 ms      │ 100     │ 100
│                              6.288 Mitem/s │ 1.836 Mitem/s │ 3.522 Mitem/s │ 3.231 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.434 ms      │ 9.388 ms      │ 4.567 ms      │ 5.32 ms       │ 100     │ 100
│  │  │                        2.254 Mitem/s │ 1.065 Mitem/s │ 2.189 Mitem/s │ 1.879 Mitem/s │         │
│  │  ├─ 2                     10.28 ms      │ 19.82 ms      │ 14.51 ms      │ 14.55 ms      │ 100     │ 100
│  │  │                        1.944 Mitem/s │ 1.009 Mitem/s │ 1.377 Mitem/s │ 1.374 Mitem/s │         │
│  │  ├─ 3                     20.91 ms      │ 37.34 ms      │ 30.31 ms      │ 29.7 ms       │ 100     │ 100
│  │  │                        1.434 Mitem/s │ 803.3 Kitem/s │ 989.7 Kitem/s │ 1.009 Mitem/s │         │
│  │  ├─ 4                     34.7 ms       │ 60.72 ms      │ 45.52 ms      │ 45.55 ms      │ 100     │ 100
│  │  │                        1.152 Mitem/s │ 658.6 Kitem/s │ 878.7 Kitem/s │ 878 Kitem/s   │         │
│  │  ├─ 5                     55.41 ms      │ 89.65 ms      │ 72.3 ms       │ 72.45 ms      │ 100     │ 100
│  │  │                        902.3 Kitem/s │ 557.7 Kitem/s │ 691.5 Kitem/s │ 690 Kitem/s   │         │
│  │  ╰─ 6                     76.8 ms       │ 149.1 ms      │ 99.42 ms      │ 102.6 ms      │ 100     │ 100
│  │                           781.1 Kitem/s │ 402.2 Kitem/s │ 603.4 Kitem/s │ 584.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.078 ms      │ 18.06 ms      │ 9.562 ms      │ 10.85 ms      │ 100     │ 100
│  │  │                        1.101 Mitem/s │ 553.4 Kitem/s │ 1.045 Mitem/s │ 921.3 Kitem/s │         │
│  │  ├─ 2                     10.15 ms      │ 19.88 ms      │ 14.01 ms      │ 14.92 ms      │ 100     │ 100
│  │  │                        1.969 Mitem/s │ 1.005 Mitem/s │ 1.426 Mitem/s │ 1.34 Mitem/s  │         │
│  │  ├─ 3                     10.24 ms      │ 20.45 ms      │ 14.71 ms      │ 16.17 ms      │ 100     │ 100
│  │  │                        2.928 Mitem/s │ 1.466 Mitem/s │ 2.038 Mitem/s │ 1.855 Mitem/s │         │
│  │  ├─ 4                     10.45 ms      │ 29.29 ms      │ 16.9 ms       │ 17.1 ms       │ 100     │ 100
│  │  │                        3.826 Mitem/s │ 1.365 Mitem/s │ 2.366 Mitem/s │ 2.338 Mitem/s │         │
│  │  ├─ 5                     10.44 ms      │ 29.43 ms      │ 18.06 ms      │ 18.11 ms      │ 100     │ 100
│  │  │                        4.787 Mitem/s │ 1.698 Mitem/s │ 2.767 Mitem/s │ 2.759 Mitem/s │         │
│  │  ╰─ 6                     10.4 ms       │ 27.58 ms      │ 16.74 ms      │ 16.96 ms      │ 100     │ 100
│  │                           5.764 Mitem/s │ 2.175 Mitem/s │ 3.582 Mitem/s │ 3.537 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.387 ms      │ 11.02 ms      │ 8.696 ms      │ 8.755 ms      │ 100     │ 100
│     │                        1.192 Mitem/s │ 907.4 Kitem/s │ 1.149 Mitem/s │ 1.142 Mitem/s │         │
│     ├─ 2                     8.526 ms      │ 22.26 ms      │ 12.86 ms      │ 13.17 ms      │ 100     │ 100
│     │                        2.345 Mitem/s │ 898.2 Kitem/s │ 1.554 Mitem/s │ 1.517 Mitem/s │         │
│     ├─ 3                     8.751 ms      │ 19.51 ms      │ 13.63 ms      │ 14.56 ms      │ 100     │ 100
│     │                        3.427 Mitem/s │ 1.537 Mitem/s │ 2.2 Mitem/s   │ 2.059 Mitem/s │         │
│     ├─ 4                     8.935 ms      │ 22.26 ms      │ 16.73 ms      │ 16.07 ms      │ 100     │ 100
│     │                        4.476 Mitem/s │ 1.796 Mitem/s │ 2.39 Mitem/s  │ 2.487 Mitem/s │         │
│     ├─ 5                     8.906 ms      │ 31.5 ms       │ 16.3 ms       │ 16.23 ms      │ 100     │ 100
│     │                        5.613 Mitem/s │ 1.586 Mitem/s │ 3.067 Mitem/s │ 3.079 Mitem/s │         │
│     ╰─ 6                     8.897 ms      │ 31.8 ms       │ 16.22 ms      │ 16.07 ms      │ 100     │ 100
│                              6.743 Mitem/s │ 1.886 Mitem/s │ 3.698 Mitem/s │ 3.732 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.502 ms      │ 6.978 ms      │ 4.568 ms      │ 4.636 ms      │ 100     │ 100
│  │  │                        2.221 Mitem/s │ 1.433 Mitem/s │ 2.188 Mitem/s │ 2.156 Mitem/s │         │
│  │  ├─ 2                     10.62 ms      │ 21.38 ms      │ 15.75 ms      │ 15.35 ms      │ 100     │ 100
│  │  │                        1.883 Mitem/s │ 935.2 Kitem/s │ 1.269 Mitem/s │ 1.302 Mitem/s │         │
│  │  ├─ 3                     23.2 ms       │ 35.92 ms      │ 30.48 ms      │ 30.64 ms      │ 100     │ 100
│  │  │                        1.292 Mitem/s │ 835 Kitem/s   │ 983.9 Kitem/s │ 978.8 Kitem/s │         │
│  │  ├─ 4                     36.23 ms      │ 52.94 ms      │ 46.13 ms      │ 45.63 ms      │ 100     │ 100
│  │  │                        1.103 Mitem/s │ 755.5 Kitem/s │ 867 Kitem/s   │ 876.4 Kitem/s │         │
│  │  ├─ 5                     60.03 ms      │ 87.86 ms      │ 74.77 ms      │ 73.93 ms      │ 100     │ 100
│  │  │                        832.8 Kitem/s │ 569 Kitem/s   │ 668.6 Kitem/s │ 676.2 Kitem/s │         │
│  │  ╰─ 6                     81.33 ms      │ 170 ms        │ 116.6 ms      │ 117.8 ms      │ 100     │ 100
│  │                           737.6 Kitem/s │ 352.9 Kitem/s │ 514.5 Kitem/s │ 509.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.06 ms       │ 11.73 ms      │ 9.195 ms      │ 9.27 ms       │ 100     │ 100
│  │  │                        1.103 Mitem/s │ 851.9 Kitem/s │ 1.087 Mitem/s │ 1.078 Mitem/s │         │
│  │  ├─ 2                     9.501 ms      │ 20.04 ms      │ 13.93 ms      │ 14.06 ms      │ 100     │ 100
│  │  │                        2.105 Mitem/s │ 997.9 Kitem/s │ 1.435 Mitem/s │ 1.422 Mitem/s │         │
│  │  ├─ 3                     9.605 ms      │ 20.16 ms      │ 15.23 ms      │ 16.49 ms      │ 100     │ 100
│  │  │                        3.123 Mitem/s │ 1.487 Mitem/s │ 1.969 Mitem/s │ 1.818 Mitem/s │         │
│  │  ├─ 4                     10.31 ms      │ 29.93 ms      │ 16.9 ms       │ 16.89 ms      │ 100     │ 100
│  │  │                        3.879 Mitem/s │ 1.336 Mitem/s │ 2.366 Mitem/s │ 2.367 Mitem/s │         │
│  │  ├─ 5                     12.07 ms      │ 30.54 ms      │ 17.68 ms      │ 18.26 ms      │ 100     │ 100
│  │  │                        4.139 Mitem/s │ 1.637 Mitem/s │ 2.827 Mitem/s │ 2.737 Mitem/s │         │
│  │  ╰─ 6                     10.38 ms      │ 30.66 ms      │ 16.81 ms      │ 17.44 ms      │ 100     │ 100
│  │                           5.777 Mitem/s │ 1.956 Mitem/s │ 3.567 Mitem/s │ 3.439 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.38 ms       │ 12.66 ms      │ 8.59 ms       │ 8.896 ms      │ 100     │ 100
│     │                        1.193 Mitem/s │ 789.4 Kitem/s │ 1.164 Mitem/s │ 1.123 Mitem/s │         │
│     ├─ 2                     8.566 ms      │ 18.29 ms      │ 12.91 ms      │ 13.81 ms      │ 100     │ 100
│     │                        2.334 Mitem/s │ 1.093 Mitem/s │ 1.548 Mitem/s │ 1.447 Mitem/s │         │
│     ├─ 3                     8.833 ms      │ 18.9 ms       │ 13.88 ms      │ 14.68 ms      │ 100     │ 100
│     │                        3.396 Mitem/s │ 1.586 Mitem/s │ 2.16 Mitem/s  │ 2.042 Mitem/s │         │
│     ├─ 4                     8.862 ms      │ 29.53 ms      │ 15.48 ms      │ 15.17 ms      │ 100     │ 100
│     │                        4.513 Mitem/s │ 1.354 Mitem/s │ 2.582 Mitem/s │ 2.635 Mitem/s │         │
│     ├─ 5                     12.06 ms      │ 25.67 ms      │ 15.48 ms      │ 15.71 ms      │ 100     │ 100
│     │                        4.143 Mitem/s │ 1.947 Mitem/s │ 3.229 Mitem/s │ 3.182 Mitem/s │         │
│     ╰─ 6                     11.18 ms      │ 31.29 ms      │ 16.19 ms      │ 18.08 ms      │ 100     │ 100
│                              5.364 Mitem/s │ 1.917 Mitem/s │ 3.704 Mitem/s │ 3.317 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.072 ms      │ 5.698 ms      │ 3.16 ms       │ 3.424 ms      │ 100     │ 100
│  │  │                        3.255 Mitem/s │ 1.754 Mitem/s │ 3.164 Mitem/s │ 2.92 Mitem/s  │         │
│  │  ├─ 2                     8.622 ms      │ 13.96 ms      │ 11.89 ms      │ 11.55 ms      │ 100     │ 100
│  │  │                        2.319 Mitem/s │ 1.431 Mitem/s │ 1.681 Mitem/s │ 1.731 Mitem/s │         │
│  │  ├─ 3                     17.21 ms      │ 25.1 ms       │ 20.22 ms      │ 20.92 ms      │ 100     │ 100
│  │  │                        1.742 Mitem/s │ 1.195 Mitem/s │ 1.483 Mitem/s │ 1.433 Mitem/s │         │
│  │  ├─ 4                     27.24 ms      │ 38.73 ms      │ 31.17 ms      │ 31.58 ms      │ 100     │ 100
│  │  │                        1.468 Mitem/s │ 1.032 Mitem/s │ 1.282 Mitem/s │ 1.266 Mitem/s │         │
│  │  ├─ 5                     35.49 ms      │ 59.99 ms      │ 50.3 ms       │ 49.75 ms      │ 100     │ 100
│  │  │                        1.408 Mitem/s │ 833.3 Kitem/s │ 993.9 Kitem/s │ 1.004 Mitem/s │         │
│  │  ╰─ 6                     57.82 ms      │ 118.6 ms      │ 85.7 ms       │ 85.15 ms      │ 100     │ 100
│  │                           1.037 Mitem/s │ 505.6 Kitem/s │ 700.1 Kitem/s │ 704.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     8.36 ms       │ 10.91 ms      │ 8.477 ms      │ 8.538 ms      │ 100     │ 100
│  │  │                        1.196 Mitem/s │ 915.8 Kitem/s │ 1.179 Mitem/s │ 1.171 Mitem/s │         │
│  │  ├─ 2                     8.844 ms      │ 18.56 ms      │ 13.24 ms      │ 13.82 ms      │ 100     │ 100
│  │  │                        2.261 Mitem/s │ 1.077 Mitem/s │ 1.509 Mitem/s │ 1.446 Mitem/s │         │
│  │  ├─ 3                     9.562 ms      │ 18.73 ms      │ 16.03 ms      │ 15.31 ms      │ 100     │ 100
│  │  │                        3.137 Mitem/s │ 1.601 Mitem/s │ 1.871 Mitem/s │ 1.958 Mitem/s │         │
│  │  ├─ 4                     10.04 ms      │ 27.1 ms       │ 16.9 ms       │ 16.08 ms      │ 100     │ 100
│  │  │                        3.98 Mitem/s  │ 1.475 Mitem/s │ 2.366 Mitem/s │ 2.487 Mitem/s │         │
│  │  ├─ 5                     9.682 ms      │ 28.32 ms      │ 16.73 ms      │ 16.57 ms      │ 100     │ 100
│  │  │                        5.164 Mitem/s │ 1.765 Mitem/s │ 2.986 Mitem/s │ 3.017 Mitem/s │         │
│  │  ╰─ 6                     9.626 ms      │ 26.57 ms      │ 15.52 ms      │ 15.44 ms      │ 100     │ 100
│  │                           6.232 Mitem/s │ 2.257 Mitem/s │ 3.865 Mitem/s │ 3.884 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.422 ms      │ 11.27 ms      │ 8.558 ms      │ 8.751 ms      │ 100     │ 100
│     │                        1.187 Mitem/s │ 887.1 Kitem/s │ 1.168 Mitem/s │ 1.142 Mitem/s │         │
│     ├─ 2                     8.468 ms      │ 18.86 ms      │ 12.25 ms      │ 13.29 ms      │ 100     │ 100
│     │                        2.361 Mitem/s │ 1.06 Mitem/s  │ 1.631 Mitem/s │ 1.504 Mitem/s │         │
│     ├─ 3                     8.702 ms      │ 17.97 ms      │ 15.27 ms      │ 14.95 ms      │ 100     │ 100
│     │                        3.447 Mitem/s │ 1.668 Mitem/s │ 1.964 Mitem/s │ 2.005 Mitem/s │         │
│     ├─ 4                     11.6 ms       │ 21.68 ms      │ 17.04 ms      │ 16.54 ms      │ 100     │ 100
│     │                        3.446 Mitem/s │ 1.844 Mitem/s │ 2.346 Mitem/s │ 2.418 Mitem/s │         │
│     ├─ 5                     9.41 ms       │ 29.07 ms      │ 16.67 ms      │ 16.26 ms      │ 100     │ 100
│     │                        5.313 Mitem/s │ 1.719 Mitem/s │ 2.999 Mitem/s │ 3.073 Mitem/s │         │
│     ╰─ 6                     8.693 ms      │ 28.98 ms      │ 15.11 ms      │ 15.19 ms      │ 100     │ 100
│                              6.901 Mitem/s │ 2.07 Mitem/s  │ 3.969 Mitem/s │ 3.947 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.185 ms      │ 5.286 ms      │ 3.241 ms      │ 3.28 ms       │ 100     │ 100
│  │  │                        3.139 Mitem/s │ 1.891 Mitem/s │ 3.085 Mitem/s │ 3.048 Mitem/s │         │
│  │  ├─ 2                     7.494 ms      │ 15.02 ms      │ 10.22 ms      │ 10.18 ms      │ 100     │ 100
│  │  │                        2.668 Mitem/s │ 1.331 Mitem/s │ 1.956 Mitem/s │ 1.963 Mitem/s │         │
│  │  ├─ 3                     16.96 ms      │ 26.09 ms      │ 21.37 ms      │ 21.8 ms       │ 100     │ 100
│  │  │                        1.768 Mitem/s │ 1.149 Mitem/s │ 1.403 Mitem/s │ 1.375 Mitem/s │         │
│  │  ├─ 4                     28.92 ms      │ 41.87 ms      │ 33.63 ms      │ 33.36 ms      │ 100     │ 100
│  │  │                        1.383 Mitem/s │ 955.2 Kitem/s │ 1.189 Mitem/s │ 1.198 Mitem/s │         │
│  │  ├─ 5                     48.59 ms      │ 62.64 ms      │ 57.2 ms       │ 56.72 ms      │ 100     │ 100
│  │  │                        1.028 Mitem/s │ 798.2 Kitem/s │ 873.9 Kitem/s │ 881.3 Kitem/s │         │
│  │  ╰─ 6                     70.84 ms      │ 121 ms        │ 89.64 ms      │ 89.84 ms      │ 100     │ 100
│  │                           846.9 Kitem/s │ 495.5 Kitem/s │ 669.3 Kitem/s │ 667.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.39 ms       │ 12.65 ms      │ 9.63 ms       │ 9.91 ms       │ 100     │ 100
│  │  │                        1.064 Mitem/s │ 790.2 Kitem/s │ 1.038 Mitem/s │ 1.008 Mitem/s │         │
│  │  ├─ 2                     9.599 ms      │ 21.26 ms      │ 14.41 ms      │ 15.03 ms      │ 100     │ 100
│  │  │                        2.083 Mitem/s │ 940.3 Kitem/s │ 1.386 Mitem/s │ 1.33 Mitem/s  │         │
│  │  ├─ 3                     9.71 ms       │ 20.71 ms      │ 14.51 ms      │ 15.8 ms       │ 100     │ 100
│  │  │                        3.089 Mitem/s │ 1.448 Mitem/s │ 2.066 Mitem/s │ 1.898 Mitem/s │         │
│  │  ├─ 4                     11.73 ms      │ 28.36 ms      │ 15.3 ms       │ 16.57 ms      │ 100     │ 100
│  │  │                        3.407 Mitem/s │ 1.41 Mitem/s  │ 2.612 Mitem/s │ 2.413 Mitem/s │         │
│  │  ├─ 5                     9.993 ms      │ 32.44 ms      │ 17.03 ms      │ 17.7 ms       │ 100     │ 100
│  │  │                        5.003 Mitem/s │ 1.54 Mitem/s  │ 2.935 Mitem/s │ 2.823 Mitem/s │         │
│  │  ╰─ 6                     12.19 ms      │ 29.53 ms      │ 17.05 ms      │ 17.25 ms      │ 100     │ 100
│  │                           4.92 Mitem/s  │ 2.031 Mitem/s │ 3.517 Mitem/s │ 3.477 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.97 ms       │ 17.39 ms      │ 9.215 ms      │ 9.548 ms      │ 100     │ 100
│     │                        1.114 Mitem/s │ 574.7 Kitem/s │ 1.085 Mitem/s │ 1.047 Mitem/s │         │
│     ├─ 2                     9.174 ms      │ 20.13 ms      │ 13.03 ms      │ 14.01 ms      │ 100     │ 100
│     │                        2.179 Mitem/s │ 993.1 Kitem/s │ 1.533 Mitem/s │ 1.426 Mitem/s │         │
│     ├─ 3                     9.295 ms      │ 20.62 ms      │ 14.23 ms      │ 15.57 ms      │ 100     │ 100
│     │                        3.227 Mitem/s │ 1.454 Mitem/s │ 2.107 Mitem/s │ 1.926 Mitem/s │         │
│     ├─ 4                     9.561 ms      │ 30.14 ms      │ 16.43 ms      │ 16.12 ms      │ 100     │ 100
│     │                        4.183 Mitem/s │ 1.326 Mitem/s │ 2.433 Mitem/s │ 2.48 Mitem/s  │         │
│     ├─ 5                     9.565 ms      │ 29.41 ms      │ 17.56 ms      │ 17.29 ms      │ 100     │ 100
│     │                        5.226 Mitem/s │ 1.699 Mitem/s │ 2.846 Mitem/s │ 2.89 Mitem/s  │         │
│     ╰─ 6                     9.505 ms      │ 31.73 ms      │ 17.69 ms      │ 18.57 ms      │ 100     │ 100
│                              6.312 Mitem/s │ 1.89 Mitem/s  │ 3.391 Mitem/s │ 3.23 Mitem/s  │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.238 ms      │ 6.02 ms       │ 3.448 ms      │ 4.109 ms      │ 100     │ 100
│  │  │                        3.088 Mitem/s │ 1.661 Mitem/s │ 2.899 Mitem/s │ 2.433 Mitem/s │         │
│  │  ├─ 2                     8.304 ms      │ 14.8 ms       │ 12.21 ms      │ 11.78 ms      │ 100     │ 100
│  │  │                        2.408 Mitem/s │ 1.35 Mitem/s  │ 1.636 Mitem/s │ 1.696 Mitem/s │         │
│  │  ├─ 3                     18.19 ms      │ 27.74 ms      │ 21.13 ms      │ 21.75 ms      │ 100     │ 100
│  │  │                        1.649 Mitem/s │ 1.081 Mitem/s │ 1.419 Mitem/s │ 1.379 Mitem/s │         │
│  │  ├─ 4                     29.07 ms      │ 39.41 ms      │ 34.12 ms      │ 33.48 ms      │ 100     │ 100
│  │  │                        1.375 Mitem/s │ 1.014 Mitem/s │ 1.172 Mitem/s │ 1.194 Mitem/s │         │
│  │  ├─ 5                     44.11 ms      │ 63.16 ms      │ 57 ms         │ 55.01 ms      │ 100     │ 100
│  │  │                        1.133 Mitem/s │ 791.5 Kitem/s │ 877.1 Kitem/s │ 908.8 Kitem/s │         │
│  │  ╰─ 6                     67.5 ms       │ 115 ms        │ 82.96 ms      │ 85.36 ms      │ 100     │ 100
│  │                           888.8 Kitem/s │ 521.5 Kitem/s │ 723.1 Kitem/s │ 702.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     12.66 ms      │ 17.98 ms      │ 12.8 ms       │ 13.65 ms      │ 100     │ 100
│  │  │                        789.5 Kitem/s │ 555.9 Kitem/s │ 780.7 Kitem/s │ 732.2 Kitem/s │         │
│  │  ├─ 2                     13.08 ms      │ 27.68 ms      │ 18.78 ms      │ 18.9 ms       │ 100     │ 100
│  │  │                        1.528 Mitem/s │ 722.3 Kitem/s │ 1.064 Mitem/s │ 1.057 Mitem/s │         │
│  │  ├─ 3                     14.16 ms      │ 27.31 ms      │ 19.77 ms      │ 21.1 ms       │ 100     │ 100
│  │  │                        2.118 Mitem/s │ 1.098 Mitem/s │ 1.517 Mitem/s │ 1.421 Mitem/s │         │
│  │  ├─ 4                     14.18 ms      │ 35.65 ms      │ 20.01 ms      │ 21.58 ms      │ 100     │ 100
│  │  │                        2.82 Mitem/s  │ 1.121 Mitem/s │ 1.998 Mitem/s │ 1.852 Mitem/s │         │
│  │  ├─ 5                     16.52 ms      │ 37.47 ms      │ 26.1 ms       │ 25.17 ms      │ 100     │ 100
│  │  │                        3.024 Mitem/s │ 1.334 Mitem/s │ 1.915 Mitem/s │ 1.986 Mitem/s │         │
│  │  ╰─ 6                     14.24 ms      │ 39.73 ms      │ 22.88 ms      │ 23.28 ms      │ 100     │ 100
│  │                           4.21 Mitem/s  │ 1.51 Mitem/s  │ 2.621 Mitem/s │ 2.576 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.922 ms      │ 17.95 ms      │ 9.195 ms      │ 10.34 ms      │ 100     │ 100
│     │                        1.12 Mitem/s  │ 557 Kitem/s   │ 1.087 Mitem/s │ 966.7 Kitem/s │         │
│     ├─ 2                     9.039 ms      │ 20.69 ms      │ 13.67 ms      │ 14.18 ms      │ 100     │ 100
│     │                        2.212 Mitem/s │ 966.3 Kitem/s │ 1.462 Mitem/s │ 1.409 Mitem/s │         │
│     ├─ 3                     11.88 ms      │ 22.73 ms      │ 18.38 ms      │ 17.03 ms      │ 100     │ 100
│     │                        2.525 Mitem/s │ 1.319 Mitem/s │ 1.632 Mitem/s │ 1.761 Mitem/s │         │
│     ├─ 4                     12.17 ms      │ 24.16 ms      │ 17.99 ms      │ 17.26 ms      │ 100     │ 100
│     │                        3.285 Mitem/s │ 1.655 Mitem/s │ 2.223 Mitem/s │ 2.316 Mitem/s │         │
│     ├─ 5                     9.482 ms      │ 32.82 ms      │ 15.95 ms      │ 16.71 ms      │ 100     │ 100
│     │                        5.272 Mitem/s │ 1.523 Mitem/s │ 3.134 Mitem/s │ 2.991 Mitem/s │         │
│     ╰─ 6                     9.659 ms      │ 33.26 ms      │ 17.05 ms      │ 17.91 ms      │ 100     │ 100
│                              6.211 Mitem/s │ 1.803 Mitem/s │ 3.517 Mitem/s │ 3.349 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.496 ms      │ 7.808 ms      │ 4.572 ms      │ 4.644 ms      │ 100     │ 100
│  │  │                        2.224 Mitem/s │ 1.28 Mitem/s  │ 2.187 Mitem/s │ 2.152 Mitem/s │         │
│  │  ├─ 2                     10.41 ms      │ 20.85 ms      │ 16.17 ms      │ 16.05 ms      │ 100     │ 100
│  │  │                        1.921 Mitem/s │ 958.8 Kitem/s │ 1.236 Mitem/s │ 1.245 Mitem/s │         │
│  │  ├─ 3                     21.61 ms      │ 37.67 ms      │ 29.95 ms      │ 30.05 ms      │ 100     │ 100
│  │  │                        1.388 Mitem/s │ 796.2 Kitem/s │ 1.001 Mitem/s │ 998.1 Kitem/s │         │
│  │  ├─ 4                     35.24 ms      │ 56.05 ms      │ 45.01 ms      │ 45.43 ms      │ 100     │ 100
│  │  │                        1.134 Mitem/s │ 713.5 Kitem/s │ 888.6 Kitem/s │ 880.3 Kitem/s │         │
│  │  ├─ 5                     59.45 ms      │ 95.58 ms      │ 77.76 ms      │ 76.1 ms       │ 100     │ 100
│  │  │                        841 Kitem/s   │ 523 Kitem/s   │ 642.9 Kitem/s │ 657 Kitem/s   │         │
│  │  ╰─ 6                     81.09 ms      │ 149.9 ms      │ 104.4 ms      │ 105.9 ms      │ 100     │ 100
│  │                           739.8 Kitem/s │ 400 Kitem/s   │ 574.1 Kitem/s │ 566.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     8.537 ms      │ 12.28 ms      │ 8.663 ms      │ 8.933 ms      │ 100     │ 100
│  │  │                        1.171 Mitem/s │ 814.2 Kitem/s │ 1.154 Mitem/s │ 1.119 Mitem/s │         │
│  │  ├─ 2                     8.868 ms      │ 18.11 ms      │ 9.551 ms      │ 11.42 ms      │ 100     │ 100
│  │  │                        2.255 Mitem/s │ 1.104 Mitem/s │ 2.093 Mitem/s │ 1.751 Mitem/s │         │
│  │  ├─ 3                     9.305 ms      │ 22.2 ms       │ 12.85 ms      │ 12.84 ms      │ 100     │ 100
│  │  │                        3.224 Mitem/s │ 1.35 Mitem/s  │ 2.334 Mitem/s │ 2.336 Mitem/s │         │
│  │  ├─ 4                     10.22 ms      │ 27.25 ms      │ 16.85 ms      │ 16.5 ms       │ 100     │ 100
│  │  │                        3.91 Mitem/s  │ 1.467 Mitem/s │ 2.373 Mitem/s │ 2.423 Mitem/s │         │
│  │  ├─ 5                     9.708 ms      │ 27.87 ms      │ 17.65 ms      │ 17.32 ms      │ 100     │ 100
│  │  │                        5.15 Mitem/s  │ 1.793 Mitem/s │ 2.831 Mitem/s │ 2.885 Mitem/s │         │
│  │  ╰─ 6                     9.699 ms      │ 26.08 ms      │ 15.77 ms      │ 16.03 ms      │ 100     │ 100
│  │                           6.185 Mitem/s │ 2.299 Mitem/s │ 3.803 Mitem/s │ 3.742 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.137 ms      │ 11.39 ms      │ 8.436 ms      │ 8.624 ms      │ 100     │ 100
│     │                        1.228 Mitem/s │ 877.2 Kitem/s │ 1.185 Mitem/s │ 1.159 Mitem/s │         │
│     ├─ 2                     8.288 ms      │ 17.73 ms      │ 12.62 ms      │ 12.71 ms      │ 100     │ 100
│     │                        2.413 Mitem/s │ 1.127 Mitem/s │ 1.584 Mitem/s │ 1.573 Mitem/s │         │
│     ├─ 3                     8.365 ms      │ 17.52 ms      │ 15 ms         │ 14.28 ms      │ 100     │ 100
│     │                        3.586 Mitem/s │ 1.712 Mitem/s │ 1.999 Mitem/s │ 2.099 Mitem/s │         │
│     ├─ 4                     9.041 ms      │ 25.78 ms      │ 16.48 ms      │ 16.11 ms      │ 100     │ 100
│     │                        4.424 Mitem/s │ 1.551 Mitem/s │ 2.426 Mitem/s │ 2.482 Mitem/s │         │
│     ├─ 5                     8.589 ms      │ 27.64 ms      │ 16.37 ms      │ 15.89 ms      │ 100     │ 100
│     │                        5.821 Mitem/s │ 1.808 Mitem/s │ 3.053 Mitem/s │ 3.145 Mitem/s │         │
│     ╰─ 6                     13.76 ms      │ 25.44 ms      │ 15.27 ms      │ 16.51 ms      │ 100     │ 100
│                              4.358 Mitem/s │ 2.358 Mitem/s │ 3.927 Mitem/s │ 3.634 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.537 ms      │ 7.3 ms        │ 4.586 ms      │ 4.698 ms      │ 100     │ 100
│  │  │                        2.204 Mitem/s │ 1.369 Mitem/s │ 2.18 Mitem/s  │ 2.128 Mitem/s │         │
│  │  ├─ 2                     10.49 ms      │ 20.61 ms      │ 14.88 ms      │ 14.86 ms      │ 100     │ 100
│  │  │                        1.905 Mitem/s │ 970.2 Kitem/s │ 1.343 Mitem/s │ 1.345 Mitem/s │         │
│  │  ├─ 3                     21.63 ms      │ 36.87 ms      │ 29.1 ms       │ 29.44 ms      │ 100     │ 100
│  │  │                        1.386 Mitem/s │ 813.5 Kitem/s │ 1.03 Mitem/s  │ 1.018 Mitem/s │         │
│  │  ├─ 4                     39.94 ms      │ 59.62 ms      │ 45.92 ms      │ 46.12 ms      │ 100     │ 100
│  │  │                        1.001 Mitem/s │ 670.8 Kitem/s │ 870.9 Kitem/s │ 867.2 Kitem/s │         │
│  │  ├─ 5                     58.34 ms      │ 90.33 ms      │ 78.21 ms      │ 77.01 ms      │ 100     │ 100
│  │  │                        856.9 Kitem/s │ 553.4 Kitem/s │ 639.2 Kitem/s │ 649.2 Kitem/s │         │
│  │  ╰─ 6                     83.16 ms      │ 163.4 ms      │ 118.2 ms      │ 120 ms        │ 100     │ 100
│  │                           721.4 Kitem/s │ 367 Kitem/s   │ 507.5 Kitem/s │ 499.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.005 ms      │ 12.85 ms      │ 9.141 ms      │ 10.23 ms      │ 100     │ 100
│  │  │                        1.11 Mitem/s  │ 777.8 Kitem/s │ 1.093 Mitem/s │ 977.1 Kitem/s │         │
│  │  ├─ 2                     9.449 ms      │ 20.04 ms      │ 14.29 ms      │ 15.23 ms      │ 100     │ 100
│  │  │                        2.116 Mitem/s │ 997.7 Kitem/s │ 1.399 Mitem/s │ 1.312 Mitem/s │         │
│  │  ├─ 3                     9.415 ms      │ 23.02 ms      │ 14.41 ms      │ 15.79 ms      │ 100     │ 100
│  │  │                        3.186 Mitem/s │ 1.302 Mitem/s │ 2.081 Mitem/s │ 1.899 Mitem/s │         │
│  │  ├─ 4                     10.27 ms      │ 29.77 ms      │ 18.41 ms      │ 18.56 ms      │ 100     │ 100
│  │  │                        3.891 Mitem/s │ 1.343 Mitem/s │ 2.171 Mitem/s │ 2.154 Mitem/s │         │
│  │  ├─ 5                     13.7 ms       │ 28.5 ms       │ 17.02 ms      │ 17.5 ms       │ 100     │ 100
│  │  │                        3.647 Mitem/s │ 1.754 Mitem/s │ 2.936 Mitem/s │ 2.856 Mitem/s │         │
│  │  ╰─ 6                     11.03 ms      │ 30.51 ms      │ 16.97 ms      │ 17.71 ms      │ 100     │ 100
│  │                           5.439 Mitem/s │ 1.966 Mitem/s │ 3.535 Mitem/s │ 3.387 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.531 ms      │ 11.44 ms      │ 8.684 ms      │ 8.919 ms      │ 100     │ 100
│     │                        1.172 Mitem/s │ 873.7 Kitem/s │ 1.151 Mitem/s │ 1.121 Mitem/s │         │
│     ├─ 2                     8.585 ms      │ 18.85 ms      │ 12.33 ms      │ 13.61 ms      │ 100     │ 100
│     │                        2.329 Mitem/s │ 1.06 Mitem/s  │ 1.621 Mitem/s │ 1.468 Mitem/s │         │
│     ├─ 3                     8.828 ms      │ 19.95 ms      │ 15.83 ms      │ 15.65 ms      │ 100     │ 100
│     │                        3.397 Mitem/s │ 1.503 Mitem/s │ 1.894 Mitem/s │ 1.916 Mitem/s │         │
│     ├─ 4                     11.91 ms      │ 23.85 ms      │ 15.7 ms       │ 15.75 ms      │ 100     │ 100
│     │                        3.358 Mitem/s │ 1.676 Mitem/s │ 2.547 Mitem/s │ 2.538 Mitem/s │         │
│     ├─ 5                     12.19 ms      │ 31.62 ms      │ 15.55 ms      │ 15.94 ms      │ 100     │ 100
│     │                        4.1 Mitem/s   │ 1.581 Mitem/s │ 3.213 Mitem/s │ 3.135 Mitem/s │         │
│     ╰─ 6                     8.914 ms      │ 28.83 ms      │ 15.65 ms      │ 15.96 ms      │ 100     │ 100
│                              6.73 Mitem/s  │ 2.08 Mitem/s  │ 3.832 Mitem/s │ 3.758 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.479 ms      │ 7.216 ms      │ 4.623 ms      │ 4.72 ms       │ 100     │ 100
│  │  │                        2.232 Mitem/s │ 1.385 Mitem/s │ 2.163 Mitem/s │ 2.118 Mitem/s │         │
│  │  ├─ 2                     11.12 ms      │ 20.19 ms      │ 15.62 ms      │ 16.03 ms      │ 100     │ 100
│  │  │                        1.797 Mitem/s │ 990.5 Kitem/s │ 1.279 Mitem/s │ 1.247 Mitem/s │         │
│  │  ├─ 3                     22.35 ms      │ 36.15 ms      │ 30.15 ms      │ 29.58 ms      │ 100     │ 100
│  │  │                        1.342 Mitem/s │ 829.8 Kitem/s │ 994.9 Kitem/s │ 1.013 Mitem/s │         │
│  │  ├─ 4                     36.71 ms      │ 56.41 ms      │ 46.8 ms       │ 45.85 ms      │ 100     │ 100
│  │  │                        1.089 Mitem/s │ 709 Kitem/s   │ 854.6 Kitem/s │ 872.3 Kitem/s │         │
│  │  ├─ 5                     56.14 ms      │ 90.45 ms      │ 77.77 ms      │ 76.37 ms      │ 100     │ 100
│  │  │                        890.5 Kitem/s │ 552.7 Kitem/s │ 642.8 Kitem/s │ 654.6 Kitem/s │         │
│  │  ╰─ 6                     80.03 ms      │ 152.8 ms      │ 107.7 ms      │ 108.7 ms      │ 100     │ 100
│  │                           749.7 Kitem/s │ 392.5 Kitem/s │ 556.9 Kitem/s │ 551.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.093 ms      │ 17.14 ms      │ 9.14 ms       │ 10.15 ms      │ 100     │ 100
│  │  │                        1.099 Mitem/s │ 583 Kitem/s   │ 1.094 Mitem/s │ 984.2 Kitem/s │         │
│  │  ├─ 2                     9.639 ms      │ 20.13 ms      │ 13.92 ms      │ 14.45 ms      │ 100     │ 100
│  │  │                        2.074 Mitem/s │ 993.4 Kitem/s │ 1.436 Mitem/s │ 1.383 Mitem/s │         │
│  │  ├─ 3                     9.614 ms      │ 20.49 ms      │ 17.83 ms      │ 16.76 ms      │ 100     │ 100
│  │  │                        3.12 Mitem/s  │ 1.463 Mitem/s │ 1.681 Mitem/s │ 1.789 Mitem/s │         │
│  │  ├─ 4                     10.3 ms       │ 28.3 ms       │ 16.78 ms      │ 16.72 ms      │ 100     │ 100
│  │  │                        3.882 Mitem/s │ 1.413 Mitem/s │ 2.383 Mitem/s │ 2.39 Mitem/s  │         │
│  │  ├─ 5                     10.7 ms       │ 29.53 ms      │ 18.1 ms       │ 18.37 ms      │ 100     │ 100
│  │  │                        4.671 Mitem/s │ 1.693 Mitem/s │ 2.761 Mitem/s │ 2.721 Mitem/s │         │
│  │  ╰─ 6                     10.41 ms      │ 26.71 ms      │ 16.71 ms      │ 16.64 ms      │ 100     │ 100
│  │                           5.761 Mitem/s │ 2.245 Mitem/s │ 3.59 Mitem/s  │ 3.604 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.357 ms      │ 12.59 ms      │ 8.66 ms       │ 8.841 ms      │ 100     │ 100
│     │                        1.196 Mitem/s │ 793.8 Kitem/s │ 1.154 Mitem/s │ 1.131 Mitem/s │         │
│     ├─ 2                     8.616 ms      │ 18.71 ms      │ 12.8 ms       │ 13.59 ms      │ 100     │ 100
│     │                        2.321 Mitem/s │ 1.068 Mitem/s │ 1.562 Mitem/s │ 1.471 Mitem/s │         │
│     ├─ 3                     8.666 ms      │ 22.33 ms      │ 15.82 ms      │ 15.45 ms      │ 100     │ 100
│     │                        3.461 Mitem/s │ 1.343 Mitem/s │ 1.895 Mitem/s │ 1.941 Mitem/s │         │
│     ├─ 4                     8.821 ms      │ 29.46 ms      │ 15.75 ms      │ 15.88 ms      │ 100     │ 100
│     │                        4.534 Mitem/s │ 1.357 Mitem/s │ 2.539 Mitem/s │ 2.517 Mitem/s │         │
│     ├─ 5                     8.994 ms      │ 27.58 ms      │ 15.63 ms      │ 15.7 ms       │ 100     │ 100
│     │                        5.559 Mitem/s │ 1.812 Mitem/s │ 3.197 Mitem/s │ 3.184 Mitem/s │         │
│     ╰─ 6                     8.965 ms      │ 31.57 ms      │ 16.04 ms      │ 16.16 ms      │ 100     │ 100
│                              6.692 Mitem/s │ 1.899 Mitem/s │ 3.739 Mitem/s │ 3.71 Mitem/s  │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.449 ms      │ 7.14 ms       │ 4.543 ms      │ 4.629 ms      │ 100     │ 100
│  │  │                        2.247 Mitem/s │ 1.4 Mitem/s   │ 2.201 Mitem/s │ 2.16 Mitem/s  │         │
│  │  ├─ 2                     10.99 ms      │ 20.46 ms      │ 15.55 ms      │ 16.21 ms      │ 100     │ 100
│  │  │                        1.819 Mitem/s │ 977.3 Kitem/s │ 1.285 Mitem/s │ 1.233 Mitem/s │         │
│  │  ├─ 3                     20.83 ms      │ 35.77 ms      │ 29.79 ms      │ 29.99 ms      │ 100     │ 100
│  │  │                        1.439 Mitem/s │ 838.5 Kitem/s │ 1.006 Mitem/s │ 1 Mitem/s     │         │
│  │  ├─ 4                     35.53 ms      │ 58.37 ms      │ 46.35 ms      │ 45.5 ms       │ 100     │ 100
│  │  │                        1.125 Mitem/s │ 685.1 Kitem/s │ 862.9 Kitem/s │ 879 Kitem/s   │         │
│  │  ├─ 5                     58.58 ms      │ 85.84 ms      │ 77.97 ms      │ 76.79 ms      │ 100     │ 100
│  │  │                        853.4 Kitem/s │ 582.4 Kitem/s │ 641.2 Kitem/s │ 651 Kitem/s   │         │
│  │  ╰─ 6                     83.96 ms      │ 158.6 ms      │ 114.1 ms      │ 115.9 ms      │ 100     │ 100
│  │                           714.6 Kitem/s │ 378 Kitem/s   │ 525.4 Kitem/s │ 517.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.154 ms      │ 11.82 ms      │ 9.265 ms      │ 9.452 ms      │ 100     │ 100
│  │  │                        1.092 Mitem/s │ 845.3 Kitem/s │ 1.079 Mitem/s │ 1.057 Mitem/s │         │
│  │  ├─ 2                     9.608 ms      │ 19.98 ms      │ 13.71 ms      │ 14.54 ms      │ 100     │ 100
│  │  │                        2.081 Mitem/s │ 1 Mitem/s     │ 1.458 Mitem/s │ 1.374 Mitem/s │         │
│  │  ├─ 3                     10.28 ms      │ 20.65 ms      │ 15.31 ms      │ 16.07 ms      │ 100     │ 100
│  │  │                        2.916 Mitem/s │ 1.452 Mitem/s │ 1.959 Mitem/s │ 1.866 Mitem/s │         │
│  │  ├─ 4                     10.55 ms      │ 31.17 ms      │ 17.96 ms      │ 17.54 ms      │ 100     │ 100
│  │  │                        3.788 Mitem/s │ 1.282 Mitem/s │ 2.226 Mitem/s │ 2.279 Mitem/s │         │
│  │  ├─ 5                     13.7 ms       │ 24.34 ms      │ 18.75 ms      │ 18.37 ms      │ 100     │ 100
│  │  │                        3.648 Mitem/s │ 2.053 Mitem/s │ 2.666 Mitem/s │ 2.721 Mitem/s │         │
│  │  ╰─ 6                     10.4 ms       │ 28.46 ms      │ 16.87 ms      │ 17.32 ms      │ 100     │ 100
│  │                           5.768 Mitem/s │ 2.107 Mitem/s │ 3.556 Mitem/s │ 3.462 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.461 ms      │ 10.87 ms      │ 8.57 ms       │ 8.621 ms      │ 100     │ 100
│     │                        1.181 Mitem/s │ 919.3 Kitem/s │ 1.166 Mitem/s │ 1.159 Mitem/s │         │
│     ├─ 2                     8.615 ms      │ 20.06 ms      │ 13.05 ms      │ 13.46 ms      │ 100     │ 100
│     │                        2.321 Mitem/s │ 996.5 Kitem/s │ 1.531 Mitem/s │ 1.485 Mitem/s │         │
│     ├─ 3                     9.09 ms       │ 20.41 ms      │ 15.58 ms      │ 15.3 ms       │ 100     │ 100
│     │                        3.3 Mitem/s   │ 1.469 Mitem/s │ 1.925 Mitem/s │ 1.96 Mitem/s  │         │
│     ├─ 4                     9.109 ms      │ 28.69 ms      │ 16.42 ms      │ 16.32 ms      │ 100     │ 100
│     │                        4.39 Mitem/s  │ 1.393 Mitem/s │ 2.435 Mitem/s │ 2.449 Mitem/s │         │
│     ├─ 5                     12.22 ms      │ 30.3 ms       │ 17.88 ms      │ 17.75 ms      │ 100     │ 100
│     │                        4.091 Mitem/s │ 1.649 Mitem/s │ 2.795 Mitem/s │ 2.816 Mitem/s │         │
│     ╰─ 6                     8.947 ms      │ 29.57 ms      │ 15.62 ms      │ 16.22 ms      │ 100     │ 100
│                              6.705 Mitem/s │ 2.028 Mitem/s │ 3.839 Mitem/s │ 3.697 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.103 ms      │ 6.22 ms       │ 4.186 ms      │ 4.228 ms      │ 100     │ 100
│  │  │                        2.437 Mitem/s │ 1.607 Mitem/s │ 2.388 Mitem/s │ 2.364 Mitem/s │         │
│  │  ├─ 2                     11.25 ms      │ 18.91 ms      │ 14.54 ms      │ 14.94 ms      │ 100     │ 100
│  │  │                        1.776 Mitem/s │ 1.057 Mitem/s │ 1.375 Mitem/s │ 1.337 Mitem/s │         │
│  │  ├─ 3                     19.49 ms      │ 30.84 ms      │ 26.81 ms      │ 26.93 ms      │ 100     │ 100
│  │  │                        1.538 Mitem/s │ 972.4 Kitem/s │ 1.118 Mitem/s │ 1.113 Mitem/s │         │
│  │  ├─ 4                     33.74 ms      │ 48.13 ms      │ 40.38 ms      │ 40.21 ms      │ 100     │ 100
│  │  │                        1.185 Mitem/s │ 830.9 Kitem/s │ 990.4 Kitem/s │ 994.5 Kitem/s │         │
│  │  ├─ 5                     51.36 ms      │ 78.36 ms      │ 66.44 ms      │ 65.88 ms      │ 100     │ 100
│  │  │                        973.4 Kitem/s │ 638 Kitem/s   │ 752.4 Kitem/s │ 758.8 Kitem/s │         │
│  │  ╰─ 6                     80.94 ms      │ 145.2 ms      │ 103.9 ms      │ 105.5 ms      │ 100     │ 100
│  │                           741.2 Kitem/s │ 413 Kitem/s   │ 576.9 Kitem/s │ 568.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     13.94 ms      │ 17.48 ms      │ 14.09 ms      │ 14.49 ms      │ 100     │ 100
│  │  │                        717.3 Kitem/s │ 571.9 Kitem/s │ 709.3 Kitem/s │ 689.8 Kitem/s │         │
│  │  ├─ 2                     14.57 ms      │ 28.98 ms      │ 21 ms         │ 20.97 ms      │ 100     │ 100
│  │  │                        1.372 Mitem/s │ 690 Kitem/s   │ 952.1 Kitem/s │ 953.5 Kitem/s │         │
│  │  ├─ 3                     14.85 ms      │ 28.28 ms      │ 21.37 ms      │ 21.99 ms      │ 100     │ 100
│  │  │                        2.019 Mitem/s │ 1.06 Mitem/s  │ 1.403 Mitem/s │ 1.363 Mitem/s │         │
│  │  ├─ 4                     15.03 ms      │ 35.99 ms      │ 21.13 ms      │ 22.34 ms      │ 100     │ 100
│  │  │                        2.66 Mitem/s  │ 1.111 Mitem/s │ 1.892 Mitem/s │ 1.79 Mitem/s  │         │
│  │  ├─ 5                     16.75 ms      │ 39.3 ms       │ 21.9 ms       │ 24.73 ms      │ 100     │ 100
│  │  │                        2.985 Mitem/s │ 1.272 Mitem/s │ 2.282 Mitem/s │ 2.021 Mitem/s │         │
│  │  ╰─ 6                     15.3 ms       │ 43.35 ms      │ 23.94 ms      │ 24.63 ms      │ 100     │ 100
│  │                           3.919 Mitem/s │ 1.383 Mitem/s │ 2.506 Mitem/s │ 2.435 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.488 ms      │ 13.04 ms      │ 8.722 ms      │ 8.981 ms      │ 100     │ 100
│     │                        1.178 Mitem/s │ 766.6 Kitem/s │ 1.146 Mitem/s │ 1.113 Mitem/s │         │
│     ├─ 2                     8.764 ms      │ 25.55 ms      │ 13.02 ms      │ 13.79 ms      │ 100     │ 100
│     │                        2.281 Mitem/s │ 782.4 Kitem/s │ 1.535 Mitem/s │ 1.449 Mitem/s │         │
│     ├─ 3                     9.244 ms      │ 21.35 ms      │ 13.71 ms      │ 15.05 ms      │ 100     │ 100
│     │                        3.245 Mitem/s │ 1.404 Mitem/s │ 2.187 Mitem/s │ 1.993 Mitem/s │         │
│     ├─ 4                     9.069 ms      │ 29.5 ms       │ 17.26 ms      │ 16.6 ms       │ 100     │ 100
│     │                        4.41 Mitem/s  │ 1.355 Mitem/s │ 2.316 Mitem/s │ 2.409 Mitem/s │         │
│     ├─ 5                     12.3 ms       │ 28.22 ms      │ 15.8 ms       │ 15.74 ms      │ 100     │ 100
│     │                        4.062 Mitem/s │ 1.771 Mitem/s │ 3.164 Mitem/s │ 3.174 Mitem/s │         │
│     ╰─ 6                     8.956 ms      │ 28.86 ms      │ 15.48 ms      │ 15.85 ms      │ 100     │ 100
│                              6.698 Mitem/s │ 2.078 Mitem/s │ 3.873 Mitem/s │ 3.784 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 2                     7.131 ms      │ 12.08 ms      │ 10.78 ms      │ 10.82 ms      │ 100     │ 100
│  │  ├─ 4                     12.66 ms      │ 29.21 ms      │ 19.43 ms      │ 18.83 ms      │ 100     │ 100
│  │  ├─ 8                     17.2 ms       │ 32.09 ms      │ 19.77 ms      │ 21.91 ms      │ 100     │ 100
│  │  ├─ 16                    24.68 ms      │ 38.11 ms      │ 28.36 ms      │ 28.77 ms      │ 100     │ 100
│  │  ╰─ 32                    47.58 ms      │ 72.25 ms      │ 50.17 ms      │ 50.88 ms      │ 100     │ 100
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 2                     13.03 ms      │ 26.51 ms      │ 15.15 ms      │ 17.28 ms      │ 100     │ 100
│     ├─ 4                     12.67 ms      │ 24.58 ms      │ 17.17 ms      │ 17.95 ms      │ 100     │ 100
│     ├─ 8                     16.61 ms      │ 31.05 ms      │ 19.73 ms      │ 22.26 ms      │ 100     │ 100
│     ├─ 16                    24.74 ms      │ 36.86 ms      │ 28.49 ms      │ 28.55 ms      │ 100     │ 100
│     ╰─ 32                    46.68 ms      │ 55.25 ms      │ 50.37 ms      │ 50.54 ms      │ 100     │ 100
```
