```text
Timer precision: 30 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.498 ms      │ 8.361 ms      │ 4.622 ms      │ 4.963 ms      │ 100     │ 100
│  │  │                        2.223 Mitem/s │ 1.196 Mitem/s │ 2.163 Mitem/s │ 2.014 Mitem/s │         │
│  │  ├─ 2                     11.12 ms      │ 20.42 ms      │ 15.3 ms       │ 15.27 ms      │ 100     │ 100
│  │  │                        1.797 Mitem/s │ 979.1 Kitem/s │ 1.306 Mitem/s │ 1.309 Mitem/s │         │
│  │  ├─ 3                     21.66 ms      │ 36.36 ms      │ 28.16 ms      │ 28.08 ms      │ 100     │ 100
│  │  │                        1.385 Mitem/s │ 825 Kitem/s   │ 1.065 Mitem/s │ 1.068 Mitem/s │         │
│  │  ├─ 4                     33.73 ms      │ 60.18 ms      │ 42.65 ms      │ 43.51 ms      │ 100     │ 100
│  │  │                        1.185 Mitem/s │ 664.6 Kitem/s │ 937.6 Kitem/s │ 919.2 Kitem/s │         │
│  │  ├─ 5                     49.82 ms      │ 79.08 ms      │ 59.67 ms      │ 60.69 ms      │ 100     │ 100
│  │  │                        1.003 Mitem/s │ 632.1 Kitem/s │ 837.8 Kitem/s │ 823.8 Kitem/s │         │
│  │  ╰─ 6                     80.94 ms      │ 107.2 ms      │ 94.84 ms      │ 94.96 ms      │ 100     │ 100
│  │                           741.2 Kitem/s │ 559.2 Kitem/s │ 632.5 Kitem/s │ 631.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     12.43 ms      │ 20.25 ms      │ 12.82 ms      │ 13.25 ms      │ 100     │ 100
│  │  │                        804.2 Kitem/s │ 493.7 Kitem/s │ 779.4 Kitem/s │ 754.5 Kitem/s │         │
│  │  ├─ 2                     22.21 ms      │ 41.14 ms      │ 38.29 ms      │ 33.95 ms      │ 100     │ 100
│  │  │                        900.2 Kitem/s │ 486 Kitem/s   │ 522.2 Kitem/s │ 589 Kitem/s   │         │
│  │  ├─ 3                     25.86 ms      │ 48.69 ms      │ 44.75 ms      │ 42.69 ms      │ 100     │ 100
│  │  │                        1.159 Mitem/s │ 616 Kitem/s   │ 670.3 Kitem/s │ 702.7 Kitem/s │         │
│  │  ├─ 4                     26.57 ms      │ 57.18 ms      │ 49.51 ms      │ 47.28 ms      │ 100     │ 100
│  │  │                        1.505 Mitem/s │ 699.4 Kitem/s │ 807.7 Kitem/s │ 845.9 Kitem/s │         │
│  │  ├─ 5                     31.07 ms      │ 65.78 ms      │ 51.06 ms      │ 48.91 ms      │ 100     │ 100
│  │  │                        1.609 Mitem/s │ 760 Kitem/s   │ 979.2 Kitem/s │ 1.022 Mitem/s │         │
│  │  ╰─ 6                     34.43 ms      │ 65.59 ms      │ 60.67 ms      │ 59.54 ms      │ 100     │ 100
│  │                           1.742 Mitem/s │ 914.7 Kitem/s │ 988.9 Kitem/s │ 1.007 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.533 ms      │ 15.21 ms      │ 9.19 ms       │ 10.16 ms      │ 100     │ 100
│     │                        1.171 Mitem/s │ 657 Kitem/s   │ 1.088 Mitem/s │ 983.8 Kitem/s │         │
│     ├─ 2                     8.796 ms      │ 18.21 ms      │ 12.03 ms      │ 12.06 ms      │ 100     │ 100
│     │                        2.273 Mitem/s │ 1.097 Mitem/s │ 1.662 Mitem/s │ 1.657 Mitem/s │         │
│     ├─ 3                     8.797 ms      │ 21.01 ms      │ 11.89 ms      │ 12.41 ms      │ 100     │ 100
│     │                        3.41 Mitem/s  │ 1.427 Mitem/s │ 2.521 Mitem/s │ 2.415 Mitem/s │         │
│     ├─ 4                     9.228 ms      │ 18.93 ms      │ 13.69 ms      │ 14.11 ms      │ 100     │ 100
│     │                        4.334 Mitem/s │ 2.112 Mitem/s │ 2.919 Mitem/s │ 2.832 Mitem/s │         │
│     ├─ 5                     9.77 ms       │ 31.48 ms      │ 17.04 ms      │ 17.63 ms      │ 100     │ 100
│     │                        5.117 Mitem/s │ 1.587 Mitem/s │ 2.933 Mitem/s │ 2.835 Mitem/s │         │
│     ╰─ 6                     10.29 ms      │ 28.18 ms      │ 16.39 ms      │ 17.83 ms      │ 100     │ 100
│                              5.826 Mitem/s │ 2.128 Mitem/s │ 3.659 Mitem/s │ 3.363 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.647 ms      │ 7.627 ms      │ 4.785 ms      │ 5.306 ms      │ 100     │ 100
│  │  │                        2.151 Mitem/s │ 1.31 Mitem/s  │ 2.089 Mitem/s │ 1.884 Mitem/s │         │
│  │  ├─ 2                     10.85 ms      │ 20.42 ms      │ 14.38 ms      │ 14.8 ms       │ 100     │ 100
│  │  │                        1.841 Mitem/s │ 979.4 Kitem/s │ 1.39 Mitem/s  │ 1.351 Mitem/s │         │
│  │  ├─ 3                     22.93 ms      │ 38.57 ms      │ 28.72 ms      │ 29.37 ms      │ 100     │ 100
│  │  │                        1.308 Mitem/s │ 777.6 Kitem/s │ 1.044 Mitem/s │ 1.021 Mitem/s │         │
│  │  ├─ 4                     34.83 ms      │ 56.99 ms      │ 42.49 ms      │ 43.04 ms      │ 100     │ 100
│  │  │                        1.148 Mitem/s │ 701.8 Kitem/s │ 941.2 Kitem/s │ 929.2 Kitem/s │         │
│  │  ├─ 5                     51.49 ms      │ 71.19 ms      │ 61.72 ms      │ 61.55 ms      │ 100     │ 100
│  │  │                        970.9 Kitem/s │ 702.3 Kitem/s │ 810 Kitem/s   │ 812.2 Kitem/s │         │
│  │  ╰─ 6                     85.89 ms      │ 109.7 ms      │ 99.64 ms      │ 98.59 ms      │ 100     │ 100
│  │                           698.4 Kitem/s │ 546.6 Kitem/s │ 602.1 Kitem/s │ 608.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     12.28 ms      │ 18.29 ms      │ 13.03 ms      │ 13.79 ms      │ 100     │ 100
│  │  │                        814 Kitem/s   │ 546.6 Kitem/s │ 767.3 Kitem/s │ 725.1 Kitem/s │         │
│  │  ├─ 2                     17.92 ms      │ 47.64 ms      │ 31.76 ms      │ 32.37 ms      │ 100     │ 100
│  │  │                        1.115 Mitem/s │ 419.7 Kitem/s │ 629.5 Kitem/s │ 617.7 Kitem/s │         │
│  │  ├─ 3                     22.15 ms      │ 61.56 ms      │ 53.18 ms      │ 46.99 ms      │ 100     │ 100
│  │  │                        1.354 Mitem/s │ 487.2 Kitem/s │ 564 Kitem/s   │ 638.3 Kitem/s │         │
│  │  ├─ 4                     24.08 ms      │ 71.28 ms      │ 58.62 ms      │ 51.79 ms      │ 100     │ 100
│  │  │                        1.66 Mitem/s  │ 561.1 Kitem/s │ 682.3 Kitem/s │ 772.2 Kitem/s │         │
│  │  ├─ 5                     28.48 ms      │ 70.3 ms       │ 54.34 ms      │ 50.3 ms       │ 100     │ 100
│  │  │                        1.755 Mitem/s │ 711.1 Kitem/s │ 920.1 Kitem/s │ 993.8 Kitem/s │         │
│  │  ╰─ 6                     30.43 ms      │ 70.59 ms      │ 66.53 ms      │ 62.59 ms      │ 100     │ 100
│  │                           1.971 Mitem/s │ 849.9 Kitem/s │ 901.7 Kitem/s │ 958.5 Kitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.88 ms       │ 11.84 ms      │ 9.074 ms      │ 9.243 ms      │ 100     │ 100
│     │                        1.126 Mitem/s │ 844.1 Kitem/s │ 1.102 Mitem/s │ 1.081 Mitem/s │         │
│     ├─ 2                     9.117 ms      │ 20.15 ms      │ 12.76 ms      │ 13.14 ms      │ 100     │ 100
│     │                        2.193 Mitem/s │ 992.3 Kitem/s │ 1.566 Mitem/s │ 1.521 Mitem/s │         │
│     ├─ 3                     9.328 ms      │ 20.16 ms      │ 16.09 ms      │ 15.83 ms      │ 100     │ 100
│     │                        3.215 Mitem/s │ 1.487 Mitem/s │ 1.863 Mitem/s │ 1.894 Mitem/s │         │
│     ├─ 4                     9.418 ms      │ 27.63 ms      │ 14.2 ms       │ 15.84 ms      │ 100     │ 100
│     │                        4.247 Mitem/s │ 1.447 Mitem/s │ 2.815 Mitem/s │ 2.524 Mitem/s │         │
│     ├─ 5                     9.375 ms      │ 27.94 ms      │ 15.54 ms      │ 15.91 ms      │ 100     │ 100
│     │                        5.333 Mitem/s │ 1.789 Mitem/s │ 3.217 Mitem/s │ 3.141 Mitem/s │         │
│     ╰─ 6                     9.628 ms      │ 31.98 ms      │ 17.9 ms       │ 19.66 ms      │ 100     │ 100
│                              6.231 Mitem/s │ 1.875 Mitem/s │ 3.35 Mitem/s  │ 3.051 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.487 ms      │ 8.792 ms      │ 4.612 ms      │ 4.768 ms      │ 100     │ 100
│  │  │                        2.228 Mitem/s │ 1.137 Mitem/s │ 2.167 Mitem/s │ 2.097 Mitem/s │         │
│  │  ├─ 2                     11.14 ms      │ 19.87 ms      │ 13.85 ms      │ 14.33 ms      │ 100     │ 100
│  │  │                        1.793 Mitem/s │ 1.006 Mitem/s │ 1.443 Mitem/s │ 1.395 Mitem/s │         │
│  │  ├─ 3                     21.32 ms      │ 38.18 ms      │ 29.25 ms      │ 29.14 ms      │ 100     │ 100
│  │  │                        1.407 Mitem/s │ 785.6 Kitem/s │ 1.025 Mitem/s │ 1.029 Mitem/s │         │
│  │  ├─ 4                     35.12 ms      │ 56.46 ms      │ 41.81 ms      │ 42.58 ms      │ 100     │ 100
│  │  │                        1.138 Mitem/s │ 708.4 Kitem/s │ 956.6 Kitem/s │ 939.2 Kitem/s │         │
│  │  ├─ 5                     50.54 ms      │ 80.89 ms      │ 64.27 ms      │ 64.75 ms      │ 100     │ 100
│  │  │                        989.1 Kitem/s │ 618 Kitem/s   │ 777.9 Kitem/s │ 772.1 Kitem/s │         │
│  │  ╰─ 6                     82.01 ms      │ 110 ms        │ 95.41 ms      │ 94.6 ms       │ 100     │ 100
│  │                           731.5 Kitem/s │ 545.2 Kitem/s │ 628.8 Kitem/s │ 634.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     12.55 ms      │ 22.33 ms      │ 13.19 ms      │ 14.49 ms      │ 100     │ 100
│  │  │                        796.1 Kitem/s │ 447.7 Kitem/s │ 757.8 Kitem/s │ 690 Kitem/s   │         │
│  │  ├─ 2                     17.88 ms      │ 37.78 ms      │ 31.17 ms      │ 30.6 ms       │ 100     │ 100
│  │  │                        1.118 Mitem/s │ 529.3 Kitem/s │ 641.5 Kitem/s │ 653.5 Kitem/s │         │
│  │  ├─ 3                     25.29 ms      │ 49.74 ms      │ 44.22 ms      │ 42.49 ms      │ 100     │ 100
│  │  │                        1.185 Mitem/s │ 603 Kitem/s   │ 678.3 Kitem/s │ 706 Kitem/s   │         │
│  │  ├─ 4                     25.26 ms      │ 51.49 ms      │ 44.62 ms      │ 43.4 ms       │ 100     │ 100
│  │  │                        1.583 Mitem/s │ 776.8 Kitem/s │ 896.3 Kitem/s │ 921.6 Kitem/s │         │
│  │  ├─ 5                     31.94 ms      │ 57.05 ms      │ 51.78 ms      │ 49.97 ms      │ 100     │ 100
│  │  │                        1.565 Mitem/s │ 876.3 Kitem/s │ 965.6 Kitem/s │ 1 Mitem/s     │         │
│  │  ╰─ 6                     34.26 ms      │ 64.37 ms      │ 59.2 ms       │ 56.6 ms       │ 100     │ 100
│  │                           1.751 Mitem/s │ 931.9 Kitem/s │ 1.013 Mitem/s │ 1.059 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.613 ms      │ 14.21 ms      │ 9.409 ms      │ 10.64 ms      │ 100     │ 100
│     │                        1.16 Mitem/s  │ 703.6 Kitem/s │ 1.062 Mitem/s │ 939.1 Kitem/s │         │
│     ├─ 2                     8.708 ms      │ 18.44 ms      │ 12.72 ms      │ 13.23 ms      │ 100     │ 100
│     │                        2.296 Mitem/s │ 1.084 Mitem/s │ 1.571 Mitem/s │ 1.511 Mitem/s │         │
│     ├─ 3                     8.923 ms      │ 20.72 ms      │ 15.18 ms      │ 14.3 ms       │ 100     │ 100
│     │                        3.361 Mitem/s │ 1.447 Mitem/s │ 1.976 Mitem/s │ 2.097 Mitem/s │         │
│     ├─ 4                     9.039 ms      │ 23.58 ms      │ 14.22 ms      │ 14.19 ms      │ 100     │ 100
│     │                        4.425 Mitem/s │ 1.695 Mitem/s │ 2.811 Mitem/s │ 2.817 Mitem/s │         │
│     ├─ 5                     9.27 ms       │ 27.71 ms      │ 16.12 ms      │ 16.61 ms      │ 100     │ 100
│     │                        5.393 Mitem/s │ 1.804 Mitem/s │ 3.101 Mitem/s │ 3.009 Mitem/s │         │
│     ╰─ 6                     9.272 ms      │ 32.26 ms      │ 16.25 ms      │ 17 ms         │ 100     │ 100
│                              6.47 Mitem/s  │ 1.859 Mitem/s │ 3.691 Mitem/s │ 3.527 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.544 ms      │ 7.268 ms      │ 4.63 ms       │ 4.823 ms      │ 100     │ 100
│  │  │                        2.2 Mitem/s   │ 1.375 Mitem/s │ 2.159 Mitem/s │ 2.073 Mitem/s │         │
│  │  ├─ 2                     11.32 ms      │ 20.61 ms      │ 15.58 ms      │ 15.22 ms      │ 100     │ 100
│  │  │                        1.765 Mitem/s │ 970.3 Kitem/s │ 1.283 Mitem/s │ 1.313 Mitem/s │         │
│  │  ├─ 3                     22.77 ms      │ 36.4 ms       │ 30.56 ms      │ 29.89 ms      │ 100     │ 100
│  │  │                        1.317 Mitem/s │ 824.1 Kitem/s │ 981.3 Kitem/s │ 1.003 Mitem/s │         │
│  │  ├─ 4                     35.43 ms      │ 54.7 ms       │ 43.81 ms      │ 44.16 ms      │ 100     │ 100
│  │  │                        1.128 Mitem/s │ 731.1 Kitem/s │ 912.9 Kitem/s │ 905.6 Kitem/s │         │
│  │  ├─ 5                     50.13 ms      │ 74.86 ms      │ 61.81 ms      │ 62.38 ms      │ 100     │ 100
│  │  │                        997.2 Kitem/s │ 667.8 Kitem/s │ 808.8 Kitem/s │ 801.4 Kitem/s │         │
│  │  ╰─ 6                     83.95 ms      │ 118.7 ms      │ 95.93 ms      │ 97.08 ms      │ 100     │ 100
│  │                           714.6 Kitem/s │ 505.1 Kitem/s │ 625.3 Kitem/s │ 618 Kitem/s   │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     12.59 ms      │ 24.62 ms      │ 13.11 ms      │ 14 ms         │ 100     │ 100
│  │  │                        794.1 Kitem/s │ 406.1 Kitem/s │ 762.4 Kitem/s │ 714.1 Kitem/s │         │
│  │  ├─ 2                     19.34 ms      │ 40.22 ms      │ 28.92 ms      │ 29.07 ms      │ 100     │ 100
│  │  │                        1.033 Mitem/s │ 497.2 Kitem/s │ 691.4 Kitem/s │ 687.9 Kitem/s │         │
│  │  ├─ 3                     24.58 ms      │ 46.74 ms      │ 43.27 ms      │ 40.79 ms      │ 100     │ 100
│  │  │                        1.22 Mitem/s  │ 641.7 Kitem/s │ 693.2 Kitem/s │ 735.4 Kitem/s │         │
│  │  ├─ 4                     25.69 ms      │ 48.6 ms       │ 43.82 ms      │ 40.8 ms       │ 100     │ 100
│  │  │                        1.556 Mitem/s │ 823 Kitem/s   │ 912.7 Kitem/s │ 980.1 Kitem/s │         │
│  │  ├─ 5                     30.47 ms      │ 59.08 ms      │ 52.17 ms      │ 50.57 ms      │ 100     │ 100
│  │  │                        1.64 Mitem/s  │ 846.2 Kitem/s │ 958.2 Kitem/s │ 988.6 Kitem/s │         │
│  │  ╰─ 6                     34.1 ms       │ 64.14 ms      │ 59.23 ms      │ 56.79 ms      │ 100     │ 100
│  │                           1.759 Mitem/s │ 935.3 Kitem/s │ 1.012 Mitem/s │ 1.056 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.565 ms      │ 11.49 ms      │ 8.897 ms      │ 9.044 ms      │ 100     │ 100
│     │                        1.167 Mitem/s │ 869.9 Kitem/s │ 1.123 Mitem/s │ 1.105 Mitem/s │         │
│     ├─ 2                     8.835 ms      │ 20.05 ms      │ 12.11 ms      │ 11.96 ms      │ 100     │ 100
│     │                        2.263 Mitem/s │ 997.3 Kitem/s │ 1.651 Mitem/s │ 1.671 Mitem/s │         │
│     ├─ 3                     8.966 ms      │ 23.76 ms      │ 13.4 ms       │ 14.31 ms      │ 100     │ 100
│     │                        3.345 Mitem/s │ 1.262 Mitem/s │ 2.237 Mitem/s │ 2.095 Mitem/s │         │
│     ├─ 4                     8.991 ms      │ 23.9 ms       │ 14.24 ms      │ 14.76 ms      │ 100     │ 100
│     │                        4.448 Mitem/s │ 1.673 Mitem/s │ 2.807 Mitem/s │ 2.708 Mitem/s │         │
│     ├─ 5                     9.068 ms      │ 28.59 ms      │ 16.25 ms      │ 16.88 ms      │ 100     │ 100
│     │                        5.513 Mitem/s │ 1.748 Mitem/s │ 3.075 Mitem/s │ 2.96 Mitem/s  │         │
│     ╰─ 6                     9.096 ms      │ 30.13 ms      │ 16.85 ms      │ 17.94 ms      │ 100     │ 100
│                              6.595 Mitem/s │ 1.991 Mitem/s │ 3.56 Mitem/s  │ 3.343 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.096 ms      │ 5.606 ms      │ 3.181 ms      │ 3.391 ms      │ 100     │ 100
│  │  │                        3.229 Mitem/s │ 1.783 Mitem/s │ 3.142 Mitem/s │ 2.948 Mitem/s │         │
│  │  ├─ 2                     7.686 ms      │ 13.57 ms      │ 10.35 ms      │ 10.31 ms      │ 100     │ 100
│  │  │                        2.601 Mitem/s │ 1.473 Mitem/s │ 1.931 Mitem/s │ 1.938 Mitem/s │         │
│  │  ├─ 3                     14.85 ms      │ 25.52 ms      │ 20.2 ms       │ 20.05 ms      │ 100     │ 100
│  │  │                        2.018 Mitem/s │ 1.175 Mitem/s │ 1.484 Mitem/s │ 1.495 Mitem/s │         │
│  │  ├─ 4                     23.75 ms      │ 45.36 ms      │ 31.36 ms      │ 31.76 ms      │ 100     │ 100
│  │  │                        1.683 Mitem/s │ 881.7 Kitem/s │ 1.275 Mitem/s │ 1.259 Mitem/s │         │
│  │  ├─ 5                     35.52 ms      │ 58.14 ms      │ 46.68 ms      │ 46.46 ms      │ 100     │ 100
│  │  │                        1.407 Mitem/s │ 859.9 Kitem/s │ 1.07 Mitem/s  │ 1.076 Mitem/s │         │
│  │  ╰─ 6                     59.86 ms      │ 83.33 ms      │ 70.67 ms      │ 70.37 ms      │ 100     │ 100
│  │                           1.002 Mitem/s │ 720 Kitem/s   │ 848.9 Kitem/s │ 852.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     11.67 ms      │ 18.71 ms      │ 12.62 ms      │ 13.46 ms      │ 100     │ 100
│  │  │                        856.4 Kitem/s │ 534.3 Kitem/s │ 792.3 Kitem/s │ 742.8 Kitem/s │         │
│  │  ├─ 2                     20.22 ms      │ 56 ms         │ 52.88 ms      │ 48.76 ms      │ 100     │ 100
│  │  │                        988.8 Kitem/s │ 357 Kitem/s   │ 378.1 Kitem/s │ 410 Kitem/s   │         │
│  │  ├─ 3                     18.49 ms      │ 61.11 ms      │ 57.67 ms      │ 49.47 ms      │ 100     │ 100
│  │  │                        1.621 Mitem/s │ 490.8 Kitem/s │ 520.1 Kitem/s │ 606.3 Kitem/s │         │
│  │  ├─ 4                     26.49 ms      │ 66.6 ms       │ 59.59 ms      │ 55.17 ms      │ 100     │ 100
│  │  │                        1.509 Mitem/s │ 600.5 Kitem/s │ 671.1 Kitem/s │ 724.9 Kitem/s │         │
│  │  ├─ 5                     29.91 ms      │ 69.31 ms      │ 61.66 ms      │ 58.16 ms      │ 100     │ 100
│  │  │                        1.671 Mitem/s │ 721.2 Kitem/s │ 810.8 Kitem/s │ 859.6 Kitem/s │         │
│  │  ╰─ 6                     29.45 ms      │ 68 ms         │ 61.52 ms      │ 59.76 ms      │ 100     │ 100
│  │                           2.036 Mitem/s │ 882.3 Kitem/s │ 975.2 Kitem/s │ 1.003 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.253 ms      │ 11.16 ms      │ 8.442 ms      │ 8.577 ms      │ 100     │ 100
│     │                        1.211 Mitem/s │ 895.4 Kitem/s │ 1.184 Mitem/s │ 1.165 Mitem/s │         │
│     ├─ 2                     8.601 ms      │ 20.67 ms      │ 12.43 ms      │ 13.59 ms      │ 100     │ 100
│     │                        2.325 Mitem/s │ 967.1 Kitem/s │ 1.608 Mitem/s │ 1.471 Mitem/s │         │
│     ├─ 3                     8.59 ms       │ 18.49 ms      │ 15.83 ms      │ 15.27 ms      │ 100     │ 100
│     │                        3.492 Mitem/s │ 1.621 Mitem/s │ 1.894 Mitem/s │ 1.964 Mitem/s │         │
│     ├─ 4                     8.983 ms      │ 25.64 ms      │ 13.37 ms      │ 13.92 ms      │ 100     │ 100
│     │                        4.452 Mitem/s │ 1.559 Mitem/s │ 2.99 Mitem/s  │ 2.872 Mitem/s │         │
│     ├─ 5                     8.905 ms      │ 26.83 ms      │ 15.19 ms      │ 15.84 ms      │ 100     │ 100
│     │                        5.614 Mitem/s │ 1.863 Mitem/s │ 3.29 Mitem/s  │ 3.155 Mitem/s │         │
│     ╰─ 6                     8.943 ms      │ 27.67 ms      │ 15 ms         │ 15.71 ms      │ 100     │ 100
│                              6.708 Mitem/s │ 2.168 Mitem/s │ 3.999 Mitem/s │ 3.818 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.266 ms      │ 6.507 ms      │ 3.333 ms      │ 3.73 ms       │ 100     │ 100
│  │  │                        3.06 Mitem/s  │ 1.536 Mitem/s │ 2.999 Mitem/s │ 2.68 Mitem/s  │         │
│  │  ├─ 2                     7.702 ms      │ 15.21 ms      │ 10.87 ms      │ 10.83 ms      │ 100     │ 100
│  │  │                        2.596 Mitem/s │ 1.314 Mitem/s │ 1.838 Mitem/s │ 1.846 Mitem/s │         │
│  │  ├─ 3                     17.02 ms      │ 25.2 ms       │ 20.97 ms      │ 21.08 ms      │ 100     │ 100
│  │  │                        1.762 Mitem/s │ 1.19 Mitem/s  │ 1.43 Mitem/s  │ 1.422 Mitem/s │         │
│  │  ├─ 4                     25.78 ms      │ 42.15 ms      │ 32.3 ms       │ 32.48 ms      │ 100     │ 100
│  │  │                        1.551 Mitem/s │ 948.9 Kitem/s │ 1.238 Mitem/s │ 1.231 Mitem/s │         │
│  │  ├─ 5                     35.61 ms      │ 58.79 ms      │ 47.23 ms      │ 47.05 ms      │ 100     │ 100
│  │  │                        1.403 Mitem/s │ 850.3 Kitem/s │ 1.058 Mitem/s │ 1.062 Mitem/s │         │
│  │  ╰─ 6                     60.7 ms       │ 91.14 ms      │ 72.69 ms      │ 72.49 ms      │ 100     │ 100
│  │                           988.4 Kitem/s │ 658.2 Kitem/s │ 825.4 Kitem/s │ 827.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     12.9 ms       │ 18.7 ms       │ 13.39 ms      │ 13.96 ms      │ 100     │ 100
│  │  │                        774.7 Kitem/s │ 534.4 Kitem/s │ 746.5 Kitem/s │ 716 Kitem/s   │         │
│  │  ├─ 2                     22.95 ms      │ 43.08 ms      │ 41.04 ms      │ 38.69 ms      │ 100     │ 100
│  │  │                        871.1 Kitem/s │ 464.1 Kitem/s │ 487.2 Kitem/s │ 516.9 Kitem/s │         │
│  │  ├─ 3                     25.48 ms      │ 47.72 ms      │ 40.17 ms      │ 39.33 ms      │ 100     │ 100
│  │  │                        1.177 Mitem/s │ 628.5 Kitem/s │ 746.8 Kitem/s │ 762.6 Kitem/s │         │
│  │  ├─ 4                     26.44 ms      │ 54.64 ms      │ 47.08 ms      │ 46.46 ms      │ 100     │ 100
│  │  │                        1.512 Mitem/s │ 731.9 Kitem/s │ 849.4 Kitem/s │ 860.8 Kitem/s │         │
│  │  ├─ 5                     32.13 ms      │ 56.43 ms      │ 50.87 ms      │ 48.77 ms      │ 100     │ 100
│  │  │                        1.556 Mitem/s │ 885.9 Kitem/s │ 982.8 Kitem/s │ 1.025 Mitem/s │         │
│  │  ╰─ 6                     34.45 ms      │ 69.46 ms      │ 62.51 ms      │ 59.82 ms      │ 100     │ 100
│  │                           1.741 Mitem/s │ 863.7 Kitem/s │ 959.7 Kitem/s │ 1.002 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.091 ms      │ 13.34 ms      │ 9.939 ms      │ 10.22 ms      │ 100     │ 100
│     │                        1.099 Mitem/s │ 749.2 Kitem/s │ 1.006 Mitem/s │ 977.5 Kitem/s │         │
│     ├─ 2                     9.493 ms      │ 20.02 ms      │ 10.72 ms      │ 11.54 ms      │ 100     │ 100
│     │                        2.106 Mitem/s │ 998.5 Kitem/s │ 1.865 Mitem/s │ 1.731 Mitem/s │         │
│     ├─ 3                     9.492 ms      │ 20.84 ms      │ 14.75 ms      │ 14.99 ms      │ 100     │ 100
│     │                        3.16 Mitem/s  │ 1.439 Mitem/s │ 2.033 Mitem/s │ 2 Mitem/s     │         │
│     ├─ 4                     9.947 ms      │ 28.31 ms      │ 15.28 ms      │ 15.65 ms      │ 100     │ 100
│     │                        4.021 Mitem/s │ 1.412 Mitem/s │ 2.617 Mitem/s │ 2.554 Mitem/s │         │
│     ├─ 5                     10.38 ms      │ 31.95 ms      │ 16.32 ms      │ 17.03 ms      │ 100     │ 100
│     │                        4.812 Mitem/s │ 1.564 Mitem/s │ 3.062 Mitem/s │ 2.935 Mitem/s │         │
│     ╰─ 6                     11.12 ms      │ 32.81 ms      │ 19.26 ms      │ 20.32 ms      │ 100     │ 100
│                              5.394 Mitem/s │ 1.828 Mitem/s │ 3.114 Mitem/s │ 2.952 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.255 ms      │ 6.033 ms      │ 3.402 ms      │ 3.756 ms      │ 100     │ 100
│  │  │                        3.071 Mitem/s │ 1.657 Mitem/s │ 2.938 Mitem/s │ 2.661 Mitem/s │         │
│  │  ├─ 2                     7.59 ms       │ 14.82 ms      │ 10.33 ms      │ 10.6 ms       │ 100     │ 100
│  │  │                        2.634 Mitem/s │ 1.348 Mitem/s │ 1.935 Mitem/s │ 1.886 Mitem/s │         │
│  │  ├─ 3                     13.29 ms      │ 26.39 ms      │ 20.35 ms      │ 20.25 ms      │ 100     │ 100
│  │  │                        2.257 Mitem/s │ 1.136 Mitem/s │ 1.473 Mitem/s │ 1.48 Mitem/s  │         │
│  │  ├─ 4                     25.18 ms      │ 40.83 ms      │ 32.25 ms      │ 32.39 ms      │ 100     │ 100
│  │  │                        1.588 Mitem/s │ 979.6 Kitem/s │ 1.239 Mitem/s │ 1.234 Mitem/s │         │
│  │  ├─ 5                     36.02 ms      │ 61.21 ms      │ 46.14 ms      │ 46.7 ms       │ 100     │ 100
│  │  │                        1.387 Mitem/s │ 816.8 Kitem/s │ 1.083 Mitem/s │ 1.07 Mitem/s  │         │
│  │  ╰─ 6                     59.16 ms      │ 85.05 ms      │ 74.07 ms      │ 73.59 ms      │ 100     │ 100
│  │                           1.014 Mitem/s │ 705.3 Kitem/s │ 810 Kitem/s   │ 815.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     14.4 ms       │ 21.79 ms      │ 14.73 ms      │ 15.48 ms      │ 100     │ 100
│  │  │                        694.1 Kitem/s │ 458.8 Kitem/s │ 678.4 Kitem/s │ 645.7 Kitem/s │         │
│  │  ├─ 2                     19.19 ms      │ 35.22 ms      │ 27.77 ms      │ 27.45 ms      │ 100     │ 100
│  │  │                        1.041 Mitem/s │ 567.8 Kitem/s │ 720.1 Kitem/s │ 728.5 Kitem/s │         │
│  │  ├─ 3                     26.87 ms      │ 53.45 ms      │ 48.27 ms      │ 45.12 ms      │ 100     │ 100
│  │  │                        1.116 Mitem/s │ 561.1 Kitem/s │ 621.4 Kitem/s │ 664.7 Kitem/s │         │
│  │  ├─ 4                     30.33 ms      │ 58.12 ms      │ 51.82 ms      │ 49.24 ms      │ 100     │ 100
│  │  │                        1.318 Mitem/s │ 688.1 Kitem/s │ 771.8 Kitem/s │ 812.2 Kitem/s │         │
│  │  ├─ 5                     33.87 ms      │ 62.33 ms      │ 53.58 ms      │ 51.7 ms       │ 100     │ 100
│  │  │                        1.476 Mitem/s │ 802.1 Kitem/s │ 933.1 Kitem/s │ 967.1 Kitem/s │         │
│  │  ╰─ 6                     33.27 ms      │ 63.07 ms      │ 54.26 ms      │ 53.95 ms      │ 100     │ 100
│  │                           1.803 Mitem/s │ 951.2 Kitem/s │ 1.105 Mitem/s │ 1.111 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.133 ms      │ 16.55 ms      │ 10.26 ms      │ 11.42 ms      │ 100     │ 100
│     │                        1.094 Mitem/s │ 604 Kitem/s   │ 973.7 Kitem/s │ 875.5 Kitem/s │         │
│     ├─ 2                     9.482 ms      │ 20.65 ms      │ 11.07 ms      │ 11.69 ms      │ 100     │ 100
│     │                        2.109 Mitem/s │ 968.1 Kitem/s │ 1.805 Mitem/s │ 1.71 Mitem/s  │         │
│     ├─ 3                     9.411 ms      │ 20.15 ms      │ 14.23 ms      │ 14.03 ms      │ 100     │ 100
│     │                        3.187 Mitem/s │ 1.488 Mitem/s │ 2.107 Mitem/s │ 2.137 Mitem/s │         │
│     ├─ 4                     9.961 ms      │ 28.84 ms      │ 14.9 ms       │ 15.62 ms      │ 100     │ 100
│     │                        4.015 Mitem/s │ 1.386 Mitem/s │ 2.682 Mitem/s │ 2.559 Mitem/s │         │
│     ├─ 5                     11.43 ms      │ 30.75 ms      │ 16.79 ms      │ 17.71 ms      │ 100     │ 100
│     │                        4.373 Mitem/s │ 1.625 Mitem/s │ 2.977 Mitem/s │ 2.822 Mitem/s │         │
│     ╰─ 6                     10.88 ms      │ 32.9 ms       │ 17.61 ms      │ 19.21 ms      │ 100     │ 100
│                              5.51 Mitem/s  │ 1.823 Mitem/s │ 3.406 Mitem/s │ 3.122 Mitem/s │         │
```
