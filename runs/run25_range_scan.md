```text
Timer precision: 20 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.478 ms      │ 7.112 ms      │ 4.639 ms      │ 4.915 ms      │ 100     │ 100
│  │  │                        2.232 Mitem/s │ 1.405 Mitem/s │ 2.155 Mitem/s │ 2.034 Mitem/s │         │
│  │  ├─ 2                     10.69 ms      │ 23.1 ms       │ 15.25 ms      │ 15.37 ms      │ 100     │ 100
│  │  │                        1.869 Mitem/s │ 865.4 Kitem/s │ 1.311 Mitem/s │ 1.3 Mitem/s   │         │
│  │  ├─ 3                     21.65 ms      │ 42.12 ms      │ 27.78 ms      │ 28.12 ms      │ 100     │ 100
│  │  │                        1.385 Mitem/s │ 712.1 Kitem/s │ 1.079 Mitem/s │ 1.066 Mitem/s │         │
│  │  ├─ 4                     34.19 ms      │ 58.15 ms      │ 42.87 ms      │ 43.98 ms      │ 100     │ 100
│  │  │                        1.169 Mitem/s │ 687.8 Kitem/s │ 932.9 Kitem/s │ 909.4 Kitem/s │         │
│  │  ├─ 5                     52.92 ms      │ 78.83 ms      │ 61.34 ms      │ 62.42 ms      │ 100     │ 100
│  │  │                        944.6 Kitem/s │ 634.2 Kitem/s │ 815 Kitem/s   │ 801 Kitem/s   │         │
│  │  ╰─ 6                     81.24 ms      │ 124.3 ms      │ 95.25 ms      │ 95.97 ms      │ 100     │ 100
│  │                           738.5 Kitem/s │ 482.3 Kitem/s │ 629.9 Kitem/s │ 625.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.2 ms        │ 13.46 ms      │ 7.375 ms      │ 8.43 ms       │ 100     │ 100
│  │  │                        1.388 Mitem/s │ 742.6 Kitem/s │ 1.355 Mitem/s │ 1.186 Mitem/s │         │
│  │  ├─ 2                     7.564 ms      │ 17.36 ms      │ 13.76 ms      │ 12.54 ms      │ 100     │ 100
│  │  │                        2.644 Mitem/s │ 1.151 Mitem/s │ 1.452 Mitem/s │ 1.593 Mitem/s │         │
│  │  ├─ 3                     7.64 ms       │ 16.62 ms      │ 11.68 ms      │ 12.44 ms      │ 100     │ 100
│  │  │                        3.926 Mitem/s │ 1.804 Mitem/s │ 2.566 Mitem/s │ 2.409 Mitem/s │         │
│  │  ├─ 4                     8.348 ms      │ 25.43 ms      │ 14.46 ms      │ 14.15 ms      │ 100     │ 100
│  │  │                        4.791 Mitem/s │ 1.572 Mitem/s │ 2.764 Mitem/s │ 2.825 Mitem/s │         │
│  │  ├─ 5                     8.633 ms      │ 24.09 ms      │ 13.79 ms      │ 14.45 ms      │ 100     │ 100
│  │  │                        5.791 Mitem/s │ 2.075 Mitem/s │ 3.625 Mitem/s │ 3.458 Mitem/s │         │
│  │  ╰─ 6                     10.4 ms       │ 24.32 ms      │ 13.8 ms       │ 15.18 ms      │ 100     │ 100
│  │                           5.768 Mitem/s │ 2.466 Mitem/s │ 4.347 Mitem/s │ 3.952 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.923 ms      │ 3.214 ms      │ 2.019 ms      │ 2.177 ms      │ 100     │ 100
│  │  │                        5.198 Mitem/s │ 3.111 Mitem/s │ 4.951 Mitem/s │ 4.591 Mitem/s │         │
│  │  ├─ 2                     2.157 ms      │ 4.871 ms      │ 3.077 ms      │ 3.228 ms      │ 100     │ 100
│  │  │                        9.269 Mitem/s │ 4.105 Mitem/s │ 6.498 Mitem/s │ 6.194 Mitem/s │         │
│  │  ├─ 3                     2.309 ms      │ 5.834 ms      │ 3.308 ms      │ 3.505 ms      │ 100     │ 100
│  │  │                        12.98 Mitem/s │ 5.142 Mitem/s │ 9.067 Mitem/s │ 8.556 Mitem/s │         │
│  │  ├─ 4                     2.916 ms      │ 7.467 ms      │ 3.7 ms        │ 3.833 ms      │ 100     │ 100
│  │  │                        13.71 Mitem/s │ 5.356 Mitem/s │ 10.81 Mitem/s │ 10.43 Mitem/s │         │
│  │  ├─ 5                     2.948 ms      │ 7.123 ms      │ 4.021 ms      │ 3.979 ms      │ 100     │ 100
│  │  │                        16.95 Mitem/s │ 7.019 Mitem/s │ 12.43 Mitem/s │ 12.56 Mitem/s │         │
│  │  ╰─ 6                     3.053 ms      │ 11.12 ms      │ 4.547 ms      │ 5.246 ms      │ 100     │ 100
│  │                           19.64 Mitem/s │ 5.394 Mitem/s │ 13.19 Mitem/s │ 11.43 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.532 ms      │ 15.86 ms      │ 10.01 ms      │ 10.84 ms      │ 100     │ 100
│     │                        1.049 Mitem/s │ 630.3 Kitem/s │ 998.9 Kitem/s │ 922.2 Kitem/s │         │
│     ├─ 2                     9.908 ms      │ 22.21 ms      │ 11.06 ms      │ 11.77 ms      │ 100     │ 100
│     │                        2.018 Mitem/s │ 900.3 Kitem/s │ 1.807 Mitem/s │ 1.698 Mitem/s │         │
│     ├─ 3                     10.06 ms      │ 21.83 ms      │ 15.45 ms      │ 16.53 ms      │ 100     │ 100
│     │                        2.979 Mitem/s │ 1.374 Mitem/s │ 1.94 Mitem/s  │ 1.814 Mitem/s │         │
│     ├─ 4                     11 ms         │ 27.55 ms      │ 17.1 ms       │ 17.81 ms      │ 100     │ 100
│     │                        3.633 Mitem/s │ 1.451 Mitem/s │ 2.338 Mitem/s │ 2.245 Mitem/s │         │
│     ├─ 5                     10.22 ms      │ 34.19 ms      │ 17.7 ms       │ 18.96 ms      │ 100     │ 100
│     │                        4.89 Mitem/s  │ 1.462 Mitem/s │ 2.824 Mitem/s │ 2.637 Mitem/s │         │
│     ╰─ 6                     14.79 ms      │ 32.81 ms      │ 17.95 ms      │ 19.53 ms      │ 100     │ 100
│                              4.055 Mitem/s │ 1.828 Mitem/s │ 3.34 Mitem/s  │ 3.071 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.652 ms      │ 8.845 ms      │ 4.837 ms      │ 5.454 ms      │ 100     │ 100
│  │  │                        2.149 Mitem/s │ 1.13 Mitem/s  │ 2.067 Mitem/s │ 1.833 Mitem/s │         │
│  │  ├─ 2                     12.12 ms      │ 21.87 ms      │ 15.95 ms      │ 16.28 ms      │ 100     │ 100
│  │  │                        1.649 Mitem/s │ 914.1 Kitem/s │ 1.253 Mitem/s │ 1.228 Mitem/s │         │
│  │  ├─ 3                     22.85 ms      │ 41.78 ms      │ 30.21 ms      │ 30.2 ms       │ 100     │ 100
│  │  │                        1.312 Mitem/s │ 717.9 Kitem/s │ 992.9 Kitem/s │ 993 Kitem/s   │         │
│  │  ├─ 4                     36.22 ms      │ 54.57 ms      │ 43.16 ms      │ 43.66 ms      │ 100     │ 100
│  │  │                        1.104 Mitem/s │ 732.9 Kitem/s │ 926.6 Kitem/s │ 916.1 Kitem/s │         │
│  │  ├─ 5                     55.54 ms      │ 82.79 ms      │ 63.38 ms      │ 64.97 ms      │ 100     │ 100
│  │  │                        900.1 Kitem/s │ 603.8 Kitem/s │ 788.8 Kitem/s │ 769.5 Kitem/s │         │
│  │  ╰─ 6                     89.8 ms       │ 124 ms        │ 101.1 ms      │ 101.4 ms      │ 100     │ 100
│  │                           668.1 Kitem/s │ 483.6 Kitem/s │ 593.3 Kitem/s │ 591.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.131 ms      │ 10.91 ms      │ 7.282 ms      │ 7.612 ms      │ 100     │ 100
│  │  │                        1.402 Mitem/s │ 916.5 Kitem/s │ 1.373 Mitem/s │ 1.313 Mitem/s │         │
│  │  ├─ 2                     7.648 ms      │ 15.25 ms      │ 9.131 ms      │ 10.13 ms      │ 100     │ 100
│  │  │                        2.614 Mitem/s │ 1.311 Mitem/s │ 2.19 Mitem/s  │ 1.973 Mitem/s │         │
│  │  ├─ 3                     8.258 ms      │ 18.93 ms      │ 11.47 ms      │ 11.68 ms      │ 100     │ 100
│  │  │                        3.632 Mitem/s │ 1.584 Mitem/s │ 2.614 Mitem/s │ 2.566 Mitem/s │         │
│  │  ├─ 4                     8.283 ms      │ 18.7 ms       │ 12.48 ms      │ 13.03 ms      │ 100     │ 100
│  │  │                        4.828 Mitem/s │ 2.138 Mitem/s │ 3.203 Mitem/s │ 3.067 Mitem/s │         │
│  │  ├─ 5                     8.367 ms      │ 24.77 ms      │ 13.28 ms      │ 13.36 ms      │ 100     │ 100
│  │  │                        5.975 Mitem/s │ 2.017 Mitem/s │ 3.763 Mitem/s │ 3.742 Mitem/s │         │
│  │  ╰─ 6                     8.407 ms      │ 22.66 ms      │ 13.48 ms      │ 13.77 ms      │ 100     │ 100
│  │                           7.136 Mitem/s │ 2.647 Mitem/s │ 4.448 Mitem/s │ 4.355 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.84 ms       │ 3.495 ms      │ 1.921 ms      │ 2.271 ms      │ 100     │ 100
│  │  │                        5.434 Mitem/s │ 2.86 Mitem/s  │ 5.204 Mitem/s │ 4.402 Mitem/s │         │
│  │  ├─ 2                     2.128 ms      │ 4.988 ms      │ 4.045 ms      │ 3.929 ms      │ 100     │ 100
│  │  │                        9.398 Mitem/s │ 4.009 Mitem/s │ 4.943 Mitem/s │ 5.089 Mitem/s │         │
│  │  ├─ 3                     2.147 ms      │ 5.729 ms      │ 3.76 ms       │ 3.667 ms      │ 100     │ 100
│  │  │                        13.96 Mitem/s │ 5.235 Mitem/s │ 7.977 Mitem/s │ 8.18 Mitem/s  │         │
│  │  ├─ 4                     2.796 ms      │ 7.235 ms      │ 3.724 ms      │ 3.779 ms      │ 100     │ 100
│  │  │                        14.3 Mitem/s  │ 5.528 Mitem/s │ 10.73 Mitem/s │ 10.58 Mitem/s │         │
│  │  ├─ 5                     2.863 ms      │ 5.938 ms      │ 4.316 ms      │ 4.181 ms      │ 100     │ 100
│  │  │                        17.45 Mitem/s │ 8.42 Mitem/s  │ 11.58 Mitem/s │ 11.95 Mitem/s │         │
│  │  ╰─ 6                     3.018 ms      │ 7.806 ms      │ 4.354 ms      │ 4.329 ms      │ 100     │ 100
│  │                           19.87 Mitem/s │ 7.685 Mitem/s │ 13.77 Mitem/s │ 13.85 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.993 ms      │ 14.84 ms      │ 10.36 ms      │ 11.19 ms      │ 100     │ 100
│     │                        1 Mitem/s     │ 673.8 Kitem/s │ 964.8 Kitem/s │ 892.9 Kitem/s │         │
│     ├─ 2                     10.25 ms      │ 22.38 ms      │ 14.93 ms      │ 15.31 ms      │ 100     │ 100
│     │                        1.949 Mitem/s │ 893.4 Kitem/s │ 1.338 Mitem/s │ 1.306 Mitem/s │         │
│     ├─ 3                     10.56 ms      │ 21.99 ms      │ 15.66 ms      │ 17.21 ms      │ 100     │ 100
│     │                        2.838 Mitem/s │ 1.363 Mitem/s │ 1.914 Mitem/s │ 1.742 Mitem/s │         │
│     ├─ 4                     10.69 ms      │ 33.99 ms      │ 17.98 ms      │ 17.85 ms      │ 100     │ 100
│     │                        3.738 Mitem/s │ 1.176 Mitem/s │ 2.224 Mitem/s │ 2.24 Mitem/s  │         │
│     ├─ 5                     11.43 ms      │ 32.74 ms      │ 18.7 ms       │ 19.55 ms      │ 100     │ 100
│     │                        4.37 Mitem/s  │ 1.526 Mitem/s │ 2.673 Mitem/s │ 2.556 Mitem/s │         │
│     ╰─ 6                     11.25 ms      │ 33.26 ms      │ 19.48 ms      │ 21.27 ms      │ 100     │ 100
│                              5.332 Mitem/s │ 1.803 Mitem/s │ 3.079 Mitem/s │ 2.82 Mitem/s  │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.481 ms      │ 8.031 ms      │ 4.902 ms      │ 5.357 ms      │ 100     │ 100
│  │  │                        2.231 Mitem/s │ 1.245 Mitem/s │ 2.039 Mitem/s │ 1.866 Mitem/s │         │
│  │  ├─ 2                     11.04 ms      │ 20.2 ms       │ 14.57 ms      │ 14.77 ms      │ 100     │ 100
│  │  │                        1.81 Mitem/s  │ 989.9 Kitem/s │ 1.372 Mitem/s │ 1.353 Mitem/s │         │
│  │  ├─ 3                     21.98 ms      │ 38.61 ms      │ 28.65 ms      │ 28.48 ms      │ 100     │ 100
│  │  │                        1.364 Mitem/s │ 776.9 Kitem/s │ 1.046 Mitem/s │ 1.053 Mitem/s │         │
│  │  ├─ 4                     34.43 ms      │ 57.07 ms      │ 44.49 ms      │ 44.49 ms      │ 100     │ 100
│  │  │                        1.161 Mitem/s │ 700.8 Kitem/s │ 898.9 Kitem/s │ 898.9 Kitem/s │         │
│  │  ├─ 5                     54.62 ms      │ 79.78 ms      │ 63.62 ms      │ 64.34 ms      │ 100     │ 100
│  │  │                        915.4 Kitem/s │ 626.7 Kitem/s │ 785.8 Kitem/s │ 777.1 Kitem/s │         │
│  │  ╰─ 6                     87.93 ms      │ 109.3 ms      │ 98.6 ms       │ 98.56 ms      │ 100     │ 100
│  │                           682.3 Kitem/s │ 548.6 Kitem/s │ 608.5 Kitem/s │ 608.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.233 ms      │ 12.78 ms      │ 7.374 ms      │ 8.073 ms      │ 100     │ 100
│  │  │                        1.382 Mitem/s │ 782.3 Kitem/s │ 1.356 Mitem/s │ 1.238 Mitem/s │         │
│  │  ├─ 2                     7.669 ms      │ 16.24 ms      │ 13.77 ms      │ 12.95 ms      │ 100     │ 100
│  │  │                        2.607 Mitem/s │ 1.23 Mitem/s  │ 1.452 Mitem/s │ 1.544 Mitem/s │         │
│  │  ├─ 3                     7.79 ms       │ 17.98 ms      │ 14.08 ms      │ 13.48 ms      │ 100     │ 100
│  │  │                        3.85 Mitem/s  │ 1.667 Mitem/s │ 2.129 Mitem/s │ 2.225 Mitem/s │         │
│  │  ├─ 4                     8.566 ms      │ 23.41 ms      │ 14.22 ms      │ 14.11 ms      │ 100     │ 100
│  │  │                        4.669 Mitem/s │ 1.708 Mitem/s │ 2.812 Mitem/s │ 2.834 Mitem/s │         │
│  │  ├─ 5                     8.514 ms      │ 24.81 ms      │ 13.49 ms      │ 14.21 ms      │ 100     │ 100
│  │  │                        5.872 Mitem/s │ 2.014 Mitem/s │ 3.703 Mitem/s │ 3.518 Mitem/s │         │
│  │  ╰─ 6                     8.776 ms      │ 23.26 ms      │ 13.66 ms      │ 14.07 ms      │ 100     │ 100
│  │                           6.836 Mitem/s │ 2.578 Mitem/s │ 4.391 Mitem/s │ 4.261 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.91 ms       │ 3.895 ms      │ 2.034 ms      │ 2.25 ms       │ 100     │ 100
│  │  │                        5.233 Mitem/s │ 2.567 Mitem/s │ 4.914 Mitem/s │ 4.443 Mitem/s │         │
│  │  ├─ 2                     2.156 ms      │ 4.821 ms      │ 4.061 ms      │ 3.813 ms      │ 100     │ 100
│  │  │                        9.275 Mitem/s │ 4.148 Mitem/s │ 4.924 Mitem/s │ 5.244 Mitem/s │         │
│  │  ├─ 3                     2.234 ms      │ 5.003 ms      │ 3.402 ms      │ 3.609 ms      │ 100     │ 100
│  │  │                        13.42 Mitem/s │ 5.995 Mitem/s │ 8.816 Mitem/s │ 8.311 Mitem/s │         │
│  │  ├─ 4                     2.894 ms      │ 6.812 ms      │ 3.757 ms      │ 3.843 ms      │ 100     │ 100
│  │  │                        13.81 Mitem/s │ 5.871 Mitem/s │ 10.64 Mitem/s │ 10.4 Mitem/s  │         │
│  │  ├─ 5                     2.934 ms      │ 6.869 ms      │ 3.419 ms      │ 3.697 ms      │ 100     │ 100
│  │  │                        17.03 Mitem/s │ 7.278 Mitem/s │ 14.62 Mitem/s │ 13.52 Mitem/s │         │
│  │  ╰─ 6                     3.055 ms      │ 6.824 ms      │ 4.299 ms      │ 4.133 ms      │ 100     │ 100
│  │                           19.63 Mitem/s │ 8.792 Mitem/s │ 13.95 Mitem/s │ 14.51 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.615 ms      │ 16.17 ms      │ 10.84 ms      │ 11.39 ms      │ 100     │ 100
│     │                        1.04 Mitem/s  │ 618.1 Kitem/s │ 922 Kitem/s   │ 877.3 Kitem/s │         │
│     ├─ 2                     9.805 ms      │ 22.4 ms       │ 13.87 ms      │ 13.67 ms      │ 100     │ 100
│     │                        2.039 Mitem/s │ 892.5 Kitem/s │ 1.441 Mitem/s │ 1.462 Mitem/s │         │
│     ├─ 3                     10.11 ms      │ 21.36 ms      │ 15.41 ms      │ 16.14 ms      │ 100     │ 100
│     │                        2.965 Mitem/s │ 1.404 Mitem/s │ 1.945 Mitem/s │ 1.858 Mitem/s │         │
│     ├─ 4                     10.24 ms      │ 26.12 ms      │ 15.83 ms      │ 16.36 ms      │ 100     │ 100
│     │                        3.906 Mitem/s │ 1.531 Mitem/s │ 2.526 Mitem/s │ 2.444 Mitem/s │         │
│     ├─ 5                     10.62 ms      │ 30.52 ms      │ 17.51 ms      │ 17.76 ms      │ 100     │ 100
│     │                        4.706 Mitem/s │ 1.638 Mitem/s │ 2.855 Mitem/s │ 2.814 Mitem/s │         │
│     ╰─ 6                     10.96 ms      │ 31.53 ms      │ 17.83 ms      │ 19.39 ms      │ 100     │ 100
│                              5.47 Mitem/s  │ 1.902 Mitem/s │ 3.364 Mitem/s │ 3.093 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.535 ms      │ 7.558 ms      │ 4.778 ms      │ 5.255 ms      │ 100     │ 100
│  │  │                        2.204 Mitem/s │ 1.322 Mitem/s │ 2.092 Mitem/s │ 1.902 Mitem/s │         │
│  │  ├─ 2                     11.46 ms      │ 21.82 ms      │ 14.59 ms      │ 14.79 ms      │ 100     │ 100
│  │  │                        1.744 Mitem/s │ 916.5 Kitem/s │ 1.369 Mitem/s │ 1.352 Mitem/s │         │
│  │  ├─ 3                     22.82 ms      │ 36.62 ms      │ 28.05 ms      │ 28.55 ms      │ 100     │ 100
│  │  │                        1.314 Mitem/s │ 819.1 Kitem/s │ 1.069 Mitem/s │ 1.05 Mitem/s  │         │
│  │  ├─ 4                     34.51 ms      │ 56.93 ms      │ 44.73 ms      │ 45.43 ms      │ 100     │ 100
│  │  │                        1.158 Mitem/s │ 702.5 Kitem/s │ 894 Kitem/s   │ 880.4 Kitem/s │         │
│  │  ├─ 5                     53.08 ms      │ 72.48 ms      │ 61.26 ms      │ 61.36 ms      │ 100     │ 100
│  │  │                        941.8 Kitem/s │ 689.7 Kitem/s │ 816.1 Kitem/s │ 814.7 Kitem/s │         │
│  │  ╰─ 6                     86.44 ms      │ 111.6 ms      │ 98.51 ms      │ 98.53 ms      │ 100     │ 100
│  │                           694 Kitem/s   │ 537.2 Kitem/s │ 609 Kitem/s   │ 608.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.288 ms      │ 10.9 ms       │ 7.665 ms      │ 8.014 ms      │ 100     │ 100
│  │  │                        1.371 Mitem/s │ 917.1 Kitem/s │ 1.304 Mitem/s │ 1.247 Mitem/s │         │
│  │  ├─ 2                     7.709 ms      │ 16.8 ms       │ 9.286 ms      │ 10.38 ms      │ 100     │ 100
│  │  │                        2.594 Mitem/s │ 1.19 Mitem/s  │ 2.153 Mitem/s │ 1.926 Mitem/s │         │
│  │  ├─ 3                     7.935 ms      │ 16.92 ms      │ 13.61 ms      │ 12.91 ms      │ 100     │ 100
│  │  │                        3.78 Mitem/s  │ 1.772 Mitem/s │ 2.202 Mitem/s │ 2.322 Mitem/s │         │
│  │  ├─ 4                     8.476 ms      │ 23.34 ms      │ 13.8 ms       │ 13.19 ms      │ 100     │ 100
│  │  │                        4.718 Mitem/s │ 1.713 Mitem/s │ 2.896 Mitem/s │ 3.03 Mitem/s  │         │
│  │  ├─ 5                     8.512 ms      │ 23.34 ms      │ 13.52 ms      │ 13.48 ms      │ 100     │ 100
│  │  │                        5.873 Mitem/s │ 2.142 Mitem/s │ 3.696 Mitem/s │ 3.707 Mitem/s │         │
│  │  ╰─ 6                     8.747 ms      │ 21.97 ms      │ 13.58 ms      │ 13.98 ms      │ 100     │ 100
│  │                           6.858 Mitem/s │ 2.73 Mitem/s  │ 4.417 Mitem/s │ 4.291 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.929 ms      │ 3.409 ms      │ 2 ms          │ 2.081 ms      │ 100     │ 100
│  │  │                        5.183 Mitem/s │ 2.932 Mitem/s │ 4.999 Mitem/s │ 4.803 Mitem/s │         │
│  │  ├─ 2                     2.168 ms      │ 4.911 ms      │ 3.992 ms      │ 3.826 ms      │ 100     │ 100
│  │  │                        9.224 Mitem/s │ 4.071 Mitem/s │ 5.009 Mitem/s │ 5.226 Mitem/s │         │
│  │  ├─ 3                     2.212 ms      │ 5.499 ms      │ 3.865 ms      │ 3.896 ms      │ 100     │ 100
│  │  │                        13.55 Mitem/s │ 5.454 Mitem/s │ 7.761 Mitem/s │ 7.699 Mitem/s │         │
│  │  ├─ 4                     2.901 ms      │ 5.082 ms      │ 4.286 ms      │ 4.039 ms      │ 100     │ 100
│  │  │                        13.78 Mitem/s │ 7.87 Mitem/s  │ 9.332 Mitem/s │ 9.902 Mitem/s │         │
│  │  ├─ 5                     2.973 ms      │ 6.883 ms      │ 4.178 ms      │ 4.062 ms      │ 100     │ 100
│  │  │                        16.81 Mitem/s │ 7.263 Mitem/s │ 11.96 Mitem/s │ 12.3 Mitem/s  │         │
│  │  ╰─ 6                     3.109 ms      │ 7.753 ms      │ 4.693 ms      │ 4.751 ms      │ 100     │ 100
│  │                           19.29 Mitem/s │ 7.738 Mitem/s │ 12.78 Mitem/s │ 12.62 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.735 ms      │ 16.01 ms      │ 10.23 ms      │ 10.6 ms       │ 100     │ 100
│     │                        1.027 Mitem/s │ 624.3 Kitem/s │ 976.7 Kitem/s │ 942.7 Kitem/s │         │
│     ├─ 2                     9.946 ms      │ 20.09 ms      │ 11.09 ms      │ 12.54 ms      │ 100     │ 100
│     │                        2.01 Mitem/s  │ 995.4 Kitem/s │ 1.801 Mitem/s │ 1.593 Mitem/s │         │
│     ├─ 3                     9.979 ms      │ 20.8 ms       │ 15.26 ms      │ 15.42 ms      │ 100     │ 100
│     │                        3.006 Mitem/s │ 1.441 Mitem/s │ 1.964 Mitem/s │ 1.944 Mitem/s │         │
│     ├─ 4                     10.24 ms      │ 25.09 ms      │ 15.49 ms      │ 16.29 ms      │ 100     │ 100
│     │                        3.904 Mitem/s │ 1.593 Mitem/s │ 2.581 Mitem/s │ 2.455 Mitem/s │         │
│     ├─ 5                     10.35 ms      │ 35.05 ms      │ 17.2 ms       │ 17.29 ms      │ 100     │ 100
│     │                        4.829 Mitem/s │ 1.426 Mitem/s │ 2.906 Mitem/s │ 2.89 Mitem/s  │         │
│     ╰─ 6                     11.66 ms      │ 32.84 ms      │ 17.97 ms      │ 20.14 ms      │ 100     │ 100
│                              5.142 Mitem/s │ 1.826 Mitem/s │ 3.338 Mitem/s │ 2.978 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.872 ms      │ 6.577 ms      │ 4.024 ms      │ 4.261 ms      │ 100     │ 100
│  │  │                        2.582 Mitem/s │ 1.52 Mitem/s  │ 2.484 Mitem/s │ 2.346 Mitem/s │         │
│  │  ├─ 2                     9.149 ms      │ 18 ms         │ 13.45 ms      │ 13.23 ms      │ 100     │ 100
│  │  │                        2.185 Mitem/s │ 1.11 Mitem/s  │ 1.486 Mitem/s │ 1.51 Mitem/s  │         │
│  │  ├─ 3                     16.68 ms      │ 30.84 ms      │ 23.68 ms      │ 23.45 ms      │ 100     │ 100
│  │  │                        1.798 Mitem/s │ 972.5 Kitem/s │ 1.266 Mitem/s │ 1.278 Mitem/s │         │
│  │  ├─ 4                     31.18 ms      │ 56.73 ms      │ 39.1 ms       │ 39.58 ms      │ 100     │ 100
│  │  │                        1.282 Mitem/s │ 705 Kitem/s   │ 1.022 Mitem/s │ 1.01 Mitem/s  │         │
│  │  ├─ 5                     45.64 ms      │ 82.04 ms      │ 57.3 ms       │ 58.3 ms       │ 100     │ 100
│  │  │                        1.095 Mitem/s │ 609.3 Kitem/s │ 872.5 Kitem/s │ 857.5 Kitem/s │         │
│  │  ╰─ 6                     74.98 ms      │ 113.2 ms      │ 87.57 ms      │ 88.11 ms      │ 100     │ 100
│  │                           800.1 Kitem/s │ 529.6 Kitem/s │ 685.1 Kitem/s │ 680.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.661 ms      │ 13.13 ms      │ 6.838 ms      │ 7.251 ms      │ 100     │ 100
│  │  │                        1.501 Mitem/s │ 761.5 Kitem/s │ 1.462 Mitem/s │ 1.379 Mitem/s │         │
│  │  ├─ 2                     7.003 ms      │ 15.01 ms      │ 10.02 ms      │ 10.08 ms      │ 100     │ 100
│  │  │                        2.855 Mitem/s │ 1.332 Mitem/s │ 1.994 Mitem/s │ 1.983 Mitem/s │         │
│  │  ├─ 3                     6.997 ms      │ 14.37 ms      │ 9.51 ms       │ 10.05 ms      │ 100     │ 100
│  │  │                        4.287 Mitem/s │ 2.087 Mitem/s │ 3.154 Mitem/s │ 2.983 Mitem/s │         │
│  │  ├─ 4                     7.78 ms       │ 21.87 ms      │ 11.02 ms      │ 11.24 ms      │ 100     │ 100
│  │  │                        5.14 Mitem/s  │ 1.828 Mitem/s │ 3.629 Mitem/s │ 3.556 Mitem/s │         │
│  │  ├─ 5                     7.846 ms      │ 23.71 ms      │ 12.39 ms      │ 12.83 ms      │ 100     │ 100
│  │  │                        6.371 Mitem/s │ 2.108 Mitem/s │ 4.034 Mitem/s │ 3.894 Mitem/s │         │
│  │  ╰─ 6                     7.887 ms      │ 21.18 ms      │ 12.37 ms      │ 12.31 ms      │ 100     │ 100
│  │                           7.607 Mitem/s │ 2.832 Mitem/s │ 4.847 Mitem/s │ 4.871 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.965 ms      │ 3.912 ms      │ 2.089 ms      │ 2.368 ms      │ 100     │ 100
│  │  │                        5.086 Mitem/s │ 2.555 Mitem/s │ 4.785 Mitem/s │ 4.221 Mitem/s │         │
│  │  ├─ 2                     2.177 ms      │ 5.167 ms      │ 3.619 ms      │ 3.641 ms      │ 100     │ 100
│  │  │                        9.184 Mitem/s │ 3.87 Mitem/s  │ 5.524 Mitem/s │ 5.492 Mitem/s │         │
│  │  ├─ 3                     2.396 ms      │ 5.237 ms      │ 4.404 ms      │ 4.118 ms      │ 100     │ 100
│  │  │                        12.51 Mitem/s │ 5.727 Mitem/s │ 6.81 Mitem/s  │ 7.285 Mitem/s │         │
│  │  ├─ 4                     2.411 ms      │ 7.788 ms      │ 4.134 ms      │ 4.106 ms      │ 100     │ 100
│  │  │                        16.58 Mitem/s │ 5.135 Mitem/s │ 9.674 Mitem/s │ 9.74 Mitem/s  │         │
│  │  ├─ 5                     2.708 ms      │ 6.093 ms      │ 4.055 ms      │ 3.862 ms      │ 100     │ 100
│  │  │                        18.46 Mitem/s │ 8.205 Mitem/s │ 12.32 Mitem/s │ 12.94 Mitem/s │         │
│  │  ╰─ 6                     3.084 ms      │ 7.819 ms      │ 4.506 ms      │ 4.353 ms      │ 100     │ 100
│  │                           19.45 Mitem/s │ 7.673 Mitem/s │ 13.31 Mitem/s │ 13.78 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.848 ms      │ 13.36 ms      │ 10.57 ms      │ 11 ms         │ 100     │ 100
│     │                        1.13 Mitem/s  │ 748.1 Kitem/s │ 945.6 Kitem/s │ 908.6 Kitem/s │         │
│     ├─ 2                     9.178 ms      │ 21.18 ms      │ 12.85 ms      │ 12.86 ms      │ 100     │ 100
│     │                        2.178 Mitem/s │ 943.9 Kitem/s │ 1.556 Mitem/s │ 1.554 Mitem/s │         │
│     ├─ 3                     9.221 ms      │ 19.8 ms       │ 14.6 ms       │ 15.19 ms      │ 100     │ 100
│     │                        3.253 Mitem/s │ 1.514 Mitem/s │ 2.054 Mitem/s │ 1.974 Mitem/s │         │
│     ├─ 4                     9.335 ms      │ 28.99 ms      │ 15.98 ms      │ 15.75 ms      │ 100     │ 100
│     │                        4.284 Mitem/s │ 1.379 Mitem/s │ 2.502 Mitem/s │ 2.539 Mitem/s │         │
│     ├─ 5                     9.808 ms      │ 28.06 ms      │ 16.4 ms       │ 16.39 ms      │ 100     │ 100
│     │                        5.097 Mitem/s │ 1.781 Mitem/s │ 3.048 Mitem/s │ 3.048 Mitem/s │         │
│     ╰─ 6                     9.75 ms       │ 29.67 ms      │ 16.56 ms      │ 18.47 ms      │ 100     │ 100
│                              6.153 Mitem/s │ 2.021 Mitem/s │ 3.621 Mitem/s │ 3.247 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.599 ms      │ 5.981 ms      │ 3.702 ms      │ 3.987 ms      │ 100     │ 100
│  │  │                        2.778 Mitem/s │ 1.671 Mitem/s │ 2.7 Mitem/s   │ 2.507 Mitem/s │         │
│  │  ├─ 2                     9.282 ms      │ 16.68 ms      │ 12.33 ms      │ 12.17 ms      │ 100     │ 100
│  │  │                        2.154 Mitem/s │ 1.198 Mitem/s │ 1.621 Mitem/s │ 1.642 Mitem/s │         │
│  │  ├─ 3                     15.55 ms      │ 29.22 ms      │ 24.28 ms      │ 23.82 ms      │ 100     │ 100
│  │  │                        1.928 Mitem/s │ 1.026 Mitem/s │ 1.235 Mitem/s │ 1.259 Mitem/s │         │
│  │  ├─ 4                     29.09 ms      │ 45.47 ms      │ 35.61 ms      │ 36.35 ms      │ 100     │ 100
│  │  │                        1.374 Mitem/s │ 879.6 Kitem/s │ 1.122 Mitem/s │ 1.1 Mitem/s   │         │
│  │  ├─ 5                     39.2 ms       │ 65.65 ms      │ 49.19 ms      │ 50.23 ms      │ 100     │ 100
│  │  │                        1.275 Mitem/s │ 761.5 Kitem/s │ 1.016 Mitem/s │ 995.2 Kitem/s │         │
│  │  ╰─ 6                     63.25 ms      │ 89.62 ms      │ 76.3 ms       │ 76.2 ms       │ 100     │ 100
│  │                           948.5 Kitem/s │ 669.4 Kitem/s │ 786.3 Kitem/s │ 787.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.699 ms      │ 14.53 ms      │ 10.34 ms      │ 10.83 ms      │ 100     │ 100
│  │  │                        1.03 Mitem/s  │ 687.8 Kitem/s │ 966.8 Kitem/s │ 923.1 Kitem/s │         │
│  │  ├─ 2                     9.838 ms      │ 18.84 ms      │ 10.77 ms      │ 11.38 ms      │ 100     │ 100
│  │  │                        2.032 Mitem/s │ 1.061 Mitem/s │ 1.856 Mitem/s │ 1.757 Mitem/s │         │
│  │  ├─ 3                     10.13 ms      │ 21 ms         │ 15.01 ms      │ 15.41 ms      │ 100     │ 100
│  │  │                        2.959 Mitem/s │ 1.428 Mitem/s │ 1.998 Mitem/s │ 1.945 Mitem/s │         │
│  │  ├─ 4                     10.24 ms      │ 29.59 ms      │ 17.48 ms      │ 17.51 ms      │ 100     │ 100
│  │  │                        3.904 Mitem/s │ 1.351 Mitem/s │ 2.287 Mitem/s │ 2.283 Mitem/s │         │
│  │  ├─ 5                     10.71 ms      │ 32.84 ms      │ 18.78 ms      │ 20.17 ms      │ 100     │ 100
│  │  │                        4.666 Mitem/s │ 1.522 Mitem/s │ 2.662 Mitem/s │ 2.477 Mitem/s │         │
│  │  ╰─ 6                     11.15 ms      │ 32.7 ms       │ 18.84 ms      │ 20.31 ms      │ 100     │ 100
│  │                           5.38 Mitem/s  │ 1.834 Mitem/s │ 3.184 Mitem/s │ 2.953 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.937 ms      │ 4.158 ms      │ 1.991 ms      │ 2.195 ms      │ 100     │ 100
│  │  │                        5.161 Mitem/s │ 2.404 Mitem/s │ 5.022 Mitem/s │ 4.555 Mitem/s │         │
│  │  ├─ 2                     2.08 ms       │ 4.529 ms      │ 3.072 ms      │ 3.122 ms      │ 100     │ 100
│  │  │                        9.612 Mitem/s │ 4.415 Mitem/s │ 6.51 Mitem/s  │ 6.405 Mitem/s │         │
│  │  ├─ 3                     2.21 ms       │ 5.904 ms      │ 3.109 ms      │ 3.328 ms      │ 100     │ 100
│  │  │                        13.56 Mitem/s │ 5.08 Mitem/s  │ 9.648 Mitem/s │ 9.012 Mitem/s │         │
│  │  ├─ 4                     2.929 ms      │ 7.591 ms      │ 4.163 ms      │ 4.421 ms      │ 100     │ 100
│  │  │                        13.65 Mitem/s │ 5.269 Mitem/s │ 9.608 Mitem/s │ 9.046 Mitem/s │         │
│  │  ├─ 5                     2.608 ms      │ 7.64 ms       │ 4.33 ms       │ 4.253 ms      │ 100     │ 100
│  │  │                        19.16 Mitem/s │ 6.544 Mitem/s │ 11.54 Mitem/s │ 11.75 Mitem/s │         │
│  │  ╰─ 6                     3.122 ms      │ 6.824 ms      │ 4.293 ms      │ 4.244 ms      │ 100     │ 100
│  │                           19.21 Mitem/s │ 8.791 Mitem/s │ 13.97 Mitem/s │ 14.13 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.088 ms      │ 12.69 ms      │ 9.507 ms      │ 9.922 ms      │ 100     │ 100
│     │                        1.1 Mitem/s   │ 787.7 Kitem/s │ 1.051 Mitem/s │ 1.007 Mitem/s │         │
│     ├─ 2                     9.248 ms      │ 18.16 ms      │ 10.59 ms      │ 11.6 ms       │ 100     │ 100
│     │                        2.162 Mitem/s │ 1.101 Mitem/s │ 1.887 Mitem/s │ 1.723 Mitem/s │         │
│     ├─ 3                     9.393 ms      │ 25.14 ms      │ 14 ms         │ 14.1 ms       │ 100     │ 100
│     │                        3.193 Mitem/s │ 1.193 Mitem/s │ 2.141 Mitem/s │ 2.127 Mitem/s │         │
│     ├─ 4                     9.588 ms      │ 31.7 ms       │ 15.58 ms      │ 15.95 ms      │ 100     │ 100
│     │                        4.171 Mitem/s │ 1.261 Mitem/s │ 2.566 Mitem/s │ 2.507 Mitem/s │         │
│     ├─ 5                     11.56 ms      │ 32.72 ms      │ 17.69 ms      │ 18.93 ms      │ 100     │ 100
│     │                        4.323 Mitem/s │ 1.527 Mitem/s │ 2.825 Mitem/s │ 2.64 Mitem/s  │         │
│     ╰─ 6                     11.37 ms      │ 31.15 ms      │ 18.26 ms      │ 19.45 ms      │ 100     │ 100
│                              5.275 Mitem/s │ 1.925 Mitem/s │ 3.285 Mitem/s │ 3.083 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.573 ms      │ 6.744 ms      │ 3.651 ms      │ 4.085 ms      │ 100     │ 100
│  │  │                        2.798 Mitem/s │ 1.482 Mitem/s │ 2.738 Mitem/s │ 2.447 Mitem/s │         │
│  │  ├─ 2                     9.229 ms      │ 16.59 ms      │ 11.18 ms      │ 11.53 ms      │ 100     │ 100
│  │  │                        2.166 Mitem/s │ 1.205 Mitem/s │ 1.787 Mitem/s │ 1.734 Mitem/s │         │
│  │  ├─ 3                     17.23 ms      │ 29.98 ms      │ 24.42 ms      │ 24.39 ms      │ 100     │ 100
│  │  │                        1.74 Mitem/s  │ 1 Mitem/s     │ 1.228 Mitem/s │ 1.229 Mitem/s │         │
│  │  ├─ 4                     28.24 ms      │ 45.64 ms      │ 36 ms         │ 36.07 ms      │ 100     │ 100
│  │  │                        1.416 Mitem/s │ 876.3 Kitem/s │ 1.11 Mitem/s  │ 1.108 Mitem/s │         │
│  │  ├─ 5                     39.56 ms      │ 60.16 ms      │ 50.07 ms      │ 49.93 ms      │ 100     │ 100
│  │  │                        1.263 Mitem/s │ 831 Kitem/s   │ 998.5 Kitem/s │ 1.001 Mitem/s │         │
│  │  ╰─ 6                     65.38 ms      │ 110 ms        │ 76.48 ms      │ 77.04 ms      │ 100     │ 100
│  │                           917.7 Kitem/s │ 545.4 Kitem/s │ 784.5 Kitem/s │ 778.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.171 ms      │ 9.821 ms      │ 7.38 ms       │ 7.677 ms      │ 100     │ 100
│  │  │                        1.394 Mitem/s │ 1.018 Mitem/s │ 1.355 Mitem/s │ 1.302 Mitem/s │         │
│  │  ├─ 2                     7.486 ms      │ 16.4 ms       │ 9.881 ms      │ 10.53 ms      │ 100     │ 100
│  │  │                        2.671 Mitem/s │ 1.219 Mitem/s │ 2.023 Mitem/s │ 1.898 Mitem/s │         │
│  │  ├─ 3                     8.443 ms      │ 17.65 ms      │ 14.29 ms      │ 13.92 ms      │ 100     │ 100
│  │  │                        3.552 Mitem/s │ 1.699 Mitem/s │ 2.098 Mitem/s │ 2.154 Mitem/s │         │
│  │  ├─ 4                     8.355 ms      │ 19.27 ms      │ 11.93 ms      │ 12.54 ms      │ 100     │ 100
│  │  │                        4.787 Mitem/s │ 2.075 Mitem/s │ 3.35 Mitem/s  │ 3.187 Mitem/s │         │
│  │  ├─ 5                     8.293 ms      │ 25.7 ms       │ 13.41 ms      │ 13.16 ms      │ 100     │ 100
│  │  │                        6.028 Mitem/s │ 1.944 Mitem/s │ 3.728 Mitem/s │ 3.799 Mitem/s │         │
│  │  ╰─ 6                     8.912 ms      │ 22.1 ms       │ 13.49 ms      │ 14.27 ms      │ 100     │ 100
│  │                           6.732 Mitem/s │ 2.714 Mitem/s │ 4.444 Mitem/s │ 4.202 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.942 ms      │ 3.635 ms      │ 1.998 ms      │ 2.215 ms      │ 100     │ 100
│  │  │                        5.146 Mitem/s │ 2.75 Mitem/s  │ 5.004 Mitem/s │ 4.513 Mitem/s │         │
│  │  ├─ 2                     3.032 ms      │ 5.034 ms      │ 4.016 ms      │ 3.904 ms      │ 100     │ 100
│  │  │                        6.594 Mitem/s │ 3.972 Mitem/s │ 4.978 Mitem/s │ 5.122 Mitem/s │         │
│  │  ├─ 3                     2.201 ms      │ 5.519 ms      │ 4.322 ms      │ 4.103 ms      │ 100     │ 100
│  │  │                        13.62 Mitem/s │ 5.435 Mitem/s │ 6.94 Mitem/s  │ 7.311 Mitem/s │         │
│  │  ├─ 4                     2.886 ms      │ 5.158 ms      │ 4.017 ms      │ 3.813 ms      │ 100     │ 100
│  │  │                        13.85 Mitem/s │ 7.753 Mitem/s │ 9.955 Mitem/s │ 10.49 Mitem/s │         │
│  │  ├─ 5                     2.942 ms      │ 7.345 ms      │ 3.9 ms        │ 3.913 ms      │ 100     │ 100
│  │  │                        16.99 Mitem/s │ 6.807 Mitem/s │ 12.81 Mitem/s │ 12.77 Mitem/s │         │
│  │  ╰─ 6                     3.056 ms      │ 8.253 ms      │ 3.53 ms       │ 3.929 ms      │ 100     │ 100
│  │                           19.62 Mitem/s │ 7.27 Mitem/s  │ 16.99 Mitem/s │ 15.27 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.29 ms       │ 14.23 ms      │ 9.75 ms       │ 10.49 ms      │ 100     │ 100
│     │                        1.076 Mitem/s │ 702.6 Kitem/s │ 1.025 Mitem/s │ 952.4 Kitem/s │         │
│     ├─ 2                     9.499 ms      │ 19.94 ms      │ 11.79 ms      │ 12.25 ms      │ 100     │ 100
│     │                        2.105 Mitem/s │ 1.002 Mitem/s │ 1.695 Mitem/s │ 1.632 Mitem/s │         │
│     ├─ 3                     9.775 ms      │ 22.09 ms      │ 14.49 ms      │ 15.27 ms      │ 100     │ 100
│     │                        3.068 Mitem/s │ 1.357 Mitem/s │ 2.069 Mitem/s │ 1.963 Mitem/s │         │
│     ├─ 4                     9.545 ms      │ 30.75 ms      │ 15.03 ms      │ 15.68 ms      │ 100     │ 100
│     │                        4.19 Mitem/s  │ 1.3 Mitem/s   │ 2.659 Mitem/s │ 2.549 Mitem/s │         │
│     ├─ 5                     10.11 ms      │ 30.14 ms      │ 16.84 ms      │ 17.39 ms      │ 100     │ 100
│     │                        4.943 Mitem/s │ 1.658 Mitem/s │ 2.968 Mitem/s │ 2.873 Mitem/s │         │
│     ╰─ 6                     13.28 ms      │ 31.6 ms       │ 18.01 ms      │ 19.46 ms      │ 100     │ 100
│                              4.517 Mitem/s │ 1.898 Mitem/s │ 3.33 Mitem/s  │ 3.082 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.568 ms      │ 7.128 ms      │ 4.671 ms      │ 4.774 ms      │ 100     │ 100
│  │  │                        2.188 Mitem/s │ 1.402 Mitem/s │ 2.14 Mitem/s  │ 2.094 Mitem/s │         │
│  │  ├─ 2                     11.92 ms      │ 20.33 ms      │ 16.55 ms      │ 16.4 ms       │ 100     │ 100
│  │  │                        1.676 Mitem/s │ 983.5 Kitem/s │ 1.208 Mitem/s │ 1.219 Mitem/s │         │
│  │  ├─ 3                     20.19 ms      │ 36.42 ms      │ 28.05 ms      │ 27.92 ms      │ 100     │ 100
│  │  │                        1.485 Mitem/s │ 823.6 Kitem/s │ 1.069 Mitem/s │ 1.074 Mitem/s │         │
│  │  ├─ 4                     34.73 ms      │ 53.87 ms      │ 42.85 ms      │ 43.32 ms      │ 100     │ 100
│  │  │                        1.151 Mitem/s │ 742.5 Kitem/s │ 933.4 Kitem/s │ 923.2 Kitem/s │         │
│  │  ├─ 5                     53.16 ms      │ 75.58 ms      │ 62.21 ms      │ 62.56 ms      │ 100     │ 100
│  │  │                        940.3 Kitem/s │ 661.5 Kitem/s │ 803.6 Kitem/s │ 799.1 Kitem/s │         │
│  │  ╰─ 6                     86.83 ms      │ 149.5 ms      │ 101.2 ms      │ 101.5 ms      │ 100     │ 100
│  │                           691 Kitem/s   │ 401.3 Kitem/s │ 592.7 Kitem/s │ 590.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     6.709 ms      │ 10.34 ms      │ 6.864 ms      │ 7.237 ms      │ 100     │ 100
│  │  │                        1.49 Mitem/s  │ 966.8 Kitem/s │ 1.456 Mitem/s │ 1.381 Mitem/s │         │
│  │  ├─ 2                     7.085 ms      │ 14.65 ms      │ 10.6 ms       │ 11.08 ms      │ 100     │ 100
│  │  │                        2.822 Mitem/s │ 1.365 Mitem/s │ 1.885 Mitem/s │ 1.803 Mitem/s │         │
│  │  ├─ 3                     7.483 ms      │ 15.49 ms      │ 10.98 ms      │ 11.46 ms      │ 100     │ 100
│  │  │                        4.009 Mitem/s │ 1.936 Mitem/s │ 2.73 Mitem/s  │ 2.615 Mitem/s │         │
│  │  ├─ 4                     7.911 ms      │ 23.34 ms      │ 12 ms         │ 11.48 ms      │ 100     │ 100
│  │  │                        5.056 Mitem/s │ 1.713 Mitem/s │ 3.332 Mitem/s │ 3.482 Mitem/s │         │
│  │  ├─ 5                     7.853 ms      │ 20.85 ms      │ 12.59 ms      │ 12.48 ms      │ 100     │ 100
│  │  │                        6.366 Mitem/s │ 2.397 Mitem/s │ 3.969 Mitem/s │ 4.003 Mitem/s │         │
│  │  ╰─ 6                     8.299 ms      │ 21.32 ms      │ 12.82 ms      │ 13.55 ms      │ 100     │ 100
│  │                           7.229 Mitem/s │ 2.813 Mitem/s │ 4.677 Mitem/s │ 4.427 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.926 ms      │ 4.266 ms      │ 2.013 ms      │ 2.27 ms       │ 100     │ 100
│  │  │                        5.19 Mitem/s  │ 2.343 Mitem/s │ 4.967 Mitem/s │ 4.403 Mitem/s │         │
│  │  ├─ 2                     2.153 ms      │ 7.548 ms      │ 4.058 ms      │ 3.824 ms      │ 100     │ 100
│  │  │                        9.285 Mitem/s │ 2.649 Mitem/s │ 4.927 Mitem/s │ 5.229 Mitem/s │         │
│  │  ├─ 3                     2.075 ms      │ 5.869 ms      │ 2.826 ms      │ 2.972 ms      │ 100     │ 100
│  │  │                        14.45 Mitem/s │ 5.111 Mitem/s │ 10.61 Mitem/s │ 10.09 Mitem/s │         │
│  │  ├─ 4                     2.252 ms      │ 7.323 ms      │ 3.972 ms      │ 3.908 ms      │ 100     │ 100
│  │  │                        17.75 Mitem/s │ 5.462 Mitem/s │ 10.06 Mitem/s │ 10.23 Mitem/s │         │
│  │  ├─ 5                     2.646 ms      │ 5.951 ms      │ 3.336 ms      │ 3.56 ms       │ 100     │ 100
│  │  │                        18.89 Mitem/s │ 8.4 Mitem/s   │ 14.98 Mitem/s │ 14.04 Mitem/s │         │
│  │  ╰─ 6                     3.072 ms      │ 5.827 ms      │ 4.266 ms      │ 4.078 ms      │ 100     │ 100
│  │                           19.52 Mitem/s │ 10.29 Mitem/s │ 14.06 Mitem/s │ 14.71 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.46 ms       │ 12.57 ms      │ 10.23 ms      │ 10.47 ms      │ 100     │ 100
│     │                        1.057 Mitem/s │ 795.3 Kitem/s │ 977.4 Kitem/s │ 954.3 Kitem/s │         │
│     ├─ 2                     9.675 ms      │ 19.41 ms      │ 12.43 ms      │ 12.53 ms      │ 100     │ 100
│     │                        2.067 Mitem/s │ 1.03 Mitem/s  │ 1.608 Mitem/s │ 1.595 Mitem/s │         │
│     ├─ 3                     10.01 ms      │ 23.11 ms      │ 14.72 ms      │ 14.92 ms      │ 100     │ 100
│     │                        2.996 Mitem/s │ 1.297 Mitem/s │ 2.037 Mitem/s │ 2.01 Mitem/s  │         │
│     ├─ 4                     9.981 ms      │ 29.52 ms      │ 15.51 ms      │ 16 ms         │ 100     │ 100
│     │                        4.007 Mitem/s │ 1.354 Mitem/s │ 2.577 Mitem/s │ 2.499 Mitem/s │         │
│     ├─ 5                     10.19 ms      │ 28.67 ms      │ 16.08 ms      │ 16.42 ms      │ 100     │ 100
│     │                        4.902 Mitem/s │ 1.743 Mitem/s │ 3.109 Mitem/s │ 3.043 Mitem/s │         │
│     ╰─ 6                     10.31 ms      │ 28.92 ms      │ 16.74 ms      │ 18.08 ms      │ 100     │ 100
│                              5.814 Mitem/s │ 2.073 Mitem/s │ 3.583 Mitem/s │ 3.318 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.534 ms      │ 6.984 ms      │ 4.698 ms      │ 4.862 ms      │ 100     │ 100
│  │  │                        2.205 Mitem/s │ 1.431 Mitem/s │ 2.128 Mitem/s │ 2.056 Mitem/s │         │
│  │  ├─ 2                     11.24 ms      │ 21.5 ms       │ 15.29 ms      │ 15.23 ms      │ 100     │ 100
│  │  │                        1.778 Mitem/s │ 929.8 Kitem/s │ 1.307 Mitem/s │ 1.312 Mitem/s │         │
│  │  ├─ 3                     22.9 ms       │ 39.91 ms      │ 29.84 ms      │ 29.96 ms      │ 100     │ 100
│  │  │                        1.309 Mitem/s │ 751.6 Kitem/s │ 1.005 Mitem/s │ 1.001 Mitem/s │         │
│  │  ├─ 4                     33.93 ms      │ 63.09 ms      │ 44.39 ms      │ 44.92 ms      │ 100     │ 100
│  │  │                        1.178 Mitem/s │ 633.9 Kitem/s │ 900.9 Kitem/s │ 890.3 Kitem/s │         │
│  │  ├─ 5                     54.67 ms      │ 78.35 ms      │ 62.23 ms      │ 63.6 ms       │ 100     │ 100
│  │  │                        914.4 Kitem/s │ 638.1 Kitem/s │ 803.3 Kitem/s │ 786 Kitem/s   │         │
│  │  ╰─ 6                     85.21 ms      │ 153.9 ms      │ 98.9 ms       │ 98.99 ms      │ 100     │ 100
│  │                           704 Kitem/s   │ 389.7 Kitem/s │ 606.6 Kitem/s │ 606.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.318 ms      │ 13.04 ms      │ 7.386 ms      │ 7.64 ms       │ 100     │ 100
│  │  │                        1.366 Mitem/s │ 766.2 Kitem/s │ 1.353 Mitem/s │ 1.308 Mitem/s │         │
│  │  ├─ 2                     7.636 ms      │ 16.06 ms      │ 11.24 ms      │ 11.31 ms      │ 100     │ 100
│  │  │                        2.619 Mitem/s │ 1.245 Mitem/s │ 1.779 Mitem/s │ 1.767 Mitem/s │         │
│  │  ├─ 3                     7.752 ms      │ 16.42 ms      │ 13.89 ms      │ 13.21 ms      │ 100     │ 100
│  │  │                        3.869 Mitem/s │ 1.825 Mitem/s │ 2.159 Mitem/s │ 2.269 Mitem/s │         │
│  │  ├─ 4                     8.514 ms      │ 24.68 ms      │ 13.72 ms      │ 13.77 ms      │ 100     │ 100
│  │  │                        4.697 Mitem/s │ 1.62 Mitem/s  │ 2.915 Mitem/s │ 2.903 Mitem/s │         │
│  │  ├─ 5                     11.61 ms      │ 23.87 ms      │ 13.94 ms      │ 14.51 ms      │ 100     │ 100
│  │  │                        4.303 Mitem/s │ 2.094 Mitem/s │ 3.585 Mitem/s │ 3.444 Mitem/s │         │
│  │  ╰─ 6                     8.616 ms      │ 23.2 ms       │ 13.55 ms      │ 13.46 ms      │ 100     │ 100
│  │                           6.963 Mitem/s │ 2.586 Mitem/s │ 4.425 Mitem/s │ 4.454 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.935 ms      │ 3.621 ms      │ 2.098 ms      │ 2.332 ms      │ 100     │ 100
│  │  │                        5.166 Mitem/s │ 2.761 Mitem/s │ 4.764 Mitem/s │ 4.286 Mitem/s │         │
│  │  ├─ 2                     2.066 ms      │ 5.4 ms        │ 3.455 ms      │ 3.443 ms      │ 100     │ 100
│  │  │                        9.679 Mitem/s │ 3.703 Mitem/s │ 5.787 Mitem/s │ 5.807 Mitem/s │         │
│  │  ├─ 3                     2.232 ms      │ 5.241 ms      │ 3.422 ms      │ 3.517 ms      │ 100     │ 100
│  │  │                        13.43 Mitem/s │ 5.723 Mitem/s │ 8.764 Mitem/s │ 8.528 Mitem/s │         │
│  │  ├─ 4                     2.576 ms      │ 6.677 ms      │ 3.937 ms      │ 3.804 ms      │ 100     │ 100
│  │  │                        15.52 Mitem/s │ 5.99 Mitem/s  │ 10.15 Mitem/s │ 10.51 Mitem/s │         │
│  │  ├─ 5                     2.694 ms      │ 6.19 ms       │ 3.505 ms      │ 3.625 ms      │ 100     │ 100
│  │  │                        18.55 Mitem/s │ 8.076 Mitem/s │ 14.26 Mitem/s │ 13.79 Mitem/s │         │
│  │  ╰─ 6                     3.086 ms      │ 6.387 ms      │ 4.03 ms       │ 3.912 ms      │ 100     │ 100
│  │                           19.44 Mitem/s │ 9.393 Mitem/s │ 14.88 Mitem/s │ 15.33 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.696 ms      │ 15.21 ms      │ 10.22 ms      │ 10.73 ms      │ 100     │ 100
│     │                        1.031 Mitem/s │ 657.1 Kitem/s │ 978.4 Kitem/s │ 931.8 Kitem/s │         │
│     ├─ 2                     9.956 ms      │ 20.97 ms      │ 13.89 ms      │ 13.42 ms      │ 100     │ 100
│     │                        2.008 Mitem/s │ 953.4 Kitem/s │ 1.439 Mitem/s │ 1.489 Mitem/s │         │
│     ├─ 3                     10.11 ms      │ 20.35 ms      │ 15.42 ms      │ 15.39 ms      │ 100     │ 100
│     │                        2.965 Mitem/s │ 1.473 Mitem/s │ 1.944 Mitem/s │ 1.948 Mitem/s │         │
│     ├─ 4                     10.48 ms      │ 28.41 ms      │ 15.84 ms      │ 16.35 ms      │ 100     │ 100
│     │                        3.814 Mitem/s │ 1.407 Mitem/s │ 2.523 Mitem/s │ 2.445 Mitem/s │         │
│     ├─ 5                     10.66 ms      │ 33.42 ms      │ 17.23 ms      │ 17.98 ms      │ 100     │ 100
│     │                        4.689 Mitem/s │ 1.495 Mitem/s │ 2.9 Mitem/s   │ 2.78 Mitem/s  │         │
│     ╰─ 6                     10.69 ms      │ 34.56 ms      │ 18.05 ms      │ 19.45 ms      │ 100     │ 100
│                              5.61 Mitem/s  │ 1.735 Mitem/s │ 3.323 Mitem/s │ 3.084 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.538 ms      │ 8.394 ms      │ 4.622 ms      │ 4.814 ms      │ 100     │ 100
│  │  │                        2.203 Mitem/s │ 1.191 Mitem/s │ 2.163 Mitem/s │ 2.077 Mitem/s │         │
│  │  ├─ 2                     11.5 ms       │ 20.82 ms      │ 15.46 ms      │ 15.55 ms      │ 100     │ 100
│  │  │                        1.738 Mitem/s │ 960.5 Kitem/s │ 1.293 Mitem/s │ 1.286 Mitem/s │         │
│  │  ├─ 3                     22.35 ms      │ 39.67 ms      │ 28.56 ms      │ 28.89 ms      │ 100     │ 100
│  │  │                        1.342 Mitem/s │ 756.1 Kitem/s │ 1.05 Mitem/s  │ 1.038 Mitem/s │         │
│  │  ├─ 4                     35.53 ms      │ 52.77 ms      │ 41.36 ms      │ 42.32 ms      │ 100     │ 100
│  │  │                        1.125 Mitem/s │ 757.9 Kitem/s │ 966.9 Kitem/s │ 945.1 Kitem/s │         │
│  │  ├─ 5                     52.23 ms      │ 72.49 ms      │ 61.01 ms      │ 60.85 ms      │ 100     │ 100
│  │  │                        957.2 Kitem/s │ 689.7 Kitem/s │ 819.4 Kitem/s │ 821.6 Kitem/s │         │
│  │  ╰─ 6                     86.16 ms      │ 115.4 ms      │ 97.96 ms      │ 97.87 ms      │ 100     │ 100
│  │                           696.3 Kitem/s │ 519.8 Kitem/s │ 612.4 Kitem/s │ 613 Kitem/s   │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.35 ms       │ 13.62 ms      │ 7.53 ms       │ 7.921 ms      │ 100     │ 100
│  │  │                        1.36 Mitem/s  │ 733.8 Kitem/s │ 1.327 Mitem/s │ 1.262 Mitem/s │         │
│  │  ├─ 2                     7.725 ms      │ 16.18 ms      │ 11.58 ms      │ 11.7 ms       │ 100     │ 100
│  │  │                        2.588 Mitem/s │ 1.235 Mitem/s │ 1.726 Mitem/s │ 1.709 Mitem/s │         │
│  │  ├─ 3                     7.831 ms      │ 17.41 ms      │ 11.82 ms      │ 12.6 ms       │ 100     │ 100
│  │  │                        3.83 Mitem/s  │ 1.722 Mitem/s │ 2.537 Mitem/s │ 2.379 Mitem/s │         │
│  │  ├─ 4                     8.564 ms      │ 19.31 ms      │ 11.9 ms       │ 11.98 ms      │ 100     │ 100
│  │  │                        4.67 Mitem/s  │ 2.071 Mitem/s │ 3.361 Mitem/s │ 3.337 Mitem/s │         │
│  │  ├─ 5                     8.497 ms      │ 25.86 ms      │ 13.35 ms      │ 13.44 ms      │ 100     │ 100
│  │  │                        5.884 Mitem/s │ 1.933 Mitem/s │ 3.742 Mitem/s │ 3.719 Mitem/s │         │
│  │  ╰─ 6                     8.517 ms      │ 23 ms         │ 13.66 ms      │ 14.16 ms      │ 100     │ 100
│  │                           7.044 Mitem/s │ 2.608 Mitem/s │ 4.391 Mitem/s │ 4.235 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.91 ms       │ 3.777 ms      │ 1.967 ms      │ 2.094 ms      │ 100     │ 100
│  │  │                        5.235 Mitem/s │ 2.647 Mitem/s │ 5.082 Mitem/s │ 4.774 Mitem/s │         │
│  │  ├─ 2                     2.035 ms      │ 5.687 ms      │ 3.999 ms      │ 3.752 ms      │ 100     │ 100
│  │  │                        9.827 Mitem/s │ 3.516 Mitem/s │ 5 Mitem/s     │ 5.33 Mitem/s  │         │
│  │  ├─ 3                     2.112 ms      │ 4.155 ms      │ 2.399 ms      │ 2.675 ms      │ 100     │ 100
│  │  │                        14.2 Mitem/s  │ 7.219 Mitem/s │ 12.5 Mitem/s  │ 11.21 Mitem/s │         │
│  │  ├─ 4                     2.262 ms      │ 7.831 ms      │ 4.229 ms      │ 4.091 ms      │ 100     │ 100
│  │  │                        17.68 Mitem/s │ 5.107 Mitem/s │ 9.457 Mitem/s │ 9.776 Mitem/s │         │
│  │  ├─ 5                     2.608 ms      │ 7.022 ms      │ 4.213 ms      │ 3.95 ms       │ 100     │ 100
│  │  │                        19.16 Mitem/s │ 7.119 Mitem/s │ 11.86 Mitem/s │ 12.65 Mitem/s │         │
│  │  ╰─ 6                     3.078 ms      │ 10.5 ms       │ 4.461 ms      │ 5.178 ms      │ 100     │ 100
│  │                           19.48 Mitem/s │ 5.711 Mitem/s │ 13.44 Mitem/s │ 11.58 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.413 ms      │ 11.23 ms      │ 8.583 ms      │ 8.728 ms      │ 100     │ 100
│     │                        1.188 Mitem/s │ 890.3 Kitem/s │ 1.165 Mitem/s │ 1.145 Mitem/s │         │
│     ├─ 2                     8.74 ms       │ 19.26 ms      │ 12.11 ms      │ 12.63 ms      │ 100     │ 100
│     │                        2.288 Mitem/s │ 1.038 Mitem/s │ 1.651 Mitem/s │ 1.582 Mitem/s │         │
│     ├─ 3                     8.77 ms       │ 23.48 ms      │ 13.23 ms      │ 13.86 ms      │ 100     │ 100
│     │                        3.42 Mitem/s  │ 1.277 Mitem/s │ 2.266 Mitem/s │ 2.163 Mitem/s │         │
│     ├─ 4                     8.961 ms      │ 32.42 ms      │ 15.31 ms      │ 15.01 ms      │ 100     │ 100
│     │                        4.463 Mitem/s │ 1.233 Mitem/s │ 2.611 Mitem/s │ 2.664 Mitem/s │         │
│     ├─ 5                     9.006 ms      │ 30.35 ms      │ 16.43 ms      │ 16.15 ms      │ 100     │ 100
│     │                        5.551 Mitem/s │ 1.647 Mitem/s │ 3.042 Mitem/s │ 3.094 Mitem/s │         │
│     ╰─ 6                     13.72 ms      │ 28.51 ms      │ 17.12 ms      │ 19.27 ms      │ 100     │ 100
│                              4.37 Mitem/s  │ 2.104 Mitem/s │ 3.503 Mitem/s │ 3.112 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.596 ms      │ 7.538 ms      │ 4.728 ms      │ 5.108 ms      │ 100     │ 100
│  │  │                        2.175 Mitem/s │ 1.326 Mitem/s │ 2.114 Mitem/s │ 1.957 Mitem/s │         │
│  │  ├─ 2                     11.18 ms      │ 21 ms         │ 14.57 ms      │ 14.92 ms      │ 100     │ 100
│  │  │                        1.788 Mitem/s │ 952.1 Kitem/s │ 1.372 Mitem/s │ 1.339 Mitem/s │         │
│  │  ├─ 3                     20.75 ms      │ 38.6 ms       │ 28.11 ms      │ 28.02 ms      │ 100     │ 100
│  │  │                        1.445 Mitem/s │ 777 Kitem/s   │ 1.067 Mitem/s │ 1.07 Mitem/s  │         │
│  │  ├─ 4                     35.07 ms      │ 55.97 ms      │ 43.42 ms      │ 44.09 ms      │ 100     │ 100
│  │  │                        1.14 Mitem/s  │ 714.5 Kitem/s │ 921.1 Kitem/s │ 907 Kitem/s   │         │
│  │  ├─ 5                     55.12 ms      │ 90.21 ms      │ 63.44 ms      │ 65 ms         │ 100     │ 100
│  │  │                        906.9 Kitem/s │ 554.2 Kitem/s │ 788 Kitem/s   │ 769.2 Kitem/s │         │
│  │  ╰─ 6                     85.91 ms      │ 138.1 ms      │ 100.7 ms      │ 101.5 ms      │ 100     │ 100
│  │                           698.3 Kitem/s │ 434.3 Kitem/s │ 595.8 Kitem/s │ 590.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.366 ms      │ 14.21 ms      │ 7.542 ms      │ 8.403 ms      │ 100     │ 100
│  │  │                        1.357 Mitem/s │ 703.3 Kitem/s │ 1.325 Mitem/s │ 1.189 Mitem/s │         │
│  │  ├─ 2                     7.754 ms      │ 15.85 ms      │ 13.62 ms      │ 12.5 ms       │ 100     │ 100
│  │  │                        2.579 Mitem/s │ 1.261 Mitem/s │ 1.467 Mitem/s │ 1.599 Mitem/s │         │
│  │  ├─ 3                     8.405 ms      │ 17.44 ms      │ 14.25 ms      │ 14.16 ms      │ 100     │ 100
│  │  │                        3.569 Mitem/s │ 1.719 Mitem/s │ 2.103 Mitem/s │ 2.118 Mitem/s │         │
│  │  ├─ 4                     8.434 ms      │ 19.12 ms      │ 12.59 ms      │ 12.53 ms      │ 100     │ 100
│  │  │                        4.742 Mitem/s │ 2.091 Mitem/s │ 3.176 Mitem/s │ 3.192 Mitem/s │         │
│  │  ├─ 5                     9.271 ms      │ 24.06 ms      │ 14 ms         │ 14.37 ms      │ 100     │ 100
│  │  │                        5.392 Mitem/s │ 2.077 Mitem/s │ 3.571 Mitem/s │ 3.478 Mitem/s │         │
│  │  ╰─ 6                     8.73 ms       │ 22.51 ms      │ 13.82 ms      │ 14.42 ms      │ 100     │ 100
│  │                           6.872 Mitem/s │ 2.665 Mitem/s │ 4.34 Mitem/s  │ 4.159 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.937 ms      │ 3.664 ms      │ 2.15 ms       │ 2.467 ms      │ 100     │ 100
│  │  │                        5.161 Mitem/s │ 2.728 Mitem/s │ 4.649 Mitem/s │ 4.053 Mitem/s │         │
│  │  ├─ 2                     2.314 ms      │ 5.033 ms      │ 4.051 ms      │ 3.955 ms      │ 100     │ 100
│  │  │                        8.641 Mitem/s │ 3.972 Mitem/s │ 4.936 Mitem/s │ 5.056 Mitem/s │         │
│  │  ├─ 3                     2.381 ms      │ 5.498 ms      │ 4.353 ms      │ 4.206 ms      │ 100     │ 100
│  │  │                        12.59 Mitem/s │ 5.456 Mitem/s │ 6.891 Mitem/s │ 7.131 Mitem/s │         │
│  │  ├─ 4                     2.879 ms      │ 7.317 ms      │ 4.084 ms      │ 3.891 ms      │ 100     │ 100
│  │  │                        13.88 Mitem/s │ 5.466 Mitem/s │ 9.792 Mitem/s │ 10.27 Mitem/s │         │
│  │  ├─ 5                     2.957 ms      │ 7.426 ms      │ 4.292 ms      │ 4.339 ms      │ 100     │ 100
│  │  │                        16.9 Mitem/s  │ 6.732 Mitem/s │ 11.64 Mitem/s │ 11.52 Mitem/s │         │
│  │  ╰─ 6                     3.057 ms      │ 7.575 ms      │ 4.296 ms      │ 4.145 ms      │ 100     │ 100
│  │                           19.62 Mitem/s │ 7.919 Mitem/s │ 13.96 Mitem/s │ 14.47 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.742 ms      │ 13.1 ms       │ 10.32 ms      │ 10.61 ms      │ 100     │ 100
│     │                        1.026 Mitem/s │ 763.2 Kitem/s │ 968.1 Kitem/s │ 942 Kitem/s   │         │
│     ├─ 2                     10.09 ms      │ 19.93 ms      │ 12.57 ms      │ 13.13 ms      │ 100     │ 100
│     │                        1.981 Mitem/s │ 1.003 Mitem/s │ 1.59 Mitem/s  │ 1.522 Mitem/s │         │
│     ├─ 3                     10.23 ms      │ 20.99 ms      │ 15.32 ms      │ 15.79 ms      │ 100     │ 100
│     │                        2.929 Mitem/s │ 1.428 Mitem/s │ 1.958 Mitem/s │ 1.899 Mitem/s │         │
│     ├─ 4                     10.34 ms      │ 29.86 ms      │ 16.32 ms      │ 16.45 ms      │ 100     │ 100
│     │                        3.867 Mitem/s │ 1.339 Mitem/s │ 2.45 Mitem/s  │ 2.431 Mitem/s │         │
│     ├─ 5                     10.69 ms      │ 31.97 ms      │ 17.42 ms      │ 17.23 ms      │ 100     │ 100
│     │                        4.674 Mitem/s │ 1.563 Mitem/s │ 2.868 Mitem/s │ 2.901 Mitem/s │         │
│     ╰─ 6                     10.69 ms      │ 31.25 ms      │ 18.27 ms      │ 19.99 ms      │ 100     │ 100
│                              5.608 Mitem/s │ 1.919 Mitem/s │ 3.282 Mitem/s │ 3.001 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     5.17 ms       │ 9.561 ms      │ 5.29 ms       │ 5.895 ms      │ 100     │ 100
│  │  │                        1.934 Mitem/s │ 1.045 Mitem/s │ 1.89 Mitem/s  │ 1.696 Mitem/s │         │
│  │  ├─ 2                     12.44 ms      │ 24.01 ms      │ 18.29 ms      │ 18.67 ms      │ 100     │ 100
│  │  │                        1.607 Mitem/s │ 832.6 Kitem/s │ 1.092 Mitem/s │ 1.071 Mitem/s │         │
│  │  ├─ 3                     24.09 ms      │ 41.22 ms      │ 33.45 ms      │ 33.15 ms      │ 100     │ 100
│  │  │                        1.245 Mitem/s │ 727.7 Kitem/s │ 896.6 Kitem/s │ 904.7 Kitem/s │         │
│  │  ├─ 4                     38.99 ms      │ 63.73 ms      │ 44.76 ms      │ 46.19 ms      │ 100     │ 100
│  │  │                        1.025 Mitem/s │ 627.5 Kitem/s │ 893.5 Kitem/s │ 865.9 Kitem/s │         │
│  │  ├─ 5                     58.69 ms      │ 77.44 ms      │ 64.74 ms      │ 65.29 ms      │ 100     │ 100
│  │  │                        851.8 Kitem/s │ 645.6 Kitem/s │ 772.3 Kitem/s │ 765.7 Kitem/s │         │
│  │  ╰─ 6                     87.8 ms       │ 140.6 ms      │ 104.2 ms      │ 103.6 ms      │ 100     │ 100
│  │                           683.3 Kitem/s │ 426.6 Kitem/s │ 575.7 Kitem/s │ 578.7 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.413 ms      │ 10.26 ms      │ 7.551 ms      │ 7.853 ms      │ 100     │ 100
│  │  │                        1.348 Mitem/s │ 973.9 Kitem/s │ 1.324 Mitem/s │ 1.273 Mitem/s │         │
│  │  ├─ 2                     7.684 ms      │ 15.34 ms      │ 9.471 ms      │ 9.842 ms      │ 100     │ 100
│  │  │                        2.602 Mitem/s │ 1.303 Mitem/s │ 2.111 Mitem/s │ 2.032 Mitem/s │         │
│  │  ├─ 3                     7.75 ms       │ 23.66 ms      │ 14.42 ms      │ 13.94 ms      │ 100     │ 100
│  │  │                        3.87 Mitem/s  │ 1.267 Mitem/s │ 2.08 Mitem/s  │ 2.15 Mitem/s  │         │
│  │  ├─ 4                     8.51 ms       │ 25 ms         │ 13.73 ms      │ 13.71 ms      │ 100     │ 100
│  │  │                        4.7 Mitem/s   │ 1.599 Mitem/s │ 2.912 Mitem/s │ 2.917 Mitem/s │         │
│  │  ├─ 5                     8.833 ms      │ 24.3 ms       │ 13.67 ms      │ 14.7 ms       │ 100     │ 100
│  │  │                        5.66 Mitem/s  │ 2.057 Mitem/s │ 3.657 Mitem/s │ 3.399 Mitem/s │         │
│  │  ╰─ 6                     9.392 ms      │ 24.59 ms      │ 13.97 ms      │ 14.97 ms      │ 100     │ 100
│  │                           6.388 Mitem/s │ 2.439 Mitem/s │ 4.292 Mitem/s │ 4.007 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.939 ms      │ 3.603 ms      │ 1.997 ms      │ 2.121 ms      │ 100     │ 100
│  │  │                        5.156 Mitem/s │ 2.775 Mitem/s │ 5.006 Mitem/s │ 4.714 Mitem/s │         │
│  │  ├─ 2                     2.09 ms       │ 4.577 ms      │ 2.918 ms      │ 3.204 ms      │ 100     │ 100
│  │  │                        9.565 Mitem/s │ 4.369 Mitem/s │ 6.853 Mitem/s │ 6.24 Mitem/s  │         │
│  │  ├─ 3                     2.222 ms      │ 5.718 ms      │ 3.704 ms      │ 3.648 ms      │ 100     │ 100
│  │  │                        13.49 Mitem/s │ 5.245 Mitem/s │ 8.098 Mitem/s │ 8.221 Mitem/s │         │
│  │  ├─ 4                     2.268 ms      │ 4.978 ms      │ 4.013 ms      │ 3.732 ms      │ 100     │ 100
│  │  │                        17.63 Mitem/s │ 8.035 Mitem/s │ 9.967 Mitem/s │ 10.71 Mitem/s │         │
│  │  ├─ 5                     2.662 ms      │ 5.828 ms      │ 4.054 ms      │ 3.9 ms        │ 100     │ 100
│  │  │                        18.77 Mitem/s │ 8.578 Mitem/s │ 12.33 Mitem/s │ 12.81 Mitem/s │         │
│  │  ╰─ 6                     3.032 ms      │ 7.942 ms      │ 4.309 ms      │ 4.209 ms      │ 100     │ 100
│  │                           19.78 Mitem/s │ 7.554 Mitem/s │ 13.92 Mitem/s │ 14.25 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.25 ms       │ 14.11 ms      │ 9.703 ms      │ 10.18 ms      │ 100     │ 100
│     │                        1.08 Mitem/s  │ 708.2 Kitem/s │ 1.03 Mitem/s  │ 981.7 Kitem/s │         │
│     ├─ 2                     9.384 ms      │ 20.97 ms      │ 10.63 ms      │ 12 ms         │ 100     │ 100
│     │                        2.131 Mitem/s │ 953.4 Kitem/s │ 1.88 Mitem/s  │ 1.665 Mitem/s │         │
│     ├─ 3                     9.697 ms      │ 21.66 ms      │ 14.74 ms      │ 14.84 ms      │ 100     │ 100
│     │                        3.093 Mitem/s │ 1.384 Mitem/s │ 2.033 Mitem/s │ 2.02 Mitem/s  │         │
│     ├─ 4                     9.881 ms      │ 29.65 ms      │ 16.84 ms      │ 16.72 ms      │ 100     │ 100
│     │                        4.047 Mitem/s │ 1.348 Mitem/s │ 2.374 Mitem/s │ 2.391 Mitem/s │         │
│     ├─ 5                     9.745 ms      │ 29.09 ms      │ 17.43 ms      │ 17.65 ms      │ 100     │ 100
│     │                        5.13 Mitem/s  │ 1.718 Mitem/s │ 2.867 Mitem/s │ 2.831 Mitem/s │         │
│     ╰─ 6                     9.98 ms       │ 29.94 ms      │ 17.95 ms      │ 20.56 ms      │ 100     │ 100
│                              6.011 Mitem/s │ 2.003 Mitem/s │ 3.342 Mitem/s │ 2.917 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 3                     8.152 ms      │ 22.86 ms      │ 10.88 ms      │ 11.52 ms      │ 100     │ 100
│  │  ├─ 4                     9.552 ms      │ 20.6 ms       │ 14.83 ms      │ 14.24 ms      │ 100     │ 100
│  │  ├─ 5                     10.27 ms      │ 25.92 ms      │ 15.19 ms      │ 15.28 ms      │ 100     │ 100
│  │  ╰─ 6                     11.24 ms      │ 26.01 ms      │ 14.91 ms      │ 15.59 ms      │ 100     │ 100
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 3                     19.82 ms      │ 36.46 ms      │ 26.73 ms      │ 26.95 ms      │ 100     │ 100
│  │  ├─ 4                     23.68 ms      │ 37.06 ms      │ 29.83 ms      │ 29.95 ms      │ 100     │ 100
│  │  ├─ 5                     24.91 ms      │ 39.82 ms      │ 32.41 ms      │ 32.22 ms      │ 100     │ 100
│  │  ╰─ 6                     26.84 ms      │ 49.73 ms      │ 33.77 ms      │ 34.07 ms      │ 100     │ 100
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 3                     12.39 ms      │ 24.5 ms       │ 14.53 ms      │ 16.28 ms      │ 100     │ 100
│     ├─ 4                     13.15 ms      │ 27.4 ms       │ 18.37 ms      │ 17.68 ms      │ 100     │ 100
│     ├─ 5                     13.64 ms      │ 30.84 ms      │ 19.31 ms      │ 19.99 ms      │ 100     │ 100
│     ╰─ 6                     13.57 ms      │ 34.01 ms      │ 19.2 ms       │ 20.17 ms      │ 100     │ 100
├─ 14_prefix_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     23.25 ms      │ 26.74 ms      │ 23.54 ms      │ 24.22 ms      │ 20      │ 20
│     │                        4.3 Kitem/s   │ 3.738 Kitem/s │ 4.247 Kitem/s │ 4.127 Kitem/s │         │
│     ├─ 2                     23.75 ms      │ 43.95 ms      │ 39.73 ms      │ 36.55 ms      │ 20      │ 20
│     │                        8.42 Kitem/s  │ 4.549 Kitem/s │ 5.033 Kitem/s │ 5.471 Kitem/s │         │
│     ├─ 3                     25.96 ms      │ 44.61 ms      │ 39.92 ms      │ 38.47 ms      │ 20      │ 20
│     │                        11.55 Kitem/s │ 6.723 Kitem/s │ 7.514 Kitem/s │ 7.797 Kitem/s │         │
│     ├─ 4                     34.45 ms      │ 50.39 ms      │ 35.26 ms      │ 37.45 ms      │ 20      │ 20
│     │                        11.6 Kitem/s  │ 7.936 Kitem/s │ 11.34 Kitem/s │ 10.68 Kitem/s │         │
│     ├─ 5                     36.16 ms      │ 56.67 ms      │ 43.95 ms      │ 44.83 ms      │ 20      │ 20
│     │                        13.82 Kitem/s │ 8.821 Kitem/s │ 11.37 Kitem/s │ 11.15 Kitem/s │         │
│     ╰─ 6                     36.98 ms      │ 56.45 ms      │ 43.55 ms      │ 44.54 ms      │ 20      │ 20
│                              16.22 Kitem/s │ 10.62 Kitem/s │ 13.77 Kitem/s │ 13.46 Kitem/s │         │
├─ 15_full_scan_aggregate                    │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     14.74 ms      │ 18 ms         │ 15.15 ms      │ 15.28 ms      │ 100     │ 100
│  │  │                        6.783 Kitem/s │ 5.552 Kitem/s │ 6.599 Kitem/s │ 6.542 Kitem/s │         │
│  │  ├─ 2                     37.38 ms      │ 59.78 ms      │ 52.75 ms      │ 51.23 ms      │ 100     │ 100
│  │  │                        5.349 Kitem/s │ 3.345 Kitem/s │ 3.791 Kitem/s │ 3.903 Kitem/s │         │
│  │  ├─ 3                     48.41 ms      │ 89.69 ms      │ 81.31 ms      │ 78.33 ms      │ 100     │ 100
│  │  │                        6.196 Kitem/s │ 3.344 Kitem/s │ 3.689 Kitem/s │ 3.829 Kitem/s │         │
│  │  ├─ 4                     69.06 ms      │ 118.8 ms      │ 105 ms        │ 100.9 ms      │ 100     │ 100
│  │  │                        5.791 Kitem/s │ 3.366 Kitem/s │ 3.807 Kitem/s │ 3.961 Kitem/s │         │
│  │  ├─ 5                     86.43 ms      │ 146.5 ms      │ 131.8 ms      │ 130.2 ms      │ 100     │ 100
│  │  │                        5.784 Kitem/s │ 3.412 Kitem/s │ 3.791 Kitem/s │ 3.837 Kitem/s │         │
│  │  ╰─ 6                     103.1 ms      │ 175.9 ms      │ 163.7 ms      │ 157.8 ms      │ 100     │ 100
│  │                           5.818 Kitem/s │ 3.409 Kitem/s │ 3.663 Kitem/s │ 3.801 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     33.48 ms      │ 36.5 ms       │ 34.06 ms      │ 34.18 ms      │ 100     │ 100
│  │  │                        2.986 Kitem/s │ 2.739 Kitem/s │ 2.935 Kitem/s │ 2.925 Kitem/s │         │
│  │  ├─ 2                     33.46 ms      │ 58.78 ms      │ 35.57 ms      │ 36.39 ms      │ 100     │ 100
│  │  │                        5.976 Kitem/s │ 3.402 Kitem/s │ 5.621 Kitem/s │ 5.495 Kitem/s │         │
│  │  ├─ 3                     34.4 ms       │ 60.35 ms      │ 42.16 ms      │ 45 ms         │ 100     │ 100
│  │  │                        8.719 Kitem/s │ 4.97 Kitem/s  │ 7.114 Kitem/s │ 6.666 Kitem/s │         │
│  │  ├─ 4                     34.48 ms      │ 72.64 ms      │ 49.91 ms      │ 49.65 ms      │ 100     │ 100
│  │  │                        11.59 Kitem/s │ 5.506 Kitem/s │ 8.014 Kitem/s │ 8.055 Kitem/s │         │
│  │  ├─ 5                     35.78 ms      │ 95.87 ms      │ 58.08 ms      │ 56.42 ms      │ 100     │ 100
│  │  │                        13.97 Kitem/s │ 5.215 Kitem/s │ 8.607 Kitem/s │ 8.86 Kitem/s  │         │
│  │  ╰─ 6                     35.3 ms       │ 85.41 ms      │ 59.99 ms      │ 58.37 ms      │ 100     │ 100
│  │                           16.99 Kitem/s │ 7.024 Kitem/s │ 10 Kitem/s    │ 10.27 Kitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     6.979 ms      │ 10.75 ms      │ 7.543 ms      │ 7.9 ms        │ 100     │ 100
│  │  │                        14.32 Kitem/s │ 9.296 Kitem/s │ 13.25 Kitem/s │ 12.65 Kitem/s │         │
│  │  ├─ 2                     7.103 ms      │ 14.77 ms      │ 8.343 ms      │ 9.496 ms      │ 100     │ 100
│  │  │                        28.15 Kitem/s │ 13.53 Kitem/s │ 23.96 Kitem/s │ 21.06 Kitem/s │         │
│  │  ├─ 3                     7.61 ms       │ 15.58 ms      │ 13.51 ms      │ 12.68 ms      │ 100     │ 100
│  │  │                        39.41 Kitem/s │ 19.24 Kitem/s │ 22.19 Kitem/s │ 23.64 Kitem/s │         │
│  │  ├─ 4                     7.686 ms      │ 17.71 ms      │ 13.16 ms      │ 12.77 ms      │ 100     │ 100
│  │  │                        52.03 Kitem/s │ 22.57 Kitem/s │ 30.37 Kitem/s │ 31.29 Kitem/s │         │
│  │  ├─ 5                     8.011 ms      │ 21.38 ms      │ 12.48 ms      │ 12.85 ms      │ 100     │ 100
│  │  │                        62.4 Kitem/s  │ 23.37 Kitem/s │ 40.03 Kitem/s │ 38.89 Kitem/s │         │
│  │  ╰─ 6                     7.763 ms      │ 21.47 ms      │ 12.79 ms      │ 13.65 ms      │ 100     │ 100
│  │                           77.28 Kitem/s │ 27.93 Kitem/s │ 46.89 Kitem/s │ 43.93 Kitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     47.39 ms      │ 52.28 ms      │ 48.83 ms      │ 49.02 ms      │ 100     │ 100
│     │                        2.11 Kitem/s  │ 1.912 Kitem/s │ 2.047 Kitem/s │ 2.039 Kitem/s │         │
│     ├─ 2                     49 ms         │ 77.16 ms      │ 51.45 ms      │ 53.49 ms      │ 100     │ 100
│     │                        4.081 Kitem/s │ 2.591 Kitem/s │ 3.886 Kitem/s │ 3.738 Kitem/s │         │
│     ├─ 3                     50.33 ms      │ 89.44 ms      │ 64.37 ms      │ 62.89 ms      │ 100     │ 100
│     │                        5.96 Kitem/s  │ 3.354 Kitem/s │ 4.66 Kitem/s  │ 4.77 Kitem/s  │         │
│     ├─ 4                     50.22 ms      │ 96.1 ms       │ 64.49 ms      │ 64.29 ms      │ 100     │ 100
│     │                        7.964 Kitem/s │ 4.162 Kitem/s │ 6.201 Kitem/s │ 6.221 Kitem/s │         │
│     ├─ 5                     51 ms         │ 96 ms         │ 77.64 ms      │ 75.16 ms      │ 100     │ 100
│     │                        9.802 Kitem/s │ 5.207 Kitem/s │ 6.439 Kitem/s │ 6.651 Kitem/s │         │
│     ╰─ 6                     51.15 ms      │ 113.4 ms      │ 86.1 ms       │ 83.94 ms      │ 100     │ 100
│                              11.73 Kitem/s │ 5.289 Kitem/s │ 6.968 Kitem/s │ 7.147 Kitem/s │         │
├─ 16_insert_heavy                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.409 ms      │ 8.066 ms      │ 5.813 ms      │ 5.796 ms      │ 100     │ 100
│  │  │                        2.268 Mitem/s │ 1.239 Mitem/s │ 1.72 Mitem/s  │ 1.725 Mitem/s │         │
│  │  ├─ 2                     6.539 ms      │ 15.3 ms       │ 10.44 ms      │ 10 ms         │ 100     │ 100
│  │  │                        3.058 Mitem/s │ 1.306 Mitem/s │ 1.914 Mitem/s │ 1.998 Mitem/s │         │
│  │  ├─ 3                     8.367 ms      │ 17.5 ms       │ 12.6 ms       │ 12.23 ms      │ 100     │ 100
│  │  │                        3.585 Mitem/s │ 1.714 Mitem/s │ 2.38 Mitem/s  │ 2.451 Mitem/s │         │
│  │  ├─ 4                     9.921 ms      │ 25.54 ms      │ 14.11 ms      │ 14.63 ms      │ 100     │ 100
│  │  │                        4.031 Mitem/s │ 1.565 Mitem/s │ 2.834 Mitem/s │ 2.733 Mitem/s │         │
│  │  ├─ 5                     11.89 ms      │ 29.39 ms      │ 16.87 ms      │ 17.67 ms      │ 100     │ 100
│  │  │                        4.203 Mitem/s │ 1.7 Mitem/s   │ 2.962 Mitem/s │ 2.828 Mitem/s │         │
│  │  ╰─ 6                     12.79 ms      │ 28.92 ms      │ 18.49 ms      │ 19.08 ms      │ 100     │ 100
│  │                           4.687 Mitem/s │ 2.073 Mitem/s │ 3.243 Mitem/s │ 3.143 Mitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     1.234 ms      │ 2.667 ms      │ 2.081 ms      │ 2.042 ms      │ 100     │ 100
│  │  │                        8.102 Mitem/s │ 3.748 Mitem/s │ 4.805 Mitem/s │ 4.896 Mitem/s │         │
│  │  ├─ 2                     1.666 ms      │ 3.048 ms      │ 2.554 ms      │ 2.447 ms      │ 100     │ 100
│  │  │                        12 Mitem/s    │ 6.56 Mitem/s  │ 7.83 Mitem/s  │ 8.172 Mitem/s │         │
│  │  ├─ 3                     2.291 ms      │ 3.957 ms      │ 2.963 ms      │ 2.94 ms       │ 100     │ 100
│  │  │                        13.09 Mitem/s │ 7.579 Mitem/s │ 10.12 Mitem/s │ 10.2 Mitem/s  │         │
│  │  ├─ 4                     2.397 ms      │ 4.073 ms      │ 3.34 ms       │ 3.267 ms      │ 100     │ 100
│  │  │                        16.68 Mitem/s │ 9.82 Mitem/s  │ 11.97 Mitem/s │ 12.24 Mitem/s │         │
│  │  ├─ 5                     2.854 ms      │ 4.824 ms      │ 3.955 ms      │ 3.866 ms      │ 100     │ 100
│  │  │                        17.51 Mitem/s │ 10.36 Mitem/s │ 12.64 Mitem/s │ 12.93 Mitem/s │         │
│  │  ╰─ 6                     2.905 ms      │ 5.612 ms      │ 4.359 ms      │ 4.334 ms      │ 100     │ 100
│  │                           20.64 Mitem/s │ 10.68 Mitem/s │ 13.76 Mitem/s │ 13.84 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     1.24 ms       │ 2.333 ms      │ 1.653 ms      │ 1.717 ms      │ 100     │ 100
│  │  │                        8.061 Mitem/s │ 4.285 Mitem/s │ 6.046 Mitem/s │ 5.821 Mitem/s │         │
│  │  ├─ 2                     3.452 ms      │ 6.352 ms      │ 5.014 ms      │ 4.891 ms      │ 100     │ 100
│  │  │                        5.793 Mitem/s │ 3.148 Mitem/s │ 3.988 Mitem/s │ 4.088 Mitem/s │         │
│  │  ├─ 3                     6.227 ms      │ 10.3 ms       │ 8.017 ms      │ 8.029 ms      │ 100     │ 100
│  │  │                        4.817 Mitem/s │ 2.912 Mitem/s │ 3.741 Mitem/s │ 3.736 Mitem/s │         │
│  │  ├─ 4                     8.581 ms      │ 15.94 ms      │ 12.91 ms      │ 12.94 ms      │ 100     │ 100
│  │  │                        4.66 Mitem/s  │ 2.508 Mitem/s │ 3.098 Mitem/s │ 3.091 Mitem/s │         │
│  │  ├─ 5                     10.84 ms      │ 19.96 ms      │ 16.38 ms      │ 16.25 ms      │ 100     │ 100
│  │  │                        4.609 Mitem/s │ 2.504 Mitem/s │ 3.051 Mitem/s │ 3.076 Mitem/s │         │
│  │  ╰─ 6                     15.06 ms      │ 25.83 ms      │ 20.45 ms      │ 20.11 ms      │ 100     │ 100
│  │                           3.983 Mitem/s │ 2.322 Mitem/s │ 2.933 Mitem/s │ 2.982 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     2.206 ms      │ 4.761 ms      │ 3.233 ms      │ 3.066 ms      │ 100     │ 100
│     │                        4.531 Mitem/s │ 2.1 Mitem/s   │ 3.092 Mitem/s │ 3.26 Mitem/s  │         │
│     ├─ 2                     2.443 ms      │ 5.338 ms      │ 3.934 ms      │ 3.995 ms      │ 100     │ 100
│     │                        8.186 Mitem/s │ 3.746 Mitem/s │ 5.083 Mitem/s │ 5.005 Mitem/s │         │
│     ├─ 3                     3.479 ms      │ 5.328 ms      │ 4.602 ms      │ 4.517 ms      │ 100     │ 100
│     │                        8.622 Mitem/s │ 5.63 Mitem/s  │ 6.517 Mitem/s │ 6.641 Mitem/s │         │
│     ├─ 4                     2.924 ms      │ 7.706 ms      │ 5.079 ms      │ 5.138 ms      │ 100     │ 100
│     │                        13.67 Mitem/s │ 5.19 Mitem/s  │ 7.874 Mitem/s │ 7.784 Mitem/s │         │
│     ├─ 5                     3.933 ms      │ 7.611 ms      │ 6.279 ms      │ 6.085 ms      │ 100     │ 100
│     │                        12.71 Mitem/s │ 6.569 Mitem/s │ 7.962 Mitem/s │ 8.216 Mitem/s │         │
│     ╰─ 6                     3.907 ms      │ 8.048 ms      │ 6.079 ms      │ 6.02 ms       │ 100     │ 100
│                              15.35 Mitem/s │ 7.454 Mitem/s │ 9.868 Mitem/s │ 9.965 Mitem/s │         │
├─ 17_hot_spot                               │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     786.6 µs      │ 1.722 ms      │ 1.315 ms      │ 1.301 ms      │ 100     │ 100
│  │  │                        12.71 Mitem/s │ 5.804 Mitem/s │ 7.602 Mitem/s │ 7.684 Mitem/s │         │
│  │  ├─ 2                     2.558 ms      │ 4.948 ms      │ 3.966 ms      │ 3.913 ms      │ 100     │ 100
│  │  │                        7.816 Mitem/s │ 4.041 Mitem/s │ 5.041 Mitem/s │ 5.11 Mitem/s  │         │
│  │  ├─ 3                     3.849 ms      │ 7.577 ms      │ 6.388 ms      │ 6.363 ms      │ 100     │ 100
│  │  │                        7.794 Mitem/s │ 3.959 Mitem/s │ 4.696 Mitem/s │ 4.714 Mitem/s │         │
│  │  ├─ 4                     5.342 ms      │ 11.53 ms      │ 8.766 ms      │ 8.837 ms      │ 100     │ 100
│  │  │                        7.487 Mitem/s │ 3.466 Mitem/s │ 4.562 Mitem/s │ 4.526 Mitem/s │         │
│  │  ├─ 5                     7.667 ms      │ 14.81 ms      │ 12.52 ms      │ 12.49 ms      │ 100     │ 100
│  │  │                        6.52 Mitem/s  │ 3.374 Mitem/s │ 3.991 Mitem/s │ 4.001 Mitem/s │         │
│  │  ╰─ 6                     9.878 ms      │ 19.03 ms      │ 16.15 ms      │ 16.19 ms      │ 100     │ 100
│  │                           6.073 Mitem/s │ 3.152 Mitem/s │ 3.714 Mitem/s │ 3.703 Mitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     1.031 ms      │ 2.327 ms      │ 1.713 ms      │ 1.677 ms      │ 100     │ 100
│  │  │                        9.697 Mitem/s │ 4.296 Mitem/s │ 5.835 Mitem/s │ 5.962 Mitem/s │         │
│  │  ├─ 2                     2.3 ms        │ 5.109 ms      │ 4.26 ms       │ 4.142 ms      │ 100     │ 100
│  │  │                        8.692 Mitem/s │ 3.914 Mitem/s │ 4.693 Mitem/s │ 4.828 Mitem/s │         │
│  │  ├─ 3                     3.924 ms      │ 7.065 ms      │ 6.194 ms      │ 6.121 ms      │ 100     │ 100
│  │  │                        7.643 Mitem/s │ 4.246 Mitem/s │ 4.842 Mitem/s │ 4.9 Mitem/s   │         │
│  │  ├─ 4                     4.609 ms      │ 10.9 ms       │ 8.783 ms      │ 8.655 ms      │ 100     │ 100
│  │  │                        8.678 Mitem/s │ 3.668 Mitem/s │ 4.553 Mitem/s │ 4.621 Mitem/s │         │
│  │  ├─ 5                     6.572 ms      │ 12.28 ms      │ 10.85 ms      │ 10.77 ms      │ 100     │ 100
│  │  │                        7.607 Mitem/s │ 4.069 Mitem/s │ 4.607 Mitem/s │ 4.642 Mitem/s │         │
│  │  ╰─ 6                     10.64 ms      │ 15.48 ms      │ 12.93 ms      │ 12.91 ms      │ 100     │ 100
│  │                           5.638 Mitem/s │ 3.875 Mitem/s │ 4.637 Mitem/s │ 4.644 Mitem/s │         │
│  ├─ std_btreemap                           │               │               │               │         │
│  │  ├─ 1                     451.3 µs      │ 1.888 ms      │ 773.1 µs      │ 776.6 µs      │ 100     │ 100
│  │  │                        22.15 Mitem/s │ 5.295 Mitem/s │ 12.93 Mitem/s │ 12.87 Mitem/s │         │
│  │  ├─ 2                     1.803 ms      │ 4.084 ms      │ 3.163 ms      │ 2.988 ms      │ 100     │ 100
│  │  │                        11.09 Mitem/s │ 4.896 Mitem/s │ 6.322 Mitem/s │ 6.691 Mitem/s │         │
│  │  ├─ 3                     3.184 ms      │ 5.137 ms      │ 4.63 ms       │ 4.526 ms      │ 100     │ 100
│  │  │                        9.419 Mitem/s │ 5.839 Mitem/s │ 6.478 Mitem/s │ 6.628 Mitem/s │         │
│  │  ├─ 4                     5.402 ms      │ 8.179 ms      │ 7.05 ms       │ 6.974 ms      │ 100     │ 100
│  │  │                        7.404 Mitem/s │ 4.89 Mitem/s  │ 5.673 Mitem/s │ 5.735 Mitem/s │         │
│  │  ├─ 5                     6.938 ms      │ 10.61 ms      │ 9.276 ms      │ 9.108 ms      │ 100     │ 100
│  │  │                        7.206 Mitem/s │ 4.71 Mitem/s  │ 5.39 Mitem/s  │ 5.489 Mitem/s │         │
│  │  ╰─ 6                     8.15 ms       │ 13.11 ms      │ 11.41 ms      │ 11.34 ms      │ 100     │ 100
│  │                           7.361 Mitem/s │ 4.573 Mitem/s │ 5.254 Mitem/s │ 5.29 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     1.279 ms      │ 2.601 ms      │ 2.151 ms      │ 2.098 ms      │ 100     │ 100
│     │                        7.816 Mitem/s │ 3.843 Mitem/s │ 4.647 Mitem/s │ 4.766 Mitem/s │         │
│     ├─ 2                     1.841 ms      │ 4.043 ms      │ 3.296 ms      │ 3.226 ms      │ 100     │ 100
│     │                        10.86 Mitem/s │ 4.946 Mitem/s │ 6.067 Mitem/s │ 6.198 Mitem/s │         │
│     ├─ 3                     2.735 ms      │ 4.352 ms      │ 3.682 ms      │ 3.684 ms      │ 100     │ 100
│     │                        10.96 Mitem/s │ 6.892 Mitem/s │ 8.145 Mitem/s │ 8.141 Mitem/s │         │
│     ├─ 4                     3.138 ms      │ 4.847 ms      │ 4.094 ms      │ 4.091 ms      │ 100     │ 100
│     │                        12.74 Mitem/s │ 8.252 Mitem/s │ 9.768 Mitem/s │ 9.775 Mitem/s │         │
│     ├─ 5                     3.727 ms      │ 5.007 ms      │ 4.36 ms       │ 4.369 ms      │ 100     │ 100
│     │                        13.41 Mitem/s │ 9.984 Mitem/s │ 11.46 Mitem/s │ 11.44 Mitem/s │         │
│     ╰─ 6                     3.298 ms      │ 5.444 ms      │ 4.442 ms      │ 4.458 ms      │ 100     │ 100
│                              18.18 Mitem/s │ 11.02 Mitem/s │ 13.5 Mitem/s  │ 13.45 Mitem/s │         │
╰─ 18_split_inducing_scan                    │               │               │               │         │
   ├─ indexset                               │               │               │               │         │
   │  ├─ 3                     24.69 ms      │ 45.53 ms      │ 34.38 ms      │ 33.85 ms      │ 100     │ 100
   │  ├─ 4                     32.36 ms      │ 60.54 ms      │ 44.14 ms      │ 44.29 ms      │ 100     │ 100
   │  ├─ 5                     40.67 ms      │ 70.24 ms      │ 55.02 ms      │ 54.83 ms      │ 100     │ 100
   │  ╰─ 6                     59.99 ms      │ 84.07 ms      │ 71.53 ms      │ 71.13 ms      │ 100     │ 100
   ├─ masstree24                             │               │               │               │         │
   │  ├─ 3                     8.249 ms      │ 17.91 ms      │ 15.33 ms      │ 13.39 ms      │ 100     │ 100
   │  ├─ 4                     8.978 ms      │ 26.29 ms      │ 16.3 ms       │ 15.18 ms      │ 100     │ 100
   │  ├─ 5                     10.43 ms      │ 29.1 ms       │ 17.24 ms      │ 18.27 ms      │ 100     │ 100
   │  ╰─ 6                     10.84 ms      │ 30.64 ms      │ 17.12 ms      │ 18.2 ms       │ 100     │ 100
   ├─ std_btreemap                           │               │               │               │         │
   │  ├─ 3                     10.55 ms      │ 19.17 ms      │ 16.33 ms      │ 15.91 ms      │ 100     │ 100
   │  ├─ 4                     14.6 ms       │ 25.86 ms      │ 20.15 ms      │ 20.36 ms      │ 100     │ 100
   │  ├─ 5                     17.25 ms      │ 34.93 ms      │ 22.98 ms      │ 23.07 ms      │ 100     │ 100
   │  ╰─ 6                     18.7 ms       │ 40.89 ms      │ 24.19 ms      │ 25.54 ms      │ 100     │ 100
   ╰─ tree_index                             │               │               │               │         │
      ├─ 3                     9.893 ms      │ 25.73 ms      │ 12.11 ms      │ 14.01 ms      │ 100     │ 100
      ├─ 4                     11.07 ms      │ 23.48 ms      │ 20.14 ms      │ 18.64 ms      │ 100     │ 100
      ├─ 5                     10.98 ms      │ 31.09 ms      │ 18.12 ms      │ 18.06 ms      │ 100     │ 100
      ╰─ 6                     13.55 ms      │ 34.29 ms      │ 19.29 ms      │ 19.72 ms      │ 100     │ 100
```
