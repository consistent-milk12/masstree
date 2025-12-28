```text
Timer precision: 40 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.436 ms      │ 7.33 ms       │ 4.56 ms       │ 5.049 ms      │ 100     │ 100
│  │  │                        2.254 Mitem/s │ 1.364 Mitem/s │ 2.192 Mitem/s │ 1.98 Mitem/s  │         │
│  │  ├─ 2                     11.39 ms      │ 19.79 ms      │ 15.34 ms      │ 15.4 ms       │ 100     │ 100
│  │  │                        1.755 Mitem/s │ 1.01 Mitem/s  │ 1.303 Mitem/s │ 1.297 Mitem/s │         │
│  │  ├─ 3                     22.08 ms      │ 36.39 ms      │ 29.01 ms      │ 29.14 ms      │ 100     │ 100
│  │  │                        1.358 Mitem/s │ 824.1 Kitem/s │ 1.034 Mitem/s │ 1.029 Mitem/s │         │
│  │  ├─ 4                     34.37 ms      │ 54.17 ms      │ 43.63 ms      │ 43.78 ms      │ 100     │ 100
│  │  │                        1.163 Mitem/s │ 738.3 Kitem/s │ 916.7 Kitem/s │ 913.4 Kitem/s │         │
│  │  ├─ 5                     50.57 ms      │ 82.05 ms      │ 60.1 ms       │ 60.71 ms      │ 100     │ 100
│  │  │                        988.6 Kitem/s │ 609.3 Kitem/s │ 831.8 Kitem/s │ 823.4 Kitem/s │         │
│  │  ╰─ 6                     79.59 ms      │ 110.6 ms      │ 95.57 ms      │ 95.38 ms      │ 100     │ 100
│  │                           753.7 Kitem/s │ 542.2 Kitem/s │ 627.7 Kitem/s │ 629 Kitem/s   │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.374 ms      │ 12.15 ms      │ 9.573 ms      │ 9.862 ms      │ 100     │ 100
│  │  │                        1.066 Mitem/s │ 822.5 Kitem/s │ 1.044 Mitem/s │ 1.013 Mitem/s │         │
│  │  ├─ 2                     10.23 ms      │ 22.2 ms       │ 14.22 ms      │ 14.26 ms      │ 100     │ 100
│  │  │                        1.955 Mitem/s │ 900.5 Kitem/s │ 1.405 Mitem/s │ 1.402 Mitem/s │         │
│  │  ├─ 3                     10.73 ms      │ 22.36 ms      │ 15.17 ms      │ 16.05 ms      │ 100     │ 100
│  │  │                        2.794 Mitem/s │ 1.341 Mitem/s │ 1.977 Mitem/s │ 1.868 Mitem/s │         │
│  │  ├─ 4                     10.86 ms      │ 30.63 ms      │ 17.42 ms      │ 17.48 ms      │ 100     │ 100
│  │  │                        3.681 Mitem/s │ 1.305 Mitem/s │ 2.296 Mitem/s │ 2.287 Mitem/s │         │
│  │  ├─ 5                     11.89 ms      │ 31.22 ms      │ 17.64 ms      │ 18.67 ms      │ 100     │ 100
│  │  │                        4.202 Mitem/s │ 1.601 Mitem/s │ 2.834 Mitem/s │ 2.678 Mitem/s │         │
│  │  ╰─ 6                     10.84 ms      │ 30.66 ms      │ 17.97 ms      │ 19.18 ms      │ 100     │ 100
│  │                           5.532 Mitem/s │ 1.956 Mitem/s │ 3.338 Mitem/s │ 3.127 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.549 ms      │ 10.2 ms       │ 8.791 ms      │ 8.913 ms      │ 100     │ 100
│     │                        1.169 Mitem/s │ 979.9 Kitem/s │ 1.137 Mitem/s │ 1.121 Mitem/s │         │
│     ├─ 2                     8.83 ms       │ 18.68 ms      │ 12.36 ms      │ 12.18 ms      │ 100     │ 100
│     │                        2.264 Mitem/s │ 1.07 Mitem/s  │ 1.617 Mitem/s │ 1.64 Mitem/s  │         │
│     ├─ 3                     8.995 ms      │ 21.35 ms      │ 13.62 ms      │ 14.31 ms      │ 100     │ 100
│     │                        3.334 Mitem/s │ 1.404 Mitem/s │ 2.201 Mitem/s │ 2.095 Mitem/s │         │
│     ├─ 4                     8.994 ms      │ 23.56 ms      │ 15.62 ms      │ 14.92 ms      │ 100     │ 100
│     │                        4.447 Mitem/s │ 1.697 Mitem/s │ 2.56 Mitem/s  │ 2.679 Mitem/s │         │
│     ├─ 5                     9.244 ms      │ 28.2 ms       │ 15.95 ms      │ 16.4 ms       │ 100     │ 100
│     │                        5.408 Mitem/s │ 1.772 Mitem/s │ 3.134 Mitem/s │ 3.048 Mitem/s │         │
│     ╰─ 6                     10.89 ms      │ 32.38 ms      │ 17.8 ms       │ 19.54 ms      │ 100     │ 100
│                              5.508 Mitem/s │ 1.852 Mitem/s │ 3.37 Mitem/s  │ 3.069 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.624 ms      │ 7.161 ms      │ 4.764 ms      │ 5.079 ms      │ 100     │ 100
│  │  │                        2.162 Mitem/s │ 1.396 Mitem/s │ 2.098 Mitem/s │ 1.968 Mitem/s │         │
│  │  ├─ 2                     11.48 ms      │ 22.72 ms      │ 15.85 ms      │ 15.7 ms       │ 100     │ 100
│  │  │                        1.741 Mitem/s │ 880.1 Kitem/s │ 1.261 Mitem/s │ 1.273 Mitem/s │         │
│  │  ├─ 3                     21.36 ms      │ 36.65 ms      │ 27.97 ms      │ 28.69 ms      │ 100     │ 100
│  │  │                        1.404 Mitem/s │ 818.4 Kitem/s │ 1.072 Mitem/s │ 1.045 Mitem/s │         │
│  │  ├─ 4                     35.79 ms      │ 56.5 ms       │ 43.58 ms      │ 43.68 ms      │ 100     │ 100
│  │  │                        1.117 Mitem/s │ 707.9 Kitem/s │ 917.7 Kitem/s │ 915.7 Kitem/s │         │
│  │  ├─ 5                     54.75 ms      │ 73.71 ms      │ 62.98 ms      │ 63.1 ms       │ 100     │ 100
│  │  │                        913.1 Kitem/s │ 678.3 Kitem/s │ 793.7 Kitem/s │ 792.2 Kitem/s │         │
│  │  ╰─ 6                     80.96 ms      │ 126.6 ms      │ 96.39 ms      │ 96.47 ms      │ 100     │ 100
│  │                           741 Kitem/s   │ 473.6 Kitem/s │ 622.4 Kitem/s │ 621.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.365 ms      │ 13.69 ms      │ 9.546 ms      │ 10.29 ms      │ 100     │ 100
│  │  │                        1.067 Mitem/s │ 730.2 Kitem/s │ 1.047 Mitem/s │ 971.4 Kitem/s │         │
│  │  ├─ 2                     9.797 ms      │ 21.07 ms      │ 12.59 ms      │ 13.31 ms      │ 100     │ 100
│  │  │                        2.041 Mitem/s │ 949 Kitem/s   │ 1.587 Mitem/s │ 1.501 Mitem/s │         │
│  │  ├─ 3                     10.98 ms      │ 20.95 ms      │ 15.14 ms      │ 16.08 ms      │ 100     │ 100
│  │  │                        2.73 Mitem/s  │ 1.431 Mitem/s │ 1.981 Mitem/s │ 1.864 Mitem/s │         │
│  │  ├─ 4                     11.26 ms      │ 30.53 ms      │ 16.04 ms      │ 16.65 ms      │ 100     │ 100
│  │  │                        3.549 Mitem/s │ 1.31 Mitem/s  │ 2.492 Mitem/s │ 2.401 Mitem/s │         │
│  │  ├─ 5                     10.53 ms      │ 32.08 ms      │ 17.01 ms      │ 17.18 ms      │ 100     │ 100
│  │  │                        4.745 Mitem/s │ 1.558 Mitem/s │ 2.937 Mitem/s │ 2.908 Mitem/s │         │
│  │  ╰─ 6                     10.72 ms      │ 30.56 ms      │ 17.74 ms      │ 19.61 ms      │ 100     │ 100
│  │                           5.592 Mitem/s │ 1.962 Mitem/s │ 3.382 Mitem/s │ 3.059 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.91 ms       │ 13.18 ms      │ 9.335 ms      │ 9.711 ms      │ 100     │ 100
│     │                        1.122 Mitem/s │ 758.7 Kitem/s │ 1.071 Mitem/s │ 1.029 Mitem/s │         │
│     ├─ 2                     9.088 ms      │ 19.7 ms       │ 13.17 ms      │ 13.1 ms       │ 100     │ 100
│     │                        2.2 Mitem/s   │ 1.014 Mitem/s │ 1.518 Mitem/s │ 1.525 Mitem/s │         │
│     ├─ 3                     9.258 ms      │ 21.13 ms      │ 13.86 ms      │ 14.6 ms       │ 100     │ 100
│     │                        3.24 Mitem/s  │ 1.419 Mitem/s │ 2.163 Mitem/s │ 2.053 Mitem/s │         │
│     ├─ 4                     9.392 ms      │ 26.12 ms      │ 16.42 ms      │ 16.3 ms       │ 100     │ 100
│     │                        4.258 Mitem/s │ 1.53 Mitem/s  │ 2.435 Mitem/s │ 2.453 Mitem/s │         │
│     ├─ 5                     9.379 ms      │ 28.44 ms      │ 16.78 ms      │ 16.95 ms      │ 100     │ 100
│     │                        5.33 Mitem/s  │ 1.758 Mitem/s │ 2.979 Mitem/s │ 2.948 Mitem/s │         │
│     ╰─ 6                     12.53 ms      │ 36.66 ms      │ 16.8 ms       │ 18.75 ms      │ 100     │ 100
│                              4.787 Mitem/s │ 1.636 Mitem/s │ 3.569 Mitem/s │ 3.198 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.442 ms      │ 8.362 ms      │ 4.596 ms      │ 4.876 ms      │ 100     │ 100
│  │  │                        2.25 Mitem/s  │ 1.195 Mitem/s │ 2.175 Mitem/s │ 2.05 Mitem/s  │         │
│  │  ├─ 2                     11.83 ms      │ 21.4 ms       │ 15.95 ms      │ 15.8 ms       │ 100     │ 100
│  │  │                        1.69 Mitem/s  │ 934.3 Kitem/s │ 1.253 Mitem/s │ 1.265 Mitem/s │         │
│  │  ├─ 3                     21.83 ms      │ 34.21 ms      │ 29.55 ms      │ 29.34 ms      │ 100     │ 100
│  │  │                        1.373 Mitem/s │ 876.9 Kitem/s │ 1.015 Mitem/s │ 1.022 Mitem/s │         │
│  │  ├─ 4                     32.57 ms      │ 54.82 ms      │ 42.93 ms      │ 43.21 ms      │ 100     │ 100
│  │  │                        1.228 Mitem/s │ 729.5 Kitem/s │ 931.6 Kitem/s │ 925.5 Kitem/s │         │
│  │  ├─ 5                     51.35 ms      │ 76.88 ms      │ 61.48 ms      │ 61.64 ms      │ 100     │ 100
│  │  │                        973.5 Kitem/s │ 650.2 Kitem/s │ 813.2 Kitem/s │ 811 Kitem/s   │         │
│  │  ╰─ 6                     80.05 ms      │ 112 ms        │ 96.29 ms      │ 96.46 ms      │ 100     │ 100
│  │                           749.5 Kitem/s │ 535.4 Kitem/s │ 623 Kitem/s   │ 621.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.418 ms      │ 12.3 ms       │ 9.611 ms      │ 9.889 ms      │ 100     │ 100
│  │  │                        1.061 Mitem/s │ 812.9 Kitem/s │ 1.04 Mitem/s  │ 1.011 Mitem/s │         │
│  │  ├─ 2                     9.828 ms      │ 21.08 ms      │ 14.03 ms      │ 13.65 ms      │ 100     │ 100
│  │  │                        2.034 Mitem/s │ 948.3 Kitem/s │ 1.424 Mitem/s │ 1.465 Mitem/s │         │
│  │  ├─ 3                     9.796 ms      │ 20.42 ms      │ 14.95 ms      │ 15.35 ms      │ 100     │ 100
│  │  │                        3.062 Mitem/s │ 1.468 Mitem/s │ 2.005 Mitem/s │ 1.954 Mitem/s │         │
│  │  ├─ 4                     10.59 ms      │ 30.7 ms       │ 15.09 ms      │ 15.71 ms      │ 100     │ 100
│  │  │                        3.777 Mitem/s │ 1.302 Mitem/s │ 2.649 Mitem/s │ 2.544 Mitem/s │         │
│  │  ├─ 5                     10.92 ms      │ 29.09 ms      │ 17.5 ms       │ 17.65 ms      │ 100     │ 100
│  │  │                        4.576 Mitem/s │ 1.718 Mitem/s │ 2.855 Mitem/s │ 2.831 Mitem/s │         │
│  │  ╰─ 6                     12.12 ms      │ 31.34 ms      │ 17.97 ms      │ 19.2 ms       │ 100     │ 100
│  │                           4.95 Mitem/s  │ 1.914 Mitem/s │ 3.338 Mitem/s │ 3.124 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.694 ms      │ 16.51 ms      │ 9.157 ms      │ 9.627 ms      │ 100     │ 100
│     │                        1.15 Mitem/s  │ 605.4 Kitem/s │ 1.091 Mitem/s │ 1.038 Mitem/s │         │
│     ├─ 2                     8.815 ms      │ 20.12 ms      │ 12.1 ms       │ 11.84 ms      │ 100     │ 100
│     │                        2.268 Mitem/s │ 993.5 Kitem/s │ 1.652 Mitem/s │ 1.687 Mitem/s │         │
│     ├─ 3                     9.081 ms      │ 21.63 ms      │ 15.21 ms      │ 15.26 ms      │ 100     │ 100
│     │                        3.303 Mitem/s │ 1.386 Mitem/s │ 1.972 Mitem/s │ 1.965 Mitem/s │         │
│     ├─ 4                     10.86 ms      │ 23.72 ms      │ 16.32 ms      │ 16.19 ms      │ 100     │ 100
│     │                        3.682 Mitem/s │ 1.685 Mitem/s │ 2.45 Mitem/s  │ 2.469 Mitem/s │         │
│     ├─ 5                     9.193 ms      │ 27.76 ms      │ 15.66 ms      │ 15.67 ms      │ 100     │ 100
│     │                        5.438 Mitem/s │ 1.8 Mitem/s   │ 3.191 Mitem/s │ 3.189 Mitem/s │         │
│     ╰─ 6                     9.198 ms      │ 28.21 ms      │ 16.03 ms      │ 16.58 ms      │ 100     │ 100
│                              6.522 Mitem/s │ 2.126 Mitem/s │ 3.742 Mitem/s │ 3.617 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.521 ms      │ 8.425 ms      │ 4.646 ms      │ 5.345 ms      │ 100     │ 100
│  │  │                        2.211 Mitem/s │ 1.186 Mitem/s │ 2.151 Mitem/s │ 1.87 Mitem/s  │         │
│  │  ├─ 2                     10.78 ms      │ 20.23 ms      │ 14.81 ms      │ 15.17 ms      │ 100     │ 100
│  │  │                        1.854 Mitem/s │ 988.4 Kitem/s │ 1.35 Mitem/s  │ 1.318 Mitem/s │         │
│  │  ├─ 3                     20.41 ms      │ 35.22 ms      │ 28.54 ms      │ 28.52 ms      │ 100     │ 100
│  │  │                        1.469 Mitem/s │ 851.7 Kitem/s │ 1.051 Mitem/s │ 1.051 Mitem/s │         │
│  │  ├─ 4                     34.42 ms      │ 53.94 ms      │ 44.33 ms      │ 44.15 ms      │ 100     │ 100
│  │  │                        1.161 Mitem/s │ 741.5 Kitem/s │ 902.1 Kitem/s │ 905.9 Kitem/s │         │
│  │  ├─ 5                     52.95 ms      │ 77.49 ms      │ 60.39 ms      │ 61.45 ms      │ 100     │ 100
│  │  │                        944.2 Kitem/s │ 645.1 Kitem/s │ 827.9 Kitem/s │ 813.5 Kitem/s │         │
│  │  ╰─ 6                     80.28 ms      │ 116.3 ms      │ 94.73 ms      │ 95.48 ms      │ 100     │ 100
│  │                           747.3 Kitem/s │ 515.6 Kitem/s │ 633.3 Kitem/s │ 628.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.498 ms      │ 13.97 ms      │ 10.13 ms      │ 10.52 ms      │ 100     │ 100
│  │  │                        1.052 Mitem/s │ 715.5 Kitem/s │ 986.5 Kitem/s │ 950.5 Kitem/s │         │
│  │  ├─ 2                     10.12 ms      │ 20.79 ms      │ 14.13 ms      │ 14.24 ms      │ 100     │ 100
│  │  │                        1.974 Mitem/s │ 961.9 Kitem/s │ 1.414 Mitem/s │ 1.404 Mitem/s │         │
│  │  ├─ 3                     9.894 ms      │ 20.9 ms       │ 14.76 ms      │ 15.19 ms      │ 100     │ 100
│  │  │                        3.031 Mitem/s │ 1.434 Mitem/s │ 2.031 Mitem/s │ 1.974 Mitem/s │         │
│  │  ├─ 4                     11.21 ms      │ 26.17 ms      │ 15.59 ms      │ 16.84 ms      │ 100     │ 100
│  │  │                        3.566 Mitem/s │ 1.528 Mitem/s │ 2.565 Mitem/s │ 2.373 Mitem/s │         │
│  │  ├─ 5                     11.85 ms      │ 33.17 ms      │ 17.61 ms      │ 18.75 ms      │ 100     │ 100
│  │  │                        4.218 Mitem/s │ 1.507 Mitem/s │ 2.839 Mitem/s │ 2.665 Mitem/s │         │
│  │  ╰─ 6                     11.2 ms       │ 30.23 ms      │ 17.99 ms      │ 19.36 ms      │ 100     │ 100
│  │                           5.355 Mitem/s │ 1.984 Mitem/s │ 3.334 Mitem/s │ 3.098 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.61 ms       │ 15.95 ms      │ 9.585 ms      │ 10.38 ms      │ 100     │ 100
│     │                        1.161 Mitem/s │ 626.7 Kitem/s │ 1.043 Mitem/s │ 962.4 Kitem/s │         │
│     ├─ 2                     9.05 ms       │ 18.81 ms      │ 13.45 ms      │ 14.37 ms      │ 100     │ 100
│     │                        2.209 Mitem/s │ 1.062 Mitem/s │ 1.486 Mitem/s │ 1.39 Mitem/s  │         │
│     ├─ 3                     8.94 ms       │ 18.74 ms      │ 13.25 ms      │ 13.2 ms       │ 100     │ 100
│     │                        3.355 Mitem/s │ 1.6 Mitem/s   │ 2.263 Mitem/s │ 2.271 Mitem/s │         │
│     ├─ 4                     9.385 ms      │ 27.33 ms      │ 13.79 ms      │ 14.29 ms      │ 100     │ 100
│     │                        4.262 Mitem/s │ 1.463 Mitem/s │ 2.898 Mitem/s │ 2.798 Mitem/s │         │
│     ├─ 5                     9.135 ms      │ 31.43 ms      │ 15.62 ms      │ 15.09 ms      │ 100     │ 100
│     │                        5.473 Mitem/s │ 1.59 Mitem/s  │ 3.2 Mitem/s   │ 3.311 Mitem/s │         │
│     ╰─ 6                     9.959 ms      │ 29.89 ms      │ 16.73 ms      │ 18.07 ms      │ 100     │ 100
│                              6.024 Mitem/s │ 2.007 Mitem/s │ 3.585 Mitem/s │ 3.318 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.122 ms      │ 5.59 ms       │ 3.253 ms      │ 3.5 ms        │ 100     │ 100
│  │  │                        3.203 Mitem/s │ 1.788 Mitem/s │ 3.073 Mitem/s │ 2.857 Mitem/s │         │
│  │  ├─ 2                     8.141 ms      │ 14.5 ms       │ 11.63 ms      │ 11.59 ms      │ 100     │ 100
│  │  │                        2.456 Mitem/s │ 1.378 Mitem/s │ 1.718 Mitem/s │ 1.725 Mitem/s │         │
│  │  ├─ 3                     16.35 ms      │ 30.22 ms      │ 21.23 ms      │ 21.63 ms      │ 100     │ 100
│  │  │                        1.833 Mitem/s │ 992.7 Kitem/s │ 1.412 Mitem/s │ 1.386 Mitem/s │         │
│  │  ├─ 4                     27.12 ms      │ 51.89 ms      │ 31.63 ms      │ 32.4 ms       │ 100     │ 100
│  │  │                        1.474 Mitem/s │ 770.8 Kitem/s │ 1.264 Mitem/s │ 1.234 Mitem/s │         │
│  │  ├─ 5                     38.32 ms      │ 60.18 ms      │ 46.87 ms      │ 47.65 ms      │ 100     │ 100
│  │  │                        1.304 Mitem/s │ 830.7 Kitem/s │ 1.066 Mitem/s │ 1.049 Mitem/s │         │
│  │  ╰─ 6                     58.69 ms      │ 93.07 ms      │ 73.3 ms       │ 74.17 ms      │ 100     │ 100
│  │                           1.022 Mitem/s │ 644.6 Kitem/s │ 818.5 Kitem/s │ 808.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     8.753 ms      │ 13.58 ms      │ 9.149 ms      │ 10.27 ms      │ 100     │ 100
│  │  │                        1.142 Mitem/s │ 736.1 Kitem/s │ 1.092 Mitem/s │ 973.4 Kitem/s │         │
│  │  ├─ 2                     9.803 ms      │ 21.56 ms      │ 13.86 ms      │ 15.14 ms      │ 100     │ 100
│  │  │                        2.04 Mitem/s  │ 927.5 Kitem/s │ 1.441 Mitem/s │ 1.32 Mitem/s  │         │
│  │  ├─ 3                     9.834 ms      │ 19.98 ms      │ 14.16 ms      │ 15.14 ms      │ 100     │ 100
│  │  │                        3.05 Mitem/s  │ 1.501 Mitem/s │ 2.117 Mitem/s │ 1.98 Mitem/s  │         │
│  │  ├─ 4                     9.955 ms      │ 24.05 ms      │ 15.63 ms      │ 15.71 ms      │ 100     │ 100
│  │  │                        4.017 Mitem/s │ 1.663 Mitem/s │ 2.558 Mitem/s │ 2.545 Mitem/s │         │
│  │  ├─ 5                     10.03 ms      │ 27.85 ms      │ 16.53 ms      │ 16.95 ms      │ 100     │ 100
│  │  │                        4.983 Mitem/s │ 1.795 Mitem/s │ 3.024 Mitem/s │ 2.948 Mitem/s │         │
│  │  ╰─ 6                     10.15 ms      │ 27.99 ms      │ 16.61 ms      │ 17.75 ms      │ 100     │ 100
│  │                           5.905 Mitem/s │ 2.143 Mitem/s │ 3.611 Mitem/s │ 3.378 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.458 ms      │ 11.64 ms      │ 8.772 ms      │ 8.837 ms      │ 100     │ 100
│     │                        1.182 Mitem/s │ 858.7 Kitem/s │ 1.139 Mitem/s │ 1.131 Mitem/s │         │
│     ├─ 2                     8.763 ms      │ 19.01 ms      │ 12.19 ms      │ 12.24 ms      │ 100     │ 100
│     │                        2.282 Mitem/s │ 1.051 Mitem/s │ 1.639 Mitem/s │ 1.633 Mitem/s │         │
│     ├─ 3                     10.09 ms      │ 19.26 ms      │ 13.45 ms      │ 14.37 ms      │ 100     │ 100
│     │                        2.971 Mitem/s │ 1.557 Mitem/s │ 2.229 Mitem/s │ 2.086 Mitem/s │         │
│     ├─ 4                     9.449 ms      │ 26.69 ms      │ 15.19 ms      │ 15.45 ms      │ 100     │ 100
│     │                        4.233 Mitem/s │ 1.498 Mitem/s │ 2.633 Mitem/s │ 2.588 Mitem/s │         │
│     ├─ 5                     9.358 ms      │ 28.5 ms       │ 15.5 ms       │ 16.06 ms      │ 100     │ 100
│     │                        5.342 Mitem/s │ 1.754 Mitem/s │ 3.225 Mitem/s │ 3.111 Mitem/s │         │
│     ╰─ 6                     9.253 ms      │ 28.72 ms      │ 16.19 ms      │ 16.87 ms      │ 100     │ 100
│                              6.484 Mitem/s │ 2.089 Mitem/s │ 3.704 Mitem/s │ 3.555 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.232 ms      │ 5.396 ms      │ 3.331 ms      │ 3.511 ms      │ 100     │ 100
│  │  │                        3.093 Mitem/s │ 1.853 Mitem/s │ 3.001 Mitem/s │ 2.847 Mitem/s │         │
│  │  ├─ 2                     8.07 ms       │ 15.18 ms      │ 11.12 ms      │ 11.28 ms      │ 100     │ 100
│  │  │                        2.478 Mitem/s │ 1.317 Mitem/s │ 1.797 Mitem/s │ 1.772 Mitem/s │         │
│  │  ├─ 3                     15.45 ms      │ 24.25 ms      │ 20.25 ms      │ 20.14 ms      │ 100     │ 100
│  │  │                        1.94 Mitem/s  │ 1.236 Mitem/s │ 1.481 Mitem/s │ 1.488 Mitem/s │         │
│  │  ├─ 4                     26.84 ms      │ 42.35 ms      │ 32.36 ms      │ 32.85 ms      │ 100     │ 100
│  │  │                        1.489 Mitem/s │ 944.3 Kitem/s │ 1.235 Mitem/s │ 1.217 Mitem/s │         │
│  │  ├─ 5                     37.59 ms      │ 58.94 ms      │ 47.36 ms      │ 47.39 ms      │ 100     │ 100
│  │  │                        1.33 Mitem/s  │ 848.3 Kitem/s │ 1.055 Mitem/s │ 1.055 Mitem/s │         │
│  │  ╰─ 6                     60.4 ms       │ 85.28 ms      │ 72.32 ms      │ 72.46 ms      │ 100     │ 100
│  │                           993.2 Kitem/s │ 703.5 Kitem/s │ 829.5 Kitem/s │ 828 Kitem/s   │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.815 ms      │ 12.77 ms      │ 10.03 ms      │ 10.36 ms      │ 100     │ 100
│  │  │                        1.018 Mitem/s │ 782.6 Kitem/s │ 996.8 Kitem/s │ 965.1 Kitem/s │         │
│  │  ├─ 2                     9.949 ms      │ 20.99 ms      │ 13.82 ms      │ 13.35 ms      │ 100     │ 100
│  │  │                        2.01 Mitem/s  │ 952.3 Kitem/s │ 1.446 Mitem/s │ 1.497 Mitem/s │         │
│  │  ├─ 3                     10.07 ms      │ 21.11 ms      │ 15.13 ms      │ 15.78 ms      │ 100     │ 100
│  │  │                        2.977 Mitem/s │ 1.42 Mitem/s  │ 1.982 Mitem/s │ 1.9 Mitem/s   │         │
│  │  ├─ 4                     10.28 ms      │ 26.06 ms      │ 17.24 ms      │ 16.59 ms      │ 100     │ 100
│  │  │                        3.888 Mitem/s │ 1.534 Mitem/s │ 2.319 Mitem/s │ 2.409 Mitem/s │         │
│  │  ├─ 5                     10.79 ms      │ 33.09 ms      │ 17.81 ms      │ 18.29 ms      │ 100     │ 100
│  │  │                        4.633 Mitem/s │ 1.511 Mitem/s │ 2.805 Mitem/s │ 2.732 Mitem/s │         │
│  │  ╰─ 6                     15.12 ms      │ 32.52 ms      │ 19.44 ms      │ 21.82 ms      │ 100     │ 100
│  │                           3.965 Mitem/s │ 1.844 Mitem/s │ 3.084 Mitem/s │ 2.749 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.107 ms      │ 13.98 ms      │ 9.741 ms      │ 10.17 ms      │ 100     │ 100
│     │                        1.097 Mitem/s │ 714.8 Kitem/s │ 1.026 Mitem/s │ 982.9 Kitem/s │         │
│     ├─ 2                     9.537 ms      │ 19.71 ms      │ 13.45 ms      │ 13.51 ms      │ 100     │ 100
│     │                        2.096 Mitem/s │ 1.014 Mitem/s │ 1.486 Mitem/s │ 1.479 Mitem/s │         │
│     ├─ 3                     9.75 ms       │ 20.4 ms       │ 14.3 ms       │ 15.19 ms      │ 100     │ 100
│     │                        3.076 Mitem/s │ 1.47 Mitem/s  │ 2.097 Mitem/s │ 1.974 Mitem/s │         │
│     ├─ 4                     9.955 ms      │ 23.16 ms      │ 14.69 ms      │ 15.13 ms      │ 100     │ 100
│     │                        4.017 Mitem/s │ 1.726 Mitem/s │ 2.721 Mitem/s │ 2.642 Mitem/s │         │
│     ├─ 5                     11.38 ms      │ 30.04 ms      │ 17.77 ms      │ 18.21 ms      │ 100     │ 100
│     │                        4.393 Mitem/s │ 1.664 Mitem/s │ 2.813 Mitem/s │ 2.744 Mitem/s │         │
│     ╰─ 6                     10.24 ms      │ 32.81 ms      │ 18.09 ms      │ 19.42 ms      │ 100     │ 100
│                              5.857 Mitem/s │ 1.828 Mitem/s │ 3.315 Mitem/s │ 3.089 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.245 ms      │ 6.668 ms      │ 3.387 ms      │ 4.11 ms       │ 100     │ 100
│  │  │                        3.08 Mitem/s  │ 1.499 Mitem/s │ 2.952 Mitem/s │ 2.433 Mitem/s │         │
│  │  ├─ 2                     7.86 ms       │ 15.75 ms      │ 10.9 ms       │ 11.5 ms       │ 100     │ 100
│  │  │                        2.544 Mitem/s │ 1.269 Mitem/s │ 1.833 Mitem/s │ 1.737 Mitem/s │         │
│  │  ├─ 3                     15.83 ms      │ 25.02 ms      │ 21.18 ms      │ 21.15 ms      │ 100     │ 100
│  │  │                        1.894 Mitem/s │ 1.198 Mitem/s │ 1.415 Mitem/s │ 1.418 Mitem/s │         │
│  │  ├─ 4                     26.03 ms      │ 38.91 ms      │ 31.24 ms      │ 31.51 ms      │ 100     │ 100
│  │  │                        1.536 Mitem/s │ 1.027 Mitem/s │ 1.28 Mitem/s  │ 1.269 Mitem/s │         │
│  │  ├─ 5                     34.85 ms      │ 54.58 ms      │ 45.39 ms      │ 44.52 ms      │ 100     │ 100
│  │  │                        1.434 Mitem/s │ 915.9 Kitem/s │ 1.101 Mitem/s │ 1.122 Mitem/s │         │
│  │  ╰─ 6                     60.71 ms      │ 84.17 ms      │ 73.33 ms      │ 72.97 ms      │ 100     │ 100
│  │                           988.2 Kitem/s │ 712.7 Kitem/s │ 818.1 Kitem/s │ 822.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     13.11 ms      │ 20.02 ms      │ 13.35 ms      │ 14.04 ms      │ 100     │ 100
│  │  │                        762.5 Kitem/s │ 499.2 Kitem/s │ 748.7 Kitem/s │ 711.8 Kitem/s │         │
│  │  ├─ 2                     13.6 ms       │ 26 ms         │ 15.56 ms      │ 16.23 ms      │ 100     │ 100
│  │  │                        1.469 Mitem/s │ 769.1 Kitem/s │ 1.284 Mitem/s │ 1.231 Mitem/s │         │
│  │  ├─ 3                     14.26 ms      │ 27.41 ms      │ 20.25 ms      │ 20.35 ms      │ 100     │ 100
│  │  │                        2.103 Mitem/s │ 1.094 Mitem/s │ 1.481 Mitem/s │ 1.473 Mitem/s │         │
│  │  ├─ 4                     14.65 ms      │ 39.68 ms      │ 22.12 ms      │ 22.1 ms       │ 100     │ 100
│  │  │                        2.729 Mitem/s │ 1.007 Mitem/s │ 1.808 Mitem/s │ 1.809 Mitem/s │         │
│  │  ├─ 5                     15.05 ms      │ 38.65 ms      │ 23.65 ms      │ 24.55 ms      │ 100     │ 100
│  │  │                        3.32 Mitem/s  │ 1.293 Mitem/s │ 2.114 Mitem/s │ 2.036 Mitem/s │         │
│  │  ╰─ 6                     14.76 ms      │ 39.02 ms      │ 23.71 ms      │ 24 ms         │ 100     │ 100
│  │                           4.064 Mitem/s │ 1.537 Mitem/s │ 2.53 Mitem/s  │ 2.499 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.251 ms      │ 13.68 ms      │ 9.839 ms      │ 10.22 ms      │ 100     │ 100
│     │                        1.08 Mitem/s  │ 730.9 Kitem/s │ 1.016 Mitem/s │ 978 Kitem/s   │         │
│     ├─ 2                     9.471 ms      │ 21.23 ms      │ 13.14 ms      │ 13.04 ms      │ 100     │ 100
│     │                        2.111 Mitem/s │ 941.6 Kitem/s │ 1.521 Mitem/s │ 1.533 Mitem/s │         │
│     ├─ 3                     9.764 ms      │ 20.71 ms      │ 17.17 ms      │ 16.6 ms       │ 100     │ 100
│     │                        3.072 Mitem/s │ 1.448 Mitem/s │ 1.746 Mitem/s │ 1.806 Mitem/s │         │
│     ├─ 4                     10.28 ms      │ 29.77 ms      │ 16.9 ms       │ 16.52 ms      │ 100     │ 100
│     │                        3.887 Mitem/s │ 1.343 Mitem/s │ 2.365 Mitem/s │ 2.42 Mitem/s  │         │
│     ├─ 5                     9.94 ms       │ 31.35 ms      │ 17.74 ms      │ 17.89 ms      │ 100     │ 100
│     │                        5.029 Mitem/s │ 1.594 Mitem/s │ 2.818 Mitem/s │ 2.793 Mitem/s │         │
│     ╰─ 6                     10.96 ms      │ 30.72 ms      │ 19.86 ms      │ 20.15 ms      │ 100     │ 100
│                              5.473 Mitem/s │ 1.952 Mitem/s │ 3.02 Mitem/s  │ 2.977 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.589 ms      │ 7.261 ms      │ 4.74 ms       │ 5.005 ms      │ 100     │ 100
│  │  │                        2.179 Mitem/s │ 1.377 Mitem/s │ 2.109 Mitem/s │ 1.997 Mitem/s │         │
│  │  ├─ 2                     10.4 ms       │ 19.65 ms      │ 13.17 ms      │ 13.92 ms      │ 100     │ 100
│  │  │                        1.923 Mitem/s │ 1.017 Mitem/s │ 1.518 Mitem/s │ 1.436 Mitem/s │         │
│  │  ├─ 3                     21.09 ms      │ 36.24 ms      │ 26.17 ms      │ 26.24 ms      │ 100     │ 100
│  │  │                        1.422 Mitem/s │ 827.6 Kitem/s │ 1.146 Mitem/s │ 1.143 Mitem/s │         │
│  │  ├─ 4                     32.9 ms       │ 54.48 ms      │ 40.44 ms      │ 41.2 ms       │ 100     │ 100
│  │  │                        1.215 Mitem/s │ 734.1 Kitem/s │ 989 Kitem/s   │ 970.7 Kitem/s │         │
│  │  ├─ 5                     50.65 ms      │ 68.58 ms      │ 58.74 ms      │ 58.68 ms      │ 100     │ 100
│  │  │                        987.1 Kitem/s │ 729 Kitem/s   │ 851.1 Kitem/s │ 852 Kitem/s   │         │
│  │  ╰─ 6                     80.13 ms      │ 113.4 ms      │ 95.46 ms      │ 95.15 ms      │ 100     │ 100
│  │                           748.7 Kitem/s │ 528.7 Kitem/s │ 628.4 Kitem/s │ 630.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     8.833 ms      │ 14.21 ms      │ 9.207 ms      │ 9.748 ms      │ 100     │ 100
│  │  │                        1.132 Mitem/s │ 703.6 Kitem/s │ 1.086 Mitem/s │ 1.025 Mitem/s │         │
│  │  ├─ 2                     9.459 ms      │ 19.54 ms      │ 13.67 ms      │ 14.21 ms      │ 100     │ 100
│  │  │                        2.114 Mitem/s │ 1.023 Mitem/s │ 1.462 Mitem/s │ 1.407 Mitem/s │         │
│  │  ├─ 3                     10.18 ms      │ 22.08 ms      │ 15.02 ms      │ 15.18 ms      │ 100     │ 100
│  │  │                        2.946 Mitem/s │ 1.358 Mitem/s │ 1.996 Mitem/s │ 1.975 Mitem/s │         │
│  │  ├─ 4                     10.09 ms      │ 27.28 ms      │ 14.88 ms      │ 15.39 ms      │ 100     │ 100
│  │  │                        3.96 Mitem/s  │ 1.466 Mitem/s │ 2.687 Mitem/s │ 2.598 Mitem/s │         │
│  │  ├─ 5                     10.22 ms      │ 28.77 ms      │ 16.56 ms      │ 16.71 ms      │ 100     │ 100
│  │  │                        4.89 Mitem/s  │ 1.737 Mitem/s │ 3.017 Mitem/s │ 2.991 Mitem/s │         │
│  │  ╰─ 6                     10.09 ms      │ 28.24 ms      │ 17 ms         │ 18.34 ms      │ 100     │ 100
│  │                           5.944 Mitem/s │ 2.124 Mitem/s │ 3.527 Mitem/s │ 3.27 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.027 ms      │ 12.14 ms      │ 8.355 ms      │ 8.739 ms      │ 100     │ 100
│     │                        1.245 Mitem/s │ 823.3 Kitem/s │ 1.196 Mitem/s │ 1.144 Mitem/s │         │
│     ├─ 2                     8.251 ms      │ 16.48 ms      │ 11.36 ms      │ 11.22 ms      │ 100     │ 100
│     │                        2.423 Mitem/s │ 1.213 Mitem/s │ 1.759 Mitem/s │ 1.781 Mitem/s │         │
│     ├─ 3                     8.343 ms      │ 23.32 ms      │ 12.9 ms       │ 13.01 ms      │ 100     │ 100
│     │                        3.595 Mitem/s │ 1.285 Mitem/s │ 2.324 Mitem/s │ 2.305 Mitem/s │         │
│     ├─ 4                     8.609 ms      │ 27.57 ms      │ 13.25 ms      │ 13.88 ms      │ 100     │ 100
│     │                        4.646 Mitem/s │ 1.45 Mitem/s  │ 3.017 Mitem/s │ 2.881 Mitem/s │         │
│     ├─ 5                     8.848 ms      │ 27.12 ms      │ 14.68 ms      │ 15.25 ms      │ 100     │ 100
│     │                        5.65 Mitem/s  │ 1.843 Mitem/s │ 3.404 Mitem/s │ 3.276 Mitem/s │         │
│     ╰─ 6                     8.772 ms      │ 25.38 ms      │ 15.32 ms      │ 15.85 ms      │ 100     │ 100
│                              6.839 Mitem/s │ 2.363 Mitem/s │ 3.915 Mitem/s │ 3.785 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.553 ms      │ 7.541 ms      │ 4.658 ms      │ 5.061 ms      │ 100     │ 100
│  │  │                        2.196 Mitem/s │ 1.326 Mitem/s │ 2.146 Mitem/s │ 1.975 Mitem/s │         │
│  │  ├─ 2                     11.5 ms       │ 20.51 ms      │ 16.24 ms      │ 16 ms         │ 100     │ 100
│  │  │                        1.737 Mitem/s │ 975 Kitem/s   │ 1.231 Mitem/s │ 1.249 Mitem/s │         │
│  │  ├─ 3                     22.44 ms      │ 40.33 ms      │ 30.15 ms      │ 30.09 ms      │ 100     │ 100
│  │  │                        1.336 Mitem/s │ 743.8 Kitem/s │ 995 Kitem/s   │ 996.7 Kitem/s │         │
│  │  ├─ 4                     35.57 ms      │ 54.23 ms      │ 42 ms         │ 42.57 ms      │ 100     │ 100
│  │  │                        1.124 Mitem/s │ 737.4 Kitem/s │ 952.3 Kitem/s │ 939.5 Kitem/s │         │
│  │  ├─ 5                     50.65 ms      │ 70.96 ms      │ 59.84 ms      │ 60.19 ms      │ 100     │ 100
│  │  │                        987.1 Kitem/s │ 704.6 Kitem/s │ 835.5 Kitem/s │ 830.7 Kitem/s │         │
│  │  ╰─ 6                     83.35 ms      │ 113.8 ms      │ 98.82 ms      │ 98.1 ms       │ 100     │ 100
│  │                           719.8 Kitem/s │ 527 Kitem/s   │ 607.1 Kitem/s │ 611.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.497 ms      │ 17.28 ms      │ 9.851 ms      │ 10.54 ms      │ 100     │ 100
│  │  │                        1.052 Mitem/s │ 578.6 Kitem/s │ 1.015 Mitem/s │ 948.2 Kitem/s │         │
│  │  ├─ 2                     9.941 ms      │ 19.76 ms      │ 14.42 ms      │ 13.85 ms      │ 100     │ 100
│  │  │                        2.011 Mitem/s │ 1.011 Mitem/s │ 1.386 Mitem/s │ 1.443 Mitem/s │         │
│  │  ├─ 3                     10.44 ms      │ 22.36 ms      │ 15.09 ms      │ 15.35 ms      │ 100     │ 100
│  │  │                        2.873 Mitem/s │ 1.341 Mitem/s │ 1.988 Mitem/s │ 1.954 Mitem/s │         │
│  │  ├─ 4                     11.02 ms      │ 31.67 ms      │ 16.54 ms      │ 16.89 ms      │ 100     │ 100
│  │  │                        3.626 Mitem/s │ 1.262 Mitem/s │ 2.418 Mitem/s │ 2.367 Mitem/s │         │
│  │  ├─ 5                     10.96 ms      │ 31.16 ms      │ 17.79 ms      │ 18.42 ms      │ 100     │ 100
│  │  │                        4.558 Mitem/s │ 1.604 Mitem/s │ 2.809 Mitem/s │ 2.713 Mitem/s │         │
│  │  ╰─ 6                     11.57 ms      │ 30.71 ms      │ 18.09 ms      │ 18.82 ms      │ 100     │ 100
│  │                           5.182 Mitem/s │ 1.953 Mitem/s │ 3.316 Mitem/s │ 3.187 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.626 ms      │ 13.47 ms      │ 9.066 ms      │ 9.467 ms      │ 100     │ 100
│     │                        1.159 Mitem/s │ 742.3 Kitem/s │ 1.102 Mitem/s │ 1.056 Mitem/s │         │
│     ├─ 2                     8.951 ms      │ 19.46 ms      │ 10.92 ms      │ 11.72 ms      │ 100     │ 100
│     │                        2.234 Mitem/s │ 1.027 Mitem/s │ 1.83 Mitem/s  │ 1.705 Mitem/s │         │
│     ├─ 3                     9.018 ms      │ 19.38 ms      │ 13.57 ms      │ 13.61 ms      │ 100     │ 100
│     │                        3.326 Mitem/s │ 1.547 Mitem/s │ 2.21 Mitem/s  │ 2.203 Mitem/s │         │
│     ├─ 4                     9.383 ms      │ 28.7 ms       │ 13.83 ms      │ 14.68 ms      │ 100     │ 100
│     │                        4.262 Mitem/s │ 1.393 Mitem/s │ 2.891 Mitem/s │ 2.724 Mitem/s │         │
│     ├─ 5                     9.38 ms       │ 26.39 ms      │ 16.12 ms      │ 16.22 ms      │ 100     │ 100
│     │                        5.33 Mitem/s  │ 1.893 Mitem/s │ 3.1 Mitem/s   │ 3.081 Mitem/s │         │
│     ╰─ 6                     9.771 ms      │ 27.81 ms      │ 17.51 ms      │ 18.29 ms      │ 100     │ 100
│                              6.14 Mitem/s  │ 2.157 Mitem/s │ 3.425 Mitem/s │ 3.278 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.526 ms      │ 7.448 ms      │ 4.651 ms      │ 4.971 ms      │ 100     │ 100
│  │  │                        2.209 Mitem/s │ 1.342 Mitem/s │ 2.15 Mitem/s  │ 2.011 Mitem/s │         │
│  │  ├─ 2                     11.1 ms       │ 20.02 ms      │ 13.99 ms      │ 14.41 ms      │ 100     │ 100
│  │  │                        1.801 Mitem/s │ 998.9 Kitem/s │ 1.428 Mitem/s │ 1.387 Mitem/s │         │
│  │  ├─ 3                     20.94 ms      │ 35.7 ms       │ 26.97 ms      │ 27.07 ms      │ 100     │ 100
│  │  │                        1.432 Mitem/s │ 840.1 Kitem/s │ 1.111 Mitem/s │ 1.108 Mitem/s │         │
│  │  ├─ 4                     32.6 ms       │ 59.39 ms      │ 42.82 ms      │ 42.84 ms      │ 100     │ 100
│  │  │                        1.226 Mitem/s │ 673.4 Kitem/s │ 933.9 Kitem/s │ 933.5 Kitem/s │         │
│  │  ├─ 5                     50.61 ms      │ 77.24 ms      │ 60.25 ms      │ 60.33 ms      │ 100     │ 100
│  │  │                        987.8 Kitem/s │ 647.3 Kitem/s │ 829.8 Kitem/s │ 828.7 Kitem/s │         │
│  │  ╰─ 6                     82.72 ms      │ 109.4 ms      │ 96.52 ms      │ 96.25 ms      │ 100     │ 100
│  │                           725.2 Kitem/s │ 548 Kitem/s   │ 621.6 Kitem/s │ 623.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.545 ms      │ 14.45 ms      │ 9.75 ms       │ 10.85 ms      │ 100     │ 100
│  │  │                        1.047 Mitem/s │ 691.8 Kitem/s │ 1.025 Mitem/s │ 920.9 Kitem/s │         │
│  │  ├─ 2                     10.25 ms      │ 20.52 ms      │ 14.17 ms      │ 14.5 ms       │ 100     │ 100
│  │  │                        1.95 Mitem/s  │ 974.3 Kitem/s │ 1.41 Mitem/s  │ 1.378 Mitem/s │         │
│  │  ├─ 3                     10.94 ms      │ 21.14 ms      │ 15.02 ms      │ 15.79 ms      │ 100     │ 100
│  │  │                        2.742 Mitem/s │ 1.419 Mitem/s │ 1.996 Mitem/s │ 1.899 Mitem/s │         │
│  │  ├─ 4                     10.89 ms      │ 28.51 ms      │ 15.33 ms      │ 16.03 ms      │ 100     │ 100
│  │  │                        3.672 Mitem/s │ 1.402 Mitem/s │ 2.607 Mitem/s │ 2.494 Mitem/s │         │
│  │  ├─ 5                     10.72 ms      │ 29.79 ms      │ 15.9 ms       │ 17.14 ms      │ 100     │ 100
│  │  │                        4.66 Mitem/s  │ 1.678 Mitem/s │ 3.143 Mitem/s │ 2.915 Mitem/s │         │
│  │  ╰─ 6                     11.85 ms      │ 32.08 ms      │ 18.71 ms      │ 20.57 ms      │ 100     │ 100
│  │                           5.061 Mitem/s │ 1.87 Mitem/s  │ 3.205 Mitem/s │ 2.915 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.601 ms      │ 16.26 ms      │ 9.058 ms      │ 10.08 ms      │ 100     │ 100
│     │                        1.162 Mitem/s │ 614.9 Kitem/s │ 1.103 Mitem/s │ 991.3 Kitem/s │         │
│     ├─ 2                     8.824 ms      │ 19.17 ms      │ 12.26 ms      │ 12.82 ms      │ 100     │ 100
│     │                        2.266 Mitem/s │ 1.043 Mitem/s │ 1.63 Mitem/s  │ 1.559 Mitem/s │         │
│     ├─ 3                     9.378 ms      │ 21.48 ms      │ 14.33 ms      │ 14.94 ms      │ 100     │ 100
│     │                        3.198 Mitem/s │ 1.396 Mitem/s │ 2.092 Mitem/s │ 2.007 Mitem/s │         │
│     ├─ 4                     9.633 ms      │ 18.73 ms      │ 13.9 ms       │ 14.52 ms      │ 100     │ 100
│     │                        4.151 Mitem/s │ 2.135 Mitem/s │ 2.876 Mitem/s │ 2.753 Mitem/s │         │
│     ├─ 5                     10.01 ms      │ 28.12 ms      │ 16.05 ms      │ 16.93 ms      │ 100     │ 100
│     │                        4.99 Mitem/s  │ 1.778 Mitem/s │ 3.115 Mitem/s │ 2.952 Mitem/s │         │
│     ╰─ 6                     9.189 ms      │ 32.3 ms       │ 16.46 ms      │ 18.16 ms      │ 100     │ 100
│                              6.529 Mitem/s │ 1.857 Mitem/s │ 3.645 Mitem/s │ 3.302 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.522 ms      │ 8.705 ms      │ 4.71 ms       │ 5.803 ms      │ 100     │ 100
│  │  │                        2.21 Mitem/s  │ 1.148 Mitem/s │ 2.122 Mitem/s │ 1.723 Mitem/s │         │
│  │  ├─ 2                     10.71 ms      │ 20.9 ms       │ 14.33 ms      │ 14.74 ms      │ 100     │ 100
│  │  │                        1.865 Mitem/s │ 956.7 Kitem/s │ 1.395 Mitem/s │ 1.356 Mitem/s │         │
│  │  ├─ 3                     22.58 ms      │ 37.69 ms      │ 31.04 ms      │ 30.79 ms      │ 100     │ 100
│  │  │                        1.328 Mitem/s │ 795.7 Kitem/s │ 966.4 Kitem/s │ 974 Kitem/s   │         │
│  │  ├─ 4                     34.28 ms      │ 55.14 ms      │ 42.02 ms      │ 42.3 ms       │ 100     │ 100
│  │  │                        1.166 Mitem/s │ 725.4 Kitem/s │ 951.8 Kitem/s │ 945.6 Kitem/s │         │
│  │  ├─ 5                     50.75 ms      │ 90.31 ms      │ 58.11 ms      │ 60.13 ms      │ 100     │ 100
│  │  │                        985.1 Kitem/s │ 553.6 Kitem/s │ 860.4 Kitem/s │ 831.4 Kitem/s │         │
│  │  ╰─ 6                     79.86 ms      │ 112.8 ms      │ 95.28 ms      │ 95.2 ms       │ 100     │ 100
│  │                           751.3 Kitem/s │ 531.8 Kitem/s │ 629.6 Kitem/s │ 630.2 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.462 ms      │ 14.48 ms      │ 9.678 ms      │ 10.52 ms      │ 100     │ 100
│  │  │                        1.056 Mitem/s │ 690.5 Kitem/s │ 1.033 Mitem/s │ 949.9 Kitem/s │         │
│  │  ├─ 2                     9.868 ms      │ 20.92 ms      │ 13.09 ms      │ 13.24 ms      │ 100     │ 100
│  │  │                        2.026 Mitem/s │ 956 Kitem/s   │ 1.527 Mitem/s │ 1.509 Mitem/s │         │
│  │  ├─ 3                     11.95 ms      │ 20.62 ms      │ 14.94 ms      │ 15.71 ms      │ 100     │ 100
│  │  │                        2.509 Mitem/s │ 1.454 Mitem/s │ 2.007 Mitem/s │ 1.909 Mitem/s │         │
│  │  ├─ 4                     10.8 ms       │ 25.89 ms      │ 15.46 ms      │ 16.11 ms      │ 100     │ 100
│  │  │                        3.702 Mitem/s │ 1.544 Mitem/s │ 2.586 Mitem/s │ 2.481 Mitem/s │         │
│  │  ├─ 5                     12.77 ms      │ 31.98 ms      │ 17.9 ms       │ 18.71 ms      │ 100     │ 100
│  │  │                        3.914 Mitem/s │ 1.563 Mitem/s │ 2.793 Mitem/s │ 2.671 Mitem/s │         │
│  │  ╰─ 6                     14.33 ms      │ 30.74 ms      │ 18.13 ms      │ 18.88 ms      │ 100     │ 100
│  │                           4.186 Mitem/s │ 1.951 Mitem/s │ 3.308 Mitem/s │ 3.177 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.59 ms       │ 14.29 ms      │ 8.992 ms      │ 9.762 ms      │ 100     │ 100
│     │                        1.164 Mitem/s │ 699.4 Kitem/s │ 1.111 Mitem/s │ 1.024 Mitem/s │         │
│     ├─ 2                     8.857 ms      │ 19.56 ms      │ 13.16 ms      │ 13.61 ms      │ 100     │ 100
│     │                        2.257 Mitem/s │ 1.022 Mitem/s │ 1.518 Mitem/s │ 1.468 Mitem/s │         │
│     ├─ 3                     10.47 ms      │ 20.06 ms      │ 16.05 ms      │ 15.73 ms      │ 100     │ 100
│     │                        2.863 Mitem/s │ 1.495 Mitem/s │ 1.868 Mitem/s │ 1.906 Mitem/s │         │
│     ├─ 4                     9.127 ms      │ 27.19 ms      │ 16.12 ms      │ 16.17 ms      │ 100     │ 100
│     │                        4.382 Mitem/s │ 1.471 Mitem/s │ 2.48 Mitem/s  │ 2.472 Mitem/s │         │
│     ├─ 5                     9.602 ms      │ 28.95 ms      │ 15.9 ms       │ 16.35 ms      │ 100     │ 100
│     │                        5.207 Mitem/s │ 1.727 Mitem/s │ 3.144 Mitem/s │ 3.056 Mitem/s │         │
│     ╰─ 6                     9.23 ms       │ 28.09 ms      │ 16.42 ms      │ 17.52 ms      │ 100     │ 100
│                              6.499 Mitem/s │ 2.135 Mitem/s │ 3.652 Mitem/s │ 3.423 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.161 ms      │ 6.809 ms      │ 4.254 ms      │ 4.422 ms      │ 100     │ 100
│  │  │                        2.402 Mitem/s │ 1.468 Mitem/s │ 2.35 Mitem/s  │ 2.261 Mitem/s │         │
│  │  ├─ 2                     9.492 ms      │ 19.11 ms      │ 13.09 ms      │ 13.49 ms      │ 100     │ 100
│  │  │                        2.106 Mitem/s │ 1.046 Mitem/s │ 1.527 Mitem/s │ 1.481 Mitem/s │         │
│  │  ├─ 3                     19.35 ms      │ 36.62 ms      │ 25.88 ms      │ 25.59 ms      │ 100     │ 100
│  │  │                        1.549 Mitem/s │ 819.1 Kitem/s │ 1.159 Mitem/s │ 1.172 Mitem/s │         │
│  │  ├─ 4                     29.84 ms      │ 51.66 ms      │ 38.97 ms      │ 39.29 ms      │ 100     │ 100
│  │  │                        1.34 Mitem/s  │ 774.1 Kitem/s │ 1.026 Mitem/s │ 1.017 Mitem/s │         │
│  │  ├─ 5                     45.73 ms      │ 71.39 ms      │ 53.84 ms      │ 54.25 ms      │ 100     │ 100
│  │  │                        1.093 Mitem/s │ 700.3 Kitem/s │ 928.6 Kitem/s │ 921.6 Kitem/s │         │
│  │  ╰─ 6                     75.43 ms      │ 101.7 ms      │ 85.9 ms       │ 85.41 ms      │ 100     │ 100
│  │                           795.4 Kitem/s │ 589.5 Kitem/s │ 698.4 Kitem/s │ 702.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     13.86 ms      │ 17.43 ms      │ 14.22 ms      │ 14.34 ms      │ 100     │ 100
│  │  │                        721.1 Kitem/s │ 573.4 Kitem/s │ 702.7 Kitem/s │ 697.1 Kitem/s │         │
│  │  ├─ 2                     14.16 ms      │ 26.28 ms      │ 15.98 ms      │ 16.63 ms      │ 100     │ 100
│  │  │                        1.411 Mitem/s │ 760.8 Kitem/s │ 1.25 Mitem/s  │ 1.202 Mitem/s │         │
│  │  ├─ 3                     15.2 ms       │ 30.75 ms      │ 22.03 ms      │ 22.81 ms      │ 100     │ 100
│  │  │                        1.973 Mitem/s │ 975.4 Kitem/s │ 1.361 Mitem/s │ 1.314 Mitem/s │         │
│  │  ├─ 4                     15.39 ms      │ 39.92 ms      │ 22.96 ms      │ 23.27 ms      │ 100     │ 100
│  │  │                        2.598 Mitem/s │ 1.001 Mitem/s │ 1.741 Mitem/s │ 1.718 Mitem/s │         │
│  │  ├─ 5                     16.16 ms      │ 37.52 ms      │ 24.56 ms      │ 24.5 ms       │ 100     │ 100
│  │  │                        3.093 Mitem/s │ 1.332 Mitem/s │ 2.035 Mitem/s │ 2.04 Mitem/s  │         │
│  │  ╰─ 6                     19.67 ms      │ 41.58 ms      │ 25.42 ms      │ 27.38 ms      │ 100     │ 100
│  │                           3.05 Mitem/s  │ 1.442 Mitem/s │ 2.359 Mitem/s │ 2.191 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.763 ms      │ 12.74 ms      │ 9.183 ms      │ 10.14 ms      │ 100     │ 100
│     │                        1.141 Mitem/s │ 784.6 Kitem/s │ 1.088 Mitem/s │ 986.1 Kitem/s │         │
│     ├─ 2                     8.906 ms      │ 18.68 ms      │ 12.55 ms      │ 12.43 ms      │ 100     │ 100
│     │                        2.245 Mitem/s │ 1.07 Mitem/s  │ 1.593 Mitem/s │ 1.608 Mitem/s │         │
│     ├─ 3                     9.045 ms      │ 22.75 ms      │ 13.64 ms      │ 14.26 ms      │ 100     │ 100
│     │                        3.316 Mitem/s │ 1.318 Mitem/s │ 2.199 Mitem/s │ 2.103 Mitem/s │         │
│     ├─ 4                     9.093 ms      │ 30.21 ms      │ 15.27 ms      │ 15.38 ms      │ 100     │ 100
│     │                        4.398 Mitem/s │ 1.323 Mitem/s │ 2.618 Mitem/s │ 2.6 Mitem/s   │         │
│     ├─ 5                     9.429 ms      │ 27.89 ms      │ 16.69 ms      │ 17.11 ms      │ 100     │ 100
│     │                        5.302 Mitem/s │ 1.792 Mitem/s │ 2.995 Mitem/s │ 2.921 Mitem/s │         │
│     ╰─ 6                     11.02 ms      │ 29.97 ms      │ 18.66 ms      │ 19.98 ms      │ 100     │ 100
│                              5.44 Mitem/s  │ 2.001 Mitem/s │ 3.214 Mitem/s │ 3.001 Mitem/s │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 2                     7.055 ms      │ 12 ms         │ 10.57 ms      │ 10.46 ms      │ 100     │ 100
│  │  ├─ 4                     12.18 ms      │ 23.93 ms      │ 18.14 ms      │ 16.85 ms      │ 100     │ 100
│  │  ├─ 8                     18.15 ms      │ 31 ms         │ 20.45 ms      │ 23.43 ms      │ 100     │ 100
│  │  ├─ 16                    25.65 ms      │ 38.58 ms      │ 29.61 ms      │ 30.06 ms      │ 100     │ 100
│  │  ╰─ 32                    51.58 ms      │ 62.4 ms       │ 55.46 ms      │ 55.39 ms      │ 100     │ 100
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 2                     12.94 ms      │ 28.25 ms      │ 14.96 ms      │ 16.27 ms      │ 100     │ 100
│     ├─ 4                     12.44 ms      │ 23.02 ms      │ 16.84 ms      │ 16.78 ms      │ 100     │ 100
│     ├─ 8                     16.58 ms      │ 29.05 ms      │ 18.64 ms      │ 20.43 ms      │ 100     │ 100
│     ├─ 16                    24.76 ms      │ 35.08 ms      │ 27.97 ms      │ 27.81 ms      │ 100     │ 100
│     ╰─ 32                    46.57 ms      │ 58.83 ms      │ 50.9 ms       │ 51.19 ms      │ 100     │ 100
```
