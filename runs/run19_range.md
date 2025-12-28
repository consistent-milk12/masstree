```text
Timer precision: 30 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.489 ms      │ 7.511 ms      │ 4.69 ms       │ 5.087 ms      │ 100     │ 100
│  │  │                        2.227 Mitem/s │ 1.331 Mitem/s │ 2.131 Mitem/s │ 1.965 Mitem/s │         │
│  │  ├─ 2                     10.93 ms      │ 21.66 ms      │ 15.69 ms      │ 15.58 ms      │ 100     │ 100
│  │  │                        1.829 Mitem/s │ 923.1 Kitem/s │ 1.273 Mitem/s │ 1.283 Mitem/s │         │
│  │  ├─ 3                     20.21 ms      │ 36.34 ms      │ 25.87 ms      │ 26.66 ms      │ 100     │ 100
│  │  │                        1.484 Mitem/s │ 825.4 Kitem/s │ 1.159 Mitem/s │ 1.125 Mitem/s │         │
│  │  ├─ 4                     35.22 ms      │ 57.12 ms      │ 42.09 ms      │ 42.77 ms      │ 100     │ 100
│  │  │                        1.135 Mitem/s │ 700.2 Kitem/s │ 950.1 Kitem/s │ 935 Kitem/s   │         │
│  │  ├─ 5                     51.05 ms      │ 72.97 ms      │ 61.14 ms      │ 61.79 ms      │ 100     │ 100
│  │  │                        979.3 Kitem/s │ 685.2 Kitem/s │ 817.6 Kitem/s │ 809.1 Kitem/s │         │
│  │  ╰─ 6                     82.37 ms      │ 106.7 ms      │ 95.23 ms      │ 95.34 ms      │ 100     │ 100
│  │                           728.3 Kitem/s │ 562.1 Kitem/s │ 629.9 Kitem/s │ 629.3 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.717 ms      │ 15.16 ms      │ 8.005 ms      │ 9.218 ms      │ 100     │ 100
│  │  │                        1.295 Mitem/s │ 659.2 Kitem/s │ 1.249 Mitem/s │ 1.084 Mitem/s │         │
│  │  ├─ 2                     8.143 ms      │ 16.91 ms      │ 12.33 ms      │ 12.57 ms      │ 100     │ 100
│  │  │                        2.455 Mitem/s │ 1.182 Mitem/s │ 1.62 Mitem/s  │ 1.59 Mitem/s  │         │
│  │  ├─ 3                     8.848 ms      │ 21.68 ms      │ 12.63 ms      │ 13.11 ms      │ 100     │ 100
│  │  │                        3.39 Mitem/s  │ 1.383 Mitem/s │ 2.374 Mitem/s │ 2.287 Mitem/s │         │
│  │  ├─ 4                     9.035 ms      │ 23.77 ms      │ 13.83 ms      │ 13.99 ms      │ 100     │ 100
│  │  │                        4.427 Mitem/s │ 1.682 Mitem/s │ 2.89 Mitem/s  │ 2.857 Mitem/s │         │
│  │  ├─ 5                     9.362 ms      │ 27.65 ms      │ 14.63 ms      │ 15.26 ms      │ 100     │ 100
│  │  │                        5.34 Mitem/s  │ 1.808 Mitem/s │ 3.416 Mitem/s │ 3.275 Mitem/s │         │
│  │  ╰─ 6                     9.788 ms      │ 25.94 ms      │ 14.69 ms      │ 15.82 ms      │ 100     │ 100
│  │                           6.129 Mitem/s │ 2.312 Mitem/s │ 4.084 Mitem/s │ 3.79 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.265 ms      │ 16.1 ms       │ 8.663 ms      │ 9.89 ms       │ 100     │ 100
│     │                        1.209 Mitem/s │ 620.9 Kitem/s │ 1.154 Mitem/s │ 1.011 Mitem/s │         │
│     ├─ 2                     8.621 ms      │ 21.62 ms      │ 11.63 ms      │ 11.74 ms      │ 100     │ 100
│     │                        2.319 Mitem/s │ 924.6 Kitem/s │ 1.718 Mitem/s │ 1.702 Mitem/s │         │
│     ├─ 3                     8.628 ms      │ 19.11 ms      │ 13.04 ms      │ 12.92 ms      │ 100     │ 100
│     │                        3.476 Mitem/s │ 1.569 Mitem/s │ 2.3 Mitem/s   │ 2.321 Mitem/s │         │
│     ├─ 4                     8.762 ms      │ 22.57 ms      │ 15.33 ms      │ 14.78 ms      │ 100     │ 100
│     │                        4.564 Mitem/s │ 1.771 Mitem/s │ 2.608 Mitem/s │ 2.705 Mitem/s │         │
│     ├─ 5                     9.625 ms      │ 30.28 ms      │ 16.26 ms      │ 17.48 ms      │ 100     │ 100
│     │                        5.194 Mitem/s │ 1.65 Mitem/s  │ 3.073 Mitem/s │ 2.858 Mitem/s │         │
│     ╰─ 6                     9.856 ms      │ 28.76 ms      │ 17.1 ms       │ 18.08 ms      │ 100     │ 100
│                              6.087 Mitem/s │ 2.085 Mitem/s │ 3.508 Mitem/s │ 3.317 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.74 ms       │ 6.734 ms      │ 4.788 ms      │ 4.914 ms      │ 100     │ 100
│  │  │                        2.109 Mitem/s │ 1.484 Mitem/s │ 2.088 Mitem/s │ 2.034 Mitem/s │         │
│  │  ├─ 2                     11.54 ms      │ 22.87 ms      │ 15.43 ms      │ 15.38 ms      │ 100     │ 100
│  │  │                        1.731 Mitem/s │ 874.2 Kitem/s │ 1.296 Mitem/s │ 1.299 Mitem/s │         │
│  │  ├─ 3                     22.61 ms      │ 39.89 ms      │ 29.45 ms      │ 29.69 ms      │ 100     │ 100
│  │  │                        1.326 Mitem/s │ 751.9 Kitem/s │ 1.018 Mitem/s │ 1.01 Mitem/s  │         │
│  │  ├─ 4                     35 ms         │ 59.29 ms      │ 42.4 ms       │ 42.94 ms      │ 100     │ 100
│  │  │                        1.142 Mitem/s │ 674.6 Kitem/s │ 943.2 Kitem/s │ 931.3 Kitem/s │         │
│  │  ├─ 5                     53.49 ms      │ 76.86 ms      │ 62.46 ms      │ 62.66 ms      │ 100     │ 100
│  │  │                        934.6 Kitem/s │ 650.4 Kitem/s │ 800.4 Kitem/s │ 797.8 Kitem/s │         │
│  │  ╰─ 6                     78.02 ms      │ 114.1 ms      │ 98.47 ms      │ 97.79 ms      │ 100     │ 100
│  │                           768.9 Kitem/s │ 525.6 Kitem/s │ 609.2 Kitem/s │ 613.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.686 ms      │ 10.03 ms      │ 7.783 ms      │ 7.943 ms      │ 100     │ 100
│  │  │                        1.3 Mitem/s   │ 996.4 Kitem/s │ 1.284 Mitem/s │ 1.258 Mitem/s │         │
│  │  ├─ 2                     8.006 ms      │ 17.98 ms      │ 14.48 ms      │ 13.36 ms      │ 100     │ 100
│  │  │                        2.497 Mitem/s │ 1.112 Mitem/s │ 1.38 Mitem/s  │ 1.496 Mitem/s │         │
│  │  ├─ 3                     8.227 ms      │ 17.5 ms       │ 12.47 ms      │ 13.26 ms      │ 100     │ 100
│  │  │                        3.646 Mitem/s │ 1.714 Mitem/s │ 2.404 Mitem/s │ 2.261 Mitem/s │         │
│  │  ├─ 4                     8.877 ms      │ 21.5 ms       │ 12.57 ms      │ 12.71 ms      │ 100     │ 100
│  │  │                        4.505 Mitem/s │ 1.86 Mitem/s  │ 3.18 Mitem/s  │ 3.146 Mitem/s │         │
│  │  ├─ 5                     8.886 ms      │ 22.75 ms      │ 14.14 ms      │ 14.34 ms      │ 100     │ 100
│  │  │                        5.626 Mitem/s │ 2.197 Mitem/s │ 3.535 Mitem/s │ 3.486 Mitem/s │         │
│  │  ╰─ 6                     8.999 ms      │ 24.46 ms      │ 14.4 ms       │ 14.7 ms       │ 100     │ 100
│  │                           6.667 Mitem/s │ 2.452 Mitem/s │ 4.166 Mitem/s │ 4.08 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.746 ms      │ 17.01 ms      │ 9.204 ms      │ 10.33 ms      │ 100     │ 100
│     │                        1.143 Mitem/s │ 587.6 Kitem/s │ 1.086 Mitem/s │ 967.5 Kitem/s │         │
│     ├─ 2                     8.952 ms      │ 21.32 ms      │ 11.83 ms      │ 12.05 ms      │ 100     │ 100
│     │                        2.233 Mitem/s │ 938 Kitem/s   │ 1.689 Mitem/s │ 1.659 Mitem/s │         │
│     ├─ 3                     9.072 ms      │ 21.44 ms      │ 13.88 ms      │ 14.02 ms      │ 100     │ 100
│     │                        3.306 Mitem/s │ 1.398 Mitem/s │ 2.161 Mitem/s │ 2.139 Mitem/s │         │
│     ├─ 4                     9.241 ms      │ 30.26 ms      │ 16.4 ms       │ 16.03 ms      │ 100     │ 100
│     │                        4.328 Mitem/s │ 1.321 Mitem/s │ 2.438 Mitem/s │ 2.494 Mitem/s │         │
│     ├─ 5                     9.426 ms      │ 27.43 ms      │ 16.54 ms      │ 16.17 ms      │ 100     │ 100
│     │                        5.303 Mitem/s │ 1.822 Mitem/s │ 3.021 Mitem/s │ 3.09 Mitem/s  │         │
│     ╰─ 6                     11.76 ms      │ 28.87 ms      │ 17.03 ms      │ 18.35 ms      │ 100     │ 100
│                              5.101 Mitem/s │ 2.078 Mitem/s │ 3.523 Mitem/s │ 3.269 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.485 ms      │ 7.224 ms      │ 4.596 ms      │ 4.822 ms      │ 100     │ 100
│  │  │                        2.229 Mitem/s │ 1.384 Mitem/s │ 2.175 Mitem/s │ 2.073 Mitem/s │         │
│  │  ├─ 2                     11.01 ms      │ 19.92 ms      │ 15.66 ms      │ 15.2 ms       │ 100     │ 100
│  │  │                        1.815 Mitem/s │ 1.003 Mitem/s │ 1.276 Mitem/s │ 1.315 Mitem/s │         │
│  │  ├─ 3                     21.77 ms      │ 38.14 ms      │ 28.88 ms      │ 29.36 ms      │ 100     │ 100
│  │  │                        1.377 Mitem/s │ 786.4 Kitem/s │ 1.038 Mitem/s │ 1.021 Mitem/s │         │
│  │  ├─ 4                     33.19 ms      │ 62.09 ms      │ 42.86 ms      │ 43.37 ms      │ 100     │ 100
│  │  │                        1.205 Mitem/s │ 644.1 Kitem/s │ 933.1 Kitem/s │ 922.1 Kitem/s │         │
│  │  ├─ 5                     52.14 ms      │ 75.78 ms      │ 61.73 ms      │ 62.38 ms      │ 100     │ 100
│  │  │                        958.8 Kitem/s │ 659.7 Kitem/s │ 809.8 Kitem/s │ 801.4 Kitem/s │         │
│  │  ╰─ 6                     82.33 ms      │ 119.7 ms      │ 94.22 ms      │ 94.35 ms      │ 100     │ 100
│  │                           728.7 Kitem/s │ 500.9 Kitem/s │ 636.7 Kitem/s │ 635.8 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.738 ms      │ 10.01 ms      │ 7.885 ms      │ 8.04 ms       │ 100     │ 100
│  │  │                        1.292 Mitem/s │ 998.8 Kitem/s │ 1.268 Mitem/s │ 1.243 Mitem/s │         │
│  │  ├─ 2                     8.151 ms      │ 17.31 ms      │ 12.41 ms      │ 13.23 ms      │ 100     │ 100
│  │  │                        2.453 Mitem/s │ 1.155 Mitem/s │ 1.61 Mitem/s  │ 1.51 Mitem/s  │         │
│  │  ├─ 3                     8.31 ms       │ 19.07 ms      │ 13.18 ms      │ 13.49 ms      │ 100     │ 100
│  │  │                        3.609 Mitem/s │ 1.573 Mitem/s │ 2.275 Mitem/s │ 2.222 Mitem/s │         │
│  │  ├─ 4                     8.937 ms      │ 24.55 ms      │ 13.96 ms      │ 13.69 ms      │ 100     │ 100
│  │  │                        4.475 Mitem/s │ 1.629 Mitem/s │ 2.865 Mitem/s │ 2.92 Mitem/s  │         │
│  │  ├─ 5                     10.89 ms      │ 27.74 ms      │ 14.63 ms      │ 15.1 ms       │ 100     │ 100
│  │  │                        4.59 Mitem/s  │ 1.801 Mitem/s │ 3.416 Mitem/s │ 3.31 Mitem/s  │         │
│  │  ╰─ 6                     9.245 ms      │ 25.24 ms      │ 14.89 ms      │ 16.03 ms      │ 100     │ 100
│  │                           6.489 Mitem/s │ 2.377 Mitem/s │ 4.029 Mitem/s │ 3.742 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.459 ms      │ 14.85 ms      │ 9.041 ms      │ 9.851 ms      │ 100     │ 100
│     │                        1.182 Mitem/s │ 673.2 Kitem/s │ 1.105 Mitem/s │ 1.015 Mitem/s │         │
│     ├─ 2                     8.609 ms      │ 17.69 ms      │ 12.15 ms      │ 11.8 ms       │ 100     │ 100
│     │                        2.323 Mitem/s │ 1.13 Mitem/s  │ 1.645 Mitem/s │ 1.693 Mitem/s │         │
│     ├─ 3                     8.684 ms      │ 19.95 ms      │ 13.08 ms      │ 13.34 ms      │ 100     │ 100
│     │                        3.454 Mitem/s │ 1.503 Mitem/s │ 2.292 Mitem/s │ 2.247 Mitem/s │         │
│     ├─ 4                     8.682 ms      │ 29.1 ms       │ 13.86 ms      │ 14.49 ms      │ 100     │ 100
│     │                        4.606 Mitem/s │ 1.374 Mitem/s │ 2.884 Mitem/s │ 2.758 Mitem/s │         │
│     ├─ 5                     8.986 ms      │ 26.79 ms      │ 15.31 ms      │ 15.44 ms      │ 100     │ 100
│     │                        5.563 Mitem/s │ 1.865 Mitem/s │ 3.265 Mitem/s │ 3.238 Mitem/s │         │
│     ╰─ 6                     10.3 ms       │ 30.09 ms      │ 16.12 ms      │ 17.68 ms      │ 100     │ 100
│                              5.823 Mitem/s │ 1.993 Mitem/s │ 3.72 Mitem/s  │ 3.391 Mitem/s │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.517 ms      │ 8.896 ms      │ 4.648 ms      │ 5.379 ms      │ 100     │ 100
│  │  │                        2.213 Mitem/s │ 1.124 Mitem/s │ 2.151 Mitem/s │ 1.859 Mitem/s │         │
│  │  ├─ 2                     12.14 ms      │ 20.61 ms      │ 16.28 ms      │ 16.02 ms      │ 100     │ 100
│  │  │                        1.646 Mitem/s │ 970.3 Kitem/s │ 1.227 Mitem/s │ 1.247 Mitem/s │         │
│  │  ├─ 3                     22.6 ms       │ 36.16 ms      │ 30.91 ms      │ 30.28 ms      │ 100     │ 100
│  │  │                        1.327 Mitem/s │ 829.5 Kitem/s │ 970.5 Kitem/s │ 990.7 Kitem/s │         │
│  │  ├─ 4                     33.17 ms      │ 55 ms         │ 43.83 ms      │ 43.93 ms      │ 100     │ 100
│  │  │                        1.205 Mitem/s │ 727.2 Kitem/s │ 912.4 Kitem/s │ 910.3 Kitem/s │         │
│  │  ├─ 5                     50.87 ms      │ 82.63 ms      │ 60.23 ms      │ 60.7 ms       │ 100     │ 100
│  │  │                        982.8 Kitem/s │ 605 Kitem/s   │ 830 Kitem/s   │ 823.7 Kitem/s │         │
│  │  ╰─ 6                     80.26 ms      │ 110.2 ms      │ 97.28 ms      │ 96.35 ms      │ 100     │ 100
│  │                           747.5 Kitem/s │ 544 Kitem/s   │ 616.7 Kitem/s │ 622.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.827 ms      │ 11.06 ms      │ 7.983 ms      │ 8.193 ms      │ 100     │ 100
│  │  │                        1.277 Mitem/s │ 903.8 Kitem/s │ 1.252 Mitem/s │ 1.22 Mitem/s  │         │
│  │  ├─ 2                     8.139 ms      │ 16.11 ms      │ 10.21 ms      │ 10.72 ms      │ 100     │ 100
│  │  │                        2.457 Mitem/s │ 1.241 Mitem/s │ 1.957 Mitem/s │ 1.864 Mitem/s │         │
│  │  ├─ 3                     9.087 ms      │ 17.16 ms      │ 12.61 ms      │ 13.27 ms      │ 100     │ 100
│  │  │                        3.301 Mitem/s │ 1.747 Mitem/s │ 2.379 Mitem/s │ 2.259 Mitem/s │         │
│  │  ├─ 4                     9.205 ms      │ 26.18 ms      │ 14.76 ms      │ 14.72 ms      │ 100     │ 100
│  │  │                        4.345 Mitem/s │ 1.527 Mitem/s │ 2.708 Mitem/s │ 2.716 Mitem/s │         │
│  │  ├─ 5                     9.308 ms      │ 25.4 ms       │ 14.66 ms      │ 15.38 ms      │ 100     │ 100
│  │  │                        5.371 Mitem/s │ 1.968 Mitem/s │ 3.41 Mitem/s  │ 3.249 Mitem/s │         │
│  │  ╰─ 6                     9.073 ms      │ 25.15 ms      │ 14.82 ms      │ 16.02 ms      │ 100     │ 100
│  │                           6.612 Mitem/s │ 2.385 Mitem/s │ 4.046 Mitem/s │ 3.745 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.361 ms      │ 11.84 ms      │ 8.681 ms      │ 9.032 ms      │ 100     │ 100
│     │                        1.195 Mitem/s │ 844.3 Kitem/s │ 1.151 Mitem/s │ 1.107 Mitem/s │         │
│     ├─ 2                     8.537 ms      │ 18.26 ms      │ 9.757 ms      │ 10.57 ms      │ 100     │ 100
│     │                        2.342 Mitem/s │ 1.095 Mitem/s │ 2.049 Mitem/s │ 1.89 Mitem/s  │         │
│     ├─ 3                     8.83 ms       │ 18.7 ms       │ 14.72 ms      │ 14.07 ms      │ 100     │ 100
│     │                        3.397 Mitem/s │ 1.604 Mitem/s │ 2.037 Mitem/s │ 2.131 Mitem/s │         │
│     ├─ 4                     8.758 ms      │ 23.42 ms      │ 14.7 ms       │ 14.36 ms      │ 100     │ 100
│     │                        4.567 Mitem/s │ 1.707 Mitem/s │ 2.719 Mitem/s │ 2.784 Mitem/s │         │
│     ├─ 5                     8.767 ms      │ 23.96 ms      │ 15.66 ms      │ 15.29 ms      │ 100     │ 100
│     │                        5.702 Mitem/s │ 2.086 Mitem/s │ 3.191 Mitem/s │ 3.268 Mitem/s │         │
│     ╰─ 6                     9.138 ms      │ 27.28 ms      │ 15.89 ms      │ 16.36 ms      │ 100     │ 100
│                              6.565 Mitem/s │ 2.198 Mitem/s │ 3.775 Mitem/s │ 3.665 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.157 ms      │ 5.335 ms      │ 3.228 ms      │ 3.363 ms      │ 100     │ 100
│  │  │                        3.166 Mitem/s │ 1.874 Mitem/s │ 3.097 Mitem/s │ 2.972 Mitem/s │         │
│  │  ├─ 2                     7.964 ms      │ 14.04 ms      │ 10.79 ms      │ 10.95 ms      │ 100     │ 100
│  │  │                        2.511 Mitem/s │ 1.424 Mitem/s │ 1.852 Mitem/s │ 1.825 Mitem/s │         │
│  │  ├─ 3                     15.83 ms      │ 25.23 ms      │ 20.63 ms      │ 20.82 ms      │ 100     │ 100
│  │  │                        1.894 Mitem/s │ 1.188 Mitem/s │ 1.453 Mitem/s │ 1.44 Mitem/s  │         │
│  │  ├─ 4                     24.82 ms      │ 40.59 ms      │ 30.27 ms      │ 30.94 ms      │ 100     │ 100
│  │  │                        1.611 Mitem/s │ 985.3 Kitem/s │ 1.321 Mitem/s │ 1.292 Mitem/s │         │
│  │  ├─ 5                     37.14 ms      │ 63.93 ms      │ 45.99 ms      │ 46.26 ms      │ 100     │ 100
│  │  │                        1.346 Mitem/s │ 782 Kitem/s   │ 1.087 Mitem/s │ 1.08 Mitem/s  │         │
│  │  ╰─ 6                     60.36 ms      │ 90.15 ms      │ 70.7 ms       │ 70.71 ms      │ 100     │ 100
│  │                           993.9 Kitem/s │ 665.5 Kitem/s │ 848.6 Kitem/s │ 848.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.044 ms      │ 9.853 ms      │ 7.228 ms      │ 7.48 ms       │ 100     │ 100
│  │  │                        1.419 Mitem/s │ 1.014 Mitem/s │ 1.383 Mitem/s │ 1.336 Mitem/s │         │
│  │  ├─ 2                     7.516 ms      │ 15.99 ms      │ 10.81 ms      │ 11.14 ms      │ 100     │ 100
│  │  │                        2.66 Mitem/s  │ 1.25 Mitem/s  │ 1.849 Mitem/s │ 1.794 Mitem/s │         │
│  │  ├─ 3                     8.214 ms      │ 16.62 ms      │ 12.51 ms      │ 12.17 ms      │ 100     │ 100
│  │  │                        3.652 Mitem/s │ 1.804 Mitem/s │ 2.396 Mitem/s │ 2.464 Mitem/s │         │
│  │  ├─ 4                     8.259 ms      │ 21.23 ms      │ 13.45 ms      │ 12.98 ms      │ 100     │ 100
│  │  │                        4.842 Mitem/s │ 1.883 Mitem/s │ 2.973 Mitem/s │ 3.08 Mitem/s  │         │
│  │  ├─ 5                     8.401 ms      │ 24.28 ms      │ 13.86 ms      │ 14.51 ms      │ 100     │ 100
│  │  │                        5.951 Mitem/s │ 2.059 Mitem/s │ 3.605 Mitem/s │ 3.443 Mitem/s │         │
│  │  ╰─ 6                     8.357 ms      │ 23.28 ms      │ 13.47 ms      │ 13.75 ms      │ 100     │ 100
│  │                           7.179 Mitem/s │ 2.576 Mitem/s │ 4.451 Mitem/s │ 4.363 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.305 ms      │ 10.76 ms      │ 8.55 ms       │ 8.703 ms      │ 100     │ 100
│     │                        1.203 Mitem/s │ 928.7 Kitem/s │ 1.169 Mitem/s │ 1.148 Mitem/s │         │
│     ├─ 2                     8.523 ms      │ 17.94 ms      │ 11.71 ms      │ 11.76 ms      │ 100     │ 100
│     │                        2.346 Mitem/s │ 1.114 Mitem/s │ 1.706 Mitem/s │ 1.7 Mitem/s   │         │
│     ├─ 3                     8.552 ms      │ 18.02 ms      │ 12.96 ms      │ 12.76 ms      │ 100     │ 100
│     │                        3.507 Mitem/s │ 1.664 Mitem/s │ 2.313 Mitem/s │ 2.35 Mitem/s  │         │
│     ├─ 4                     8.77 ms       │ 27.69 ms      │ 15.33 ms      │ 15.05 ms      │ 100     │ 100
│     │                        4.56 Mitem/s  │ 1.444 Mitem/s │ 2.608 Mitem/s │ 2.656 Mitem/s │         │
│     ├─ 5                     8.836 ms      │ 26.27 ms      │ 15.23 ms      │ 15.91 ms      │ 100     │ 100
│     │                        5.658 Mitem/s │ 1.903 Mitem/s │ 3.281 Mitem/s │ 3.141 Mitem/s │         │
│     ╰─ 6                     8.863 ms      │ 26.92 ms      │ 15.23 ms      │ 16.42 ms      │ 100     │ 100
│                              6.769 Mitem/s │ 2.228 Mitem/s │ 3.938 Mitem/s │ 3.651 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.313 ms      │ 6.24 ms       │ 3.43 ms       │ 3.913 ms      │ 100     │ 100
│  │  │                        3.017 Mitem/s │ 1.602 Mitem/s │ 2.914 Mitem/s │ 2.555 Mitem/s │         │
│  │  ├─ 2                     7.397 ms      │ 15.28 ms      │ 10.55 ms      │ 10.42 ms      │ 100     │ 100
│  │  │                        2.703 Mitem/s │ 1.308 Mitem/s │ 1.894 Mitem/s │ 1.919 Mitem/s │         │
│  │  ├─ 3                     15.91 ms      │ 27.33 ms      │ 21.23 ms      │ 21.28 ms      │ 100     │ 100
│  │  │                        1.884 Mitem/s │ 1.097 Mitem/s │ 1.412 Mitem/s │ 1.409 Mitem/s │         │
│  │  ├─ 4                     26.87 ms      │ 41.71 ms      │ 32.59 ms      │ 32.7 ms       │ 100     │ 100
│  │  │                        1.488 Mitem/s │ 958.9 Kitem/s │ 1.227 Mitem/s │ 1.223 Mitem/s │         │
│  │  ├─ 5                     36.33 ms      │ 57.94 ms      │ 46.02 ms      │ 46.2 ms       │ 100     │ 100
│  │  │                        1.376 Mitem/s │ 862.9 Kitem/s │ 1.086 Mitem/s │ 1.082 Mitem/s │         │
│  │  ╰─ 6                     61.29 ms      │ 83.65 ms      │ 74.25 ms      │ 73.96 ms      │ 100     │ 100
│  │                           978.9 Kitem/s │ 717.2 Kitem/s │ 808 Kitem/s   │ 811.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     9.838 ms      │ 12.24 ms      │ 10.11 ms      │ 10.31 ms      │ 100     │ 100
│  │  │                        1.016 Mitem/s │ 816.9 Kitem/s │ 988.9 Kitem/s │ 969.7 Kitem/s │         │
│  │  ├─ 2                     10.09 ms      │ 19.22 ms      │ 12.1 ms       │ 12.84 ms      │ 100     │ 100
│  │  │                        1.981 Mitem/s │ 1.04 Mitem/s  │ 1.651 Mitem/s │ 1.557 Mitem/s │         │
│  │  ├─ 3                     11.56 ms      │ 26.96 ms      │ 15.29 ms      │ 15.78 ms      │ 100     │ 100
│  │  │                        2.595 Mitem/s │ 1.112 Mitem/s │ 1.961 Mitem/s │ 1.9 Mitem/s   │         │
│  │  ├─ 4                     10.27 ms      │ 30.36 ms      │ 15.5 ms       │ 16.14 ms      │ 100     │ 100
│  │  │                        3.892 Mitem/s │ 1.317 Mitem/s │ 2.579 Mitem/s │ 2.477 Mitem/s │         │
│  │  ├─ 5                     12.38 ms      │ 31.89 ms      │ 17.62 ms      │ 18.35 ms      │ 100     │ 100
│  │  │                        4.035 Mitem/s │ 1.567 Mitem/s │ 2.836 Mitem/s │ 2.724 Mitem/s │         │
│  │  ╰─ 6                     12.95 ms      │ 33.26 ms      │ 18.66 ms      │ 20.66 ms      │ 100     │ 100
│  │                           4.631 Mitem/s │ 1.803 Mitem/s │ 3.214 Mitem/s │ 2.903 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.202 ms      │ 11.22 ms      │ 9.505 ms      │ 9.621 ms      │ 100     │ 100
│     │                        1.086 Mitem/s │ 891 Kitem/s   │ 1.052 Mitem/s │ 1.039 Mitem/s │         │
│     ├─ 2                     9.29 ms       │ 19.82 ms      │ 13.15 ms      │ 13.38 ms      │ 100     │ 100
│     │                        2.152 Mitem/s │ 1.008 Mitem/s │ 1.52 Mitem/s  │ 1.493 Mitem/s │         │
│     ├─ 3                     9.548 ms      │ 19.41 ms      │ 14.17 ms      │ 14.76 ms      │ 100     │ 100
│     │                        3.141 Mitem/s │ 1.544 Mitem/s │ 2.117 Mitem/s │ 2.032 Mitem/s │         │
│     ├─ 4                     9.589 ms      │ 26 ms         │ 14.49 ms      │ 14.73 ms      │ 100     │ 100
│     │                        4.171 Mitem/s │ 1.538 Mitem/s │ 2.758 Mitem/s │ 2.715 Mitem/s │         │
│     ├─ 5                     9.812 ms      │ 32.04 ms      │ 17.71 ms      │ 18.91 ms      │ 100     │ 100
│     │                        5.095 Mitem/s │ 1.56 Mitem/s  │ 2.821 Mitem/s │ 2.643 Mitem/s │         │
│     ╰─ 6                     9.745 ms      │ 32.03 ms      │ 17.82 ms      │ 19.76 ms      │ 100     │ 100
│                              6.156 Mitem/s │ 1.872 Mitem/s │ 3.365 Mitem/s │ 3.035 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     3.279 ms      │ 5.777 ms      │ 3.37 ms       │ 3.629 ms      │ 100     │ 100
│  │  │                        3.049 Mitem/s │ 1.73 Mitem/s  │ 2.967 Mitem/s │ 2.754 Mitem/s │         │
│  │  ├─ 2                     8.405 ms      │ 14.15 ms      │ 10.97 ms      │ 11.08 ms      │ 100     │ 100
│  │  │                        2.379 Mitem/s │ 1.412 Mitem/s │ 1.822 Mitem/s │ 1.804 Mitem/s │         │
│  │  ├─ 3                     13.12 ms      │ 25.99 ms      │ 20.93 ms      │ 21.14 ms      │ 100     │ 100
│  │  │                        2.285 Mitem/s │ 1.153 Mitem/s │ 1.433 Mitem/s │ 1.418 Mitem/s │         │
│  │  ├─ 4                     23.75 ms      │ 44.76 ms      │ 32.47 ms      │ 32.9 ms       │ 100     │ 100
│  │  │                        1.684 Mitem/s │ 893.5 Kitem/s │ 1.231 Mitem/s │ 1.215 Mitem/s │         │
│  │  ├─ 5                     36.04 ms      │ 58.76 ms      │ 46.83 ms      │ 46.94 ms      │ 100     │ 100
│  │  │                        1.387 Mitem/s │ 850.8 Kitem/s │ 1.067 Mitem/s │ 1.065 Mitem/s │         │
│  │  ╰─ 6                     61.43 ms      │ 87.32 ms      │ 72.41 ms      │ 72.55 ms      │ 100     │ 100
│  │                           976.5 Kitem/s │ 687 Kitem/s   │ 828.6 Kitem/s │ 826.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.679 ms      │ 10.23 ms      │ 7.809 ms      │ 8.03 ms       │ 100     │ 100
│  │  │                        1.302 Mitem/s │ 977.1 Kitem/s │ 1.28 Mitem/s  │ 1.245 Mitem/s │         │
│  │  ├─ 2                     8.036 ms      │ 16.28 ms      │ 12.14 ms      │ 12.3 ms       │ 100     │ 100
│  │  │                        2.488 Mitem/s │ 1.228 Mitem/s │ 1.647 Mitem/s │ 1.625 Mitem/s │         │
│  │  ├─ 3                     8.05 ms       │ 17.14 ms      │ 12.27 ms      │ 12.93 ms      │ 100     │ 100
│  │  │                        3.726 Mitem/s │ 1.749 Mitem/s │ 2.444 Mitem/s │ 2.318 Mitem/s │         │
│  │  ├─ 4                     8.818 ms      │ 24.69 ms      │ 12.73 ms      │ 13.35 ms      │ 100     │ 100
│  │  │                        4.535 Mitem/s │ 1.619 Mitem/s │ 3.141 Mitem/s │ 2.995 Mitem/s │         │
│  │  ├─ 5                     8.938 ms      │ 26.37 ms      │ 14.29 ms      │ 14.55 ms      │ 100     │ 100
│  │  │                        5.593 Mitem/s │ 1.895 Mitem/s │ 3.497 Mitem/s │ 3.435 Mitem/s │         │
│  │  ╰─ 6                     10.43 ms      │ 24.53 ms      │ 14.78 ms      │ 16.26 ms      │ 100     │ 100
│  │                           5.747 Mitem/s │ 2.445 Mitem/s │ 4.058 Mitem/s │ 3.689 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     9.204 ms      │ 15.05 ms      │ 9.709 ms      │ 10.35 ms      │ 100     │ 100
│     │                        1.086 Mitem/s │ 664.3 Kitem/s │ 1.029 Mitem/s │ 965.7 Kitem/s │         │
│     ├─ 2                     9.289 ms      │ 19.73 ms      │ 13.32 ms      │ 13.12 ms      │ 100     │ 100
│     │                        2.153 Mitem/s │ 1.013 Mitem/s │ 1.5 Mitem/s   │ 1.523 Mitem/s │         │
│     ├─ 3                     9.528 ms      │ 21.85 ms      │ 15.9 ms       │ 15.75 ms      │ 100     │ 100
│     │                        3.148 Mitem/s │ 1.372 Mitem/s │ 1.885 Mitem/s │ 1.904 Mitem/s │         │
│     ├─ 4                     9.546 ms      │ 29.21 ms      │ 14.91 ms      │ 14.87 ms      │ 100     │ 100
│     │                        4.19 Mitem/s  │ 1.369 Mitem/s │ 2.681 Mitem/s │ 2.688 Mitem/s │         │
│     ├─ 5                     9.779 ms      │ 31.44 ms      │ 16.18 ms      │ 16.72 ms      │ 100     │ 100
│     │                        5.112 Mitem/s │ 1.59 Mitem/s  │ 3.088 Mitem/s │ 2.99 Mitem/s  │         │
│     ╰─ 6                     9.997 ms      │ 29.65 ms      │ 18.26 ms      │ 19.55 ms      │ 100     │ 100
│                              6.001 Mitem/s │ 2.023 Mitem/s │ 3.285 Mitem/s │ 3.068 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.531 ms      │ 9.034 ms      │ 4.735 ms      │ 5.978 ms      │ 100     │ 100
│  │  │                        2.206 Mitem/s │ 1.106 Mitem/s │ 2.111 Mitem/s │ 1.672 Mitem/s │         │
│  │  ├─ 2                     11.57 ms      │ 20.33 ms      │ 15.23 ms      │ 15.49 ms      │ 100     │ 100
│  │  │                        1.727 Mitem/s │ 983.7 Kitem/s │ 1.312 Mitem/s │ 1.291 Mitem/s │         │
│  │  ├─ 3                     21.69 ms      │ 39.53 ms      │ 28.66 ms      │ 28.67 ms      │ 100     │ 100
│  │  │                        1.382 Mitem/s │ 758.7 Kitem/s │ 1.046 Mitem/s │ 1.046 Mitem/s │         │
│  │  ├─ 4                     36.48 ms      │ 58.96 ms      │ 42.93 ms      │ 43.9 ms       │ 100     │ 100
│  │  │                        1.096 Mitem/s │ 678.3 Kitem/s │ 931.6 Kitem/s │ 911 Kitem/s   │         │
│  │  ├─ 5                     50.8 ms       │ 73.45 ms      │ 60.64 ms      │ 61.15 ms      │ 100     │ 100
│  │  │                        984.1 Kitem/s │ 680.7 Kitem/s │ 824.4 Kitem/s │ 817.6 Kitem/s │         │
│  │  ╰─ 6                     78.59 ms      │ 110.4 ms      │ 94.59 ms      │ 93.67 ms      │ 100     │ 100
│  │                           763.3 Kitem/s │ 543.3 Kitem/s │ 634.2 Kitem/s │ 640.5 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.217 ms      │ 10.56 ms      │ 7.341 ms      │ 7.563 ms      │ 100     │ 100
│  │  │                        1.385 Mitem/s │ 946.4 Kitem/s │ 1.362 Mitem/s │ 1.322 Mitem/s │         │
│  │  ├─ 2                     7.948 ms      │ 15.68 ms      │ 13.37 ms      │ 12.03 ms      │ 100     │ 100
│  │  │                        2.516 Mitem/s │ 1.275 Mitem/s │ 1.495 Mitem/s │ 1.662 Mitem/s │         │
│  │  ├─ 3                     8.293 ms      │ 16.56 ms      │ 13.67 ms      │ 12.96 ms      │ 100     │ 100
│  │  │                        3.617 Mitem/s │ 1.811 Mitem/s │ 2.194 Mitem/s │ 2.314 Mitem/s │         │
│  │  ├─ 4                     8.311 ms      │ 25.72 ms      │ 13.19 ms      │ 12.74 ms      │ 100     │ 100
│  │  │                        4.812 Mitem/s │ 1.554 Mitem/s │ 3.03 Mitem/s  │ 3.137 Mitem/s │         │
│  │  ├─ 5                     8.539 ms      │ 23.82 ms      │ 13.71 ms      │ 14.38 ms      │ 100     │ 100
│  │  │                        5.855 Mitem/s │ 2.098 Mitem/s │ 3.646 Mitem/s │ 3.475 Mitem/s │         │
│  │  ╰─ 6                     8.47 ms       │ 21.78 ms      │ 13.53 ms      │ 13.39 ms      │ 100     │ 100
│  │                           7.083 Mitem/s │ 2.753 Mitem/s │ 4.433 Mitem/s │ 4.478 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.074 ms      │ 11.81 ms      │ 8.394 ms      │ 8.844 ms      │ 100     │ 100
│     │                        1.238 Mitem/s │ 846.6 Kitem/s │ 1.191 Mitem/s │ 1.13 Mitem/s  │         │
│     ├─ 2                     8.181 ms      │ 16.93 ms      │ 11.45 ms      │ 11.2 ms       │ 100     │ 100
│     │                        2.444 Mitem/s │ 1.181 Mitem/s │ 1.745 Mitem/s │ 1.785 Mitem/s │         │
│     ├─ 3                     8.504 ms      │ 17.6 ms       │ 12.65 ms      │ 13.21 ms      │ 100     │ 100
│     │                        3.527 Mitem/s │ 1.703 Mitem/s │ 2.37 Mitem/s  │ 2.27 Mitem/s  │         │
│     ├─ 4                     8.363 ms      │ 24.32 ms      │ 14.17 ms      │ 13.52 ms      │ 100     │ 100
│     │                        4.782 Mitem/s │ 1.644 Mitem/s │ 2.821 Mitem/s │ 2.957 Mitem/s │         │
│     ├─ 5                     8.652 ms      │ 24.63 ms      │ 14.8 ms       │ 15.21 ms      │ 100     │ 100
│     │                        5.778 Mitem/s │ 2.029 Mitem/s │ 3.377 Mitem/s │ 3.285 Mitem/s │         │
│     ╰─ 6                     8.471 ms      │ 23.8 ms       │ 14.5 ms       │ 14.17 ms      │ 100     │ 100
│                              7.082 Mitem/s │ 2.52 Mitem/s  │ 4.137 Mitem/s │ 4.231 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.56 ms       │ 6.365 ms      │ 4.645 ms      │ 4.725 ms      │ 100     │ 100
│  │  │                        2.192 Mitem/s │ 1.571 Mitem/s │ 2.152 Mitem/s │ 2.116 Mitem/s │         │
│  │  ├─ 2                     11.3 ms       │ 20.68 ms      │ 15.08 ms      │ 15.09 ms      │ 100     │ 100
│  │  │                        1.768 Mitem/s │ 966.7 Kitem/s │ 1.326 Mitem/s │ 1.324 Mitem/s │         │
│  │  ├─ 3                     22.87 ms      │ 36.06 ms      │ 30.86 ms      │ 30.32 ms      │ 100     │ 100
│  │  │                        1.311 Mitem/s │ 831.8 Kitem/s │ 972 Kitem/s   │ 989.4 Kitem/s │         │
│  │  ├─ 4                     35.83 ms      │ 57.34 ms      │ 44.75 ms      │ 45.13 ms      │ 100     │ 100
│  │  │                        1.116 Mitem/s │ 697.5 Kitem/s │ 893.8 Kitem/s │ 886.2 Kitem/s │         │
│  │  ├─ 5                     53.53 ms      │ 81.18 ms      │ 60.7 ms       │ 61.78 ms      │ 100     │ 100
│  │  │                        933.9 Kitem/s │ 615.8 Kitem/s │ 823.6 Kitem/s │ 809.2 Kitem/s │         │
│  │  ╰─ 6                     82.96 ms      │ 109.4 ms      │ 96.53 ms      │ 96.54 ms      │ 100     │ 100
│  │                           723.1 Kitem/s │ 548.4 Kitem/s │ 621.5 Kitem/s │ 621.4 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.728 ms      │ 10.96 ms      │ 7.912 ms      │ 8.193 ms      │ 100     │ 100
│  │  │                        1.293 Mitem/s │ 912 Kitem/s   │ 1.263 Mitem/s │ 1.22 Mitem/s  │         │
│  │  ├─ 2                     8.492 ms      │ 17.56 ms      │ 11.64 ms      │ 11.81 ms      │ 100     │ 100
│  │  │                        2.354 Mitem/s │ 1.138 Mitem/s │ 1.717 Mitem/s │ 1.692 Mitem/s │         │
│  │  ├─ 3                     8.858 ms      │ 17.51 ms      │ 14.46 ms      │ 14.11 ms      │ 100     │ 100
│  │  │                        3.386 Mitem/s │ 1.712 Mitem/s │ 2.074 Mitem/s │ 2.125 Mitem/s │         │
│  │  ├─ 4                     9.032 ms      │ 26.57 ms      │ 14.07 ms      │ 14.01 ms      │ 100     │ 100
│  │  │                        4.428 Mitem/s │ 1.505 Mitem/s │ 2.842 Mitem/s │ 2.854 Mitem/s │         │
│  │  ├─ 5                     9.131 ms      │ 24.67 ms      │ 14.43 ms      │ 14.78 ms      │ 100     │ 100
│  │  │                        5.475 Mitem/s │ 2.026 Mitem/s │ 3.464 Mitem/s │ 3.38 Mitem/s  │         │
│  │  ╰─ 6                     9.25 ms       │ 24.05 ms      │ 14.8 ms       │ 15.37 ms      │ 100     │ 100
│  │                           6.485 Mitem/s │ 2.494 Mitem/s │ 4.052 Mitem/s │ 3.901 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.45 ms       │ 13.82 ms      │ 8.694 ms      │ 9.213 ms      │ 100     │ 100
│     │                        1.183 Mitem/s │ 723.2 Kitem/s │ 1.15 Mitem/s  │ 1.085 Mitem/s │         │
│     ├─ 2                     8.593 ms      │ 17.98 ms      │ 12.01 ms      │ 11.87 ms      │ 100     │ 100
│     │                        2.327 Mitem/s │ 1.112 Mitem/s │ 1.664 Mitem/s │ 1.684 Mitem/s │         │
│     ├─ 3                     8.815 ms      │ 18.68 ms      │ 13.36 ms      │ 14.11 ms      │ 100     │ 100
│     │                        3.403 Mitem/s │ 1.605 Mitem/s │ 2.244 Mitem/s │ 2.126 Mitem/s │         │
│     ├─ 4                     8.989 ms      │ 28.15 ms      │ 15.21 ms      │ 15.28 ms      │ 100     │ 100
│     │                        4.449 Mitem/s │ 1.42 Mitem/s  │ 2.629 Mitem/s │ 2.617 Mitem/s │         │
│     ├─ 5                     8.972 ms      │ 30.19 ms      │ 15.93 ms      │ 15.86 ms      │ 100     │ 100
│     │                        5.572 Mitem/s │ 1.655 Mitem/s │ 3.137 Mitem/s │ 3.151 Mitem/s │         │
│     ╰─ 6                     9.006 ms      │ 29.67 ms      │ 16.21 ms      │ 17.36 ms      │ 100     │ 100
│                              6.661 Mitem/s │ 2.021 Mitem/s │ 3.699 Mitem/s │ 3.455 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.503 ms      │ 9.055 ms      │ 4.584 ms      │ 5.184 ms      │ 100     │ 100
│  │  │                        2.22 Mitem/s  │ 1.104 Mitem/s │ 2.181 Mitem/s │ 1.928 Mitem/s │         │
│  │  ├─ 2                     10.68 ms      │ 20.61 ms      │ 14.94 ms      │ 15.21 ms      │ 100     │ 100
│  │  │                        1.872 Mitem/s │ 970 Kitem/s   │ 1.338 Mitem/s │ 1.314 Mitem/s │         │
│  │  ├─ 3                     20.95 ms      │ 35.84 ms      │ 27.56 ms      │ 27.9 ms       │ 100     │ 100
│  │  │                        1.431 Mitem/s │ 836.8 Kitem/s │ 1.088 Mitem/s │ 1.075 Mitem/s │         │
│  │  ├─ 4                     33.7 ms       │ 52.69 ms      │ 41.15 ms      │ 41.05 ms      │ 100     │ 100
│  │  │                        1.186 Mitem/s │ 759 Kitem/s   │ 971.8 Kitem/s │ 974.3 Kitem/s │         │
│  │  ├─ 5                     51.33 ms      │ 81.09 ms      │ 60.59 ms      │ 61.45 ms      │ 100     │ 100
│  │  │                        974 Kitem/s   │ 616.5 Kitem/s │ 825.1 Kitem/s │ 813.6 Kitem/s │         │
│  │  ╰─ 6                     78.73 ms      │ 106.4 ms      │ 95.47 ms      │ 94.46 ms      │ 100     │ 100
│  │                           762 Kitem/s   │ 563.4 Kitem/s │ 628.4 Kitem/s │ 635.1 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.747 ms      │ 11.2 ms       │ 7.946 ms      │ 8.458 ms      │ 100     │ 100
│  │  │                        1.29 Mitem/s  │ 892.6 Kitem/s │ 1.258 Mitem/s │ 1.182 Mitem/s │         │
│  │  ├─ 2                     8.802 ms      │ 17.05 ms      │ 14.42 ms      │ 13.16 ms      │ 100     │ 100
│  │  │                        2.272 Mitem/s │ 1.172 Mitem/s │ 1.386 Mitem/s │ 1.519 Mitem/s │         │
│  │  ├─ 3                     9.06 ms       │ 17.43 ms      │ 12.5 ms       │ 12.94 ms      │ 100     │ 100
│  │  │                        3.311 Mitem/s │ 1.72 Mitem/s  │ 2.398 Mitem/s │ 2.317 Mitem/s │         │
│  │  ├─ 4                     9.025 ms      │ 24.13 ms      │ 14.43 ms      │ 14.32 ms      │ 100     │ 100
│  │  │                        4.431 Mitem/s │ 1.657 Mitem/s │ 2.771 Mitem/s │ 2.792 Mitem/s │         │
│  │  ├─ 5                     8.973 ms      │ 23.6 ms       │ 14.64 ms      │ 15.39 ms      │ 100     │ 100
│  │  │                        5.571 Mitem/s │ 2.118 Mitem/s │ 3.413 Mitem/s │ 3.247 Mitem/s │         │
│  │  ╰─ 6                     9.329 ms      │ 25.37 ms      │ 14.76 ms      │ 15.26 ms      │ 100     │ 100
│  │                           6.43 Mitem/s  │ 2.364 Mitem/s │ 4.062 Mitem/s │ 3.931 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.347 ms      │ 14.18 ms      │ 8.656 ms      │ 9.797 ms      │ 100     │ 100
│     │                        1.197 Mitem/s │ 705.1 Kitem/s │ 1.155 Mitem/s │ 1.02 Mitem/s  │         │
│     ├─ 2                     8.585 ms      │ 21.27 ms      │ 12.12 ms      │ 12.45 ms      │ 100     │ 100
│     │                        2.329 Mitem/s │ 940 Kitem/s   │ 1.649 Mitem/s │ 1.605 Mitem/s │         │
│     ├─ 3                     8.829 ms      │ 19.16 ms      │ 12.92 ms      │ 13.5 ms       │ 100     │ 100
│     │                        3.397 Mitem/s │ 1.565 Mitem/s │ 2.32 Mitem/s  │ 2.221 Mitem/s │         │
│     ├─ 4                     8.801 ms      │ 23.73 ms      │ 13.82 ms      │ 14.26 ms      │ 100     │ 100
│     │                        4.544 Mitem/s │ 1.685 Mitem/s │ 2.893 Mitem/s │ 2.803 Mitem/s │         │
│     ├─ 5                     8.946 ms      │ 27.64 ms      │ 15.28 ms      │ 15.65 ms      │ 100     │ 100
│     │                        5.588 Mitem/s │ 1.808 Mitem/s │ 3.272 Mitem/s │ 3.193 Mitem/s │         │
│     ╰─ 6                     9.015 ms      │ 28.24 ms      │ 16.14 ms      │ 17.03 ms      │ 100     │ 100
│                              6.655 Mitem/s │ 2.123 Mitem/s │ 3.715 Mitem/s │ 3.522 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.547 ms      │ 6.866 ms      │ 4.653 ms      │ 4.859 ms      │ 100     │ 100
│  │  │                        2.198 Mitem/s │ 1.456 Mitem/s │ 2.148 Mitem/s │ 2.057 Mitem/s │         │
│  │  ├─ 2                     11.87 ms      │ 19.82 ms      │ 15.97 ms      │ 15.96 ms      │ 100     │ 100
│  │  │                        1.683 Mitem/s │ 1.008 Mitem/s │ 1.251 Mitem/s │ 1.252 Mitem/s │         │
│  │  ├─ 3                     21.86 ms      │ 40.2 ms       │ 30.2 ms       │ 29.63 ms      │ 100     │ 100
│  │  │                        1.372 Mitem/s │ 746.1 Kitem/s │ 993.2 Kitem/s │ 1.012 Mitem/s │         │
│  │  ├─ 4                     33.17 ms      │ 56.75 ms      │ 42.84 ms      │ 43.25 ms      │ 100     │ 100
│  │  │                        1.205 Mitem/s │ 704.7 Kitem/s │ 933.5 Kitem/s │ 924.8 Kitem/s │         │
│  │  ├─ 5                     52.71 ms      │ 75.34 ms      │ 62.38 ms      │ 62.91 ms      │ 100     │ 100
│  │  │                        948.5 Kitem/s │ 663.6 Kitem/s │ 801.4 Kitem/s │ 794.7 Kitem/s │         │
│  │  ╰─ 6                     81.68 ms      │ 110.7 ms      │ 94.01 ms      │ 94.19 ms      │ 100     │ 100
│  │                           734.5 Kitem/s │ 541.6 Kitem/s │ 638.1 Kitem/s │ 636.9 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.766 ms      │ 11.39 ms      │ 7.906 ms      │ 8.319 ms      │ 100     │ 100
│  │  │                        1.287 Mitem/s │ 877.8 Kitem/s │ 1.264 Mitem/s │ 1.202 Mitem/s │         │
│  │  ├─ 2                     8.21 ms       │ 16.69 ms      │ 10.37 ms      │ 11.42 ms      │ 100     │ 100
│  │  │                        2.435 Mitem/s │ 1.197 Mitem/s │ 1.928 Mitem/s │ 1.751 Mitem/s │         │
│  │  ├─ 3                     8.503 ms      │ 20.55 ms      │ 12.39 ms      │ 12.59 ms      │ 100     │ 100
│  │  │                        3.527 Mitem/s │ 1.459 Mitem/s │ 2.419 Mitem/s │ 2.382 Mitem/s │         │
│  │  ├─ 4                     8.993 ms      │ 23.06 ms      │ 14.34 ms      │ 13.95 ms      │ 100     │ 100
│  │  │                        4.447 Mitem/s │ 1.734 Mitem/s │ 2.788 Mitem/s │ 2.865 Mitem/s │         │
│  │  ├─ 5                     9.116 ms      │ 24.91 ms      │ 14.65 ms      │ 15.47 ms      │ 100     │ 100
│  │  │                        5.484 Mitem/s │ 2.006 Mitem/s │ 3.412 Mitem/s │ 3.231 Mitem/s │         │
│  │  ╰─ 6                     9.274 ms      │ 25.7 ms       │ 14.98 ms      │ 16.35 ms      │ 100     │ 100
│  │                           6.469 Mitem/s │ 2.334 Mitem/s │ 4.003 Mitem/s │ 3.668 Mitem/s │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.365 ms      │ 11.62 ms      │ 8.659 ms      │ 8.94 ms       │ 100     │ 100
│     │                        1.195 Mitem/s │ 860.4 Kitem/s │ 1.154 Mitem/s │ 1.118 Mitem/s │         │
│     ├─ 2                     8.514 ms      │ 17.96 ms      │ 11.97 ms      │ 12.06 ms      │ 100     │ 100
│     │                        2.348 Mitem/s │ 1.113 Mitem/s │ 1.67 Mitem/s  │ 1.657 Mitem/s │         │
│     ├─ 3                     10.73 ms      │ 24.83 ms      │ 15.45 ms      │ 15.31 ms      │ 100     │ 100
│     │                        2.793 Mitem/s │ 1.208 Mitem/s │ 1.941 Mitem/s │ 1.958 Mitem/s │         │
│     ├─ 4                     8.869 ms      │ 30.47 ms      │ 13.21 ms      │ 13.89 ms      │ 100     │ 100
│     │                        4.509 Mitem/s │ 1.312 Mitem/s │ 3.025 Mitem/s │ 2.879 Mitem/s │         │
│     ├─ 5                     8.931 ms      │ 25.77 ms      │ 15.06 ms      │ 14.92 ms      │ 100     │ 100
│     │                        5.598 Mitem/s │ 1.94 Mitem/s  │ 3.318 Mitem/s │ 3.35 Mitem/s  │         │
│     ╰─ 6                     8.806 ms      │ 30.23 ms      │ 16.07 ms      │ 17.44 ms      │ 100     │ 100
│                              6.813 Mitem/s │ 1.984 Mitem/s │ 3.731 Mitem/s │ 3.438 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ├─ indexset                               │               │               │               │         │
│  │  ├─ 1                     4.147 ms      │ 7.528 ms      │ 4.223 ms      │ 4.482 ms      │ 100     │ 100
│  │  │                        2.411 Mitem/s │ 1.328 Mitem/s │ 2.367 Mitem/s │ 2.23 Mitem/s  │         │
│  │  ├─ 2                     10.84 ms      │ 18.93 ms      │ 15.22 ms      │ 15 ms         │ 100     │ 100
│  │  │                        1.844 Mitem/s │ 1.055 Mitem/s │ 1.313 Mitem/s │ 1.332 Mitem/s │         │
│  │  ├─ 3                     18.77 ms      │ 33.72 ms      │ 25.94 ms      │ 25.83 ms      │ 100     │ 100
│  │  │                        1.597 Mitem/s │ 889.5 Kitem/s │ 1.156 Mitem/s │ 1.161 Mitem/s │         │
│  │  ├─ 4                     30.61 ms      │ 53.59 ms      │ 37.27 ms      │ 37.8 ms       │ 100     │ 100
│  │  │                        1.306 Mitem/s │ 746.4 Kitem/s │ 1.073 Mitem/s │ 1.057 Mitem/s │         │
│  │  ├─ 5                     44.94 ms      │ 66.78 ms      │ 52.03 ms      │ 53.16 ms      │ 100     │ 100
│  │  │                        1.112 Mitem/s │ 748.7 Kitem/s │ 960.9 Kitem/s │ 940.3 Kitem/s │         │
│  │  ╰─ 6                     72.94 ms      │ 104.5 ms      │ 87.28 ms      │ 86.25 ms      │ 100     │ 100
│  │                           822.5 Kitem/s │ 574 Kitem/s   │ 687.4 Kitem/s │ 695.6 Kitem/s │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 1                     7.91 ms       │ 15.89 ms      │ 8.34 ms       │ 9.395 ms      │ 100     │ 100
│  │  │                        1.264 Mitem/s │ 629.2 Kitem/s │ 1.198 Mitem/s │ 1.064 Mitem/s │         │
│  │  ├─ 2                     8.133 ms      │ 17.84 ms      │ 12.02 ms      │ 12.63 ms      │ 100     │ 100
│  │  │                        2.459 Mitem/s │ 1.121 Mitem/s │ 1.663 Mitem/s │ 1.582 Mitem/s │         │
│  │  ├─ 3                     8.289 ms      │ 17.25 ms      │ 12.45 ms      │ 13.3 ms       │ 100     │ 100
│  │  │                        3.619 Mitem/s │ 1.738 Mitem/s │ 2.408 Mitem/s │ 2.255 Mitem/s │         │
│  │  ├─ 4                     8.987 ms      │ 20.12 ms      │ 12.81 ms      │ 13.33 ms      │ 100     │ 100
│  │  │                        4.45 Mitem/s  │ 1.987 Mitem/s │ 3.122 Mitem/s │ 3 Mitem/s     │         │
│  │  ├─ 5                     9.043 ms      │ 27.21 ms      │ 14.65 ms      │ 15.39 ms      │ 100     │ 100
│  │  │                        5.529 Mitem/s │ 1.837 Mitem/s │ 3.41 Mitem/s  │ 3.248 Mitem/s │         │
│  │  ╰─ 6                     9.262 ms      │ 26.79 ms      │ 15.25 ms      │ 17.59 ms      │ 100     │ 100
│  │                           6.477 Mitem/s │ 2.239 Mitem/s │ 3.933 Mitem/s │ 3.41 Mitem/s  │         │
│  ╰─ tree_index                             │               │               │               │         │
│     ├─ 1                     8.871 ms      │ 12.72 ms      │ 9.375 ms      │ 9.73 ms       │ 100     │ 100
│     │                        1.127 Mitem/s │ 786 Kitem/s   │ 1.066 Mitem/s │ 1.027 Mitem/s │         │
│     ├─ 2                     8.903 ms      │ 17.79 ms      │ 12.51 ms      │ 12.19 ms      │ 100     │ 100
│     │                        2.246 Mitem/s │ 1.124 Mitem/s │ 1.597 Mitem/s │ 1.639 Mitem/s │         │
│     ├─ 3                     9.1 ms        │ 23.02 ms      │ 13.1 ms       │ 13.18 ms      │ 100     │ 100
│     │                        3.296 Mitem/s │ 1.302 Mitem/s │ 2.289 Mitem/s │ 2.275 Mitem/s │         │
│     ├─ 4                     9.214 ms      │ 27.01 ms      │ 15.6 ms       │ 15.16 ms      │ 100     │ 100
│     │                        4.341 Mitem/s │ 1.48 Mitem/s  │ 2.563 Mitem/s │ 2.637 Mitem/s │         │
│     ├─ 5                     9.675 ms      │ 28.38 ms      │ 15.84 ms      │ 16.08 ms      │ 100     │ 100
│     │                        5.167 Mitem/s │ 1.761 Mitem/s │ 3.154 Mitem/s │ 3.108 Mitem/s │         │
│     ╰─ 6                     9.284 ms      │ 30.63 ms      │ 16.7 ms       │ 17.96 ms      │ 100     │ 100
│                              6.462 Mitem/s │ 1.958 Mitem/s │ 3.592 Mitem/s │ 3.34 Mitem/s  │         │
```
