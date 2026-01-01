```bash
Timer precision: 30 ns
range_concurrent               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
  ├─ masstree24                              │               │               │               │         │
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
─ 02_reverse_scan                            │               │               │               │         │
  ├─ masstree24                              │               │               │               │         │
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
├─ 03_clustered_scan                         │               │               │               │         │
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
├─ 04_sparse_scan                            │               │               │               │         │
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
├─ 05_shared_prefix_scan                     │               │               │               │         │
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
├─ 06_suffix_differ_scan                     │               │               │               │         │
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
├─ 07_hierarchical_scan                      │               │               │               │         │
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
├─ 08_adversarial_splits_scan                │               │               │               │         │
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
├─ 09_interleaved_scan                       │               │               │               │         │
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
├─ 10_blink_stress_scan                      │               │               │               │         │
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
├─ 11_random_keys_scan                       │               │               │               │         │
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
├─ 12_long_keys_64b_scan                     │               │               │               │         │
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
├─ 13_scan_while_insert                      │               │               │               │         │
│  ├─ masstree24                             │               │               │               │         │
│  │  ├─ 3                     8.818 ms      │ 15.99 ms      │ 12.69 ms      │ 12.7 ms       │ 100     │ 100
│  │  ├─ 4                     9.948 ms      │ 17.7 ms       │ 13.76 ms      │ 13.78 ms      │ 100     │ 100
│  │  ├─ 5                     12.21 ms      │ 24.43 ms      │ 14.16 ms      │ 14.42 ms      │ 100     │ 100
│  │  ╰─ 6                     13.24 ms      │ 22.42 ms      │ 14.66 ms      │ 14.88 ms      │ 100     │ 100
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
├─ 16_insert_heavy                           │               │               │               │         │
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
├─ 17_hot_spot                               │               │               │               │         │
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
╰─ 18_split_inducing_scan                    │               │               │               │         │
   ├─ masstree24                             │               │               │               │         │
   │  ├─ 3                     7.891 ms      │ 16.3 ms       │ 9.77 ms       │ 11.01 ms      │ 100     │ 100
   │  ├─ 4                     9.355 ms      │ 22.7 ms       │ 14.79 ms      │ 14 ms         │ 100     │ 100
   │  ├─ 5                     9.598 ms      │ 23.92 ms      │ 13.91 ms      │ 13.73 ms      │ 100     │ 100
   │  ╰─ 6                     10.24 ms      │ 24.63 ms      │ 14.36 ms      │ 15.05 ms      │ 100     │ 100
```
