```bash
imer precision: 20 ns
range_masstree24               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ 01_sequential_full_scan                   │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.063 ms      │ 10.49 ms      │ 7.4 ms        │ 7.825 ms      │ 100     │ 100
│     │                        1.415 Mitem/s │ 952.8 Kitem/s │ 1.351 Mitem/s │ 1.277 Mitem/s │         │
│     ├─ 2                     7.659 ms      │ 15.48 ms      │ 9.458 ms      │ 10.03 ms      │ 100     │ 100
│     │                        2.611 Mitem/s │ 1.291 Mitem/s │ 2.114 Mitem/s │ 1.992 Mitem/s │         │
│     ├─ 3                     8.036 ms      │ 18.9 ms       │ 11.78 ms      │ 11.95 ms      │ 100     │ 100
│     │                        3.732 Mitem/s │ 1.586 Mitem/s │ 2.544 Mitem/s │ 2.51 Mitem/s  │         │
│     ├─ 4                     8.76 ms       │ 19.54 ms      │ 12.5 ms       │ 12.2 ms       │ 100     │ 100
│     │                        4.566 Mitem/s │ 2.046 Mitem/s │ 3.197 Mitem/s │ 3.277 Mitem/s │         │
│     ├─ 5                     8.984 ms      │ 21.93 ms      │ 13.55 ms      │ 13.68 ms      │ 100     │ 100
│     │                        5.565 Mitem/s │ 2.279 Mitem/s │ 3.688 Mitem/s │ 3.652 Mitem/s │         │
│     ╰─ 6                     9.026 ms      │ 22.72 ms      │ 13.69 ms      │ 14.04 ms      │ 100     │ 100
│                              6.647 Mitem/s │ 2.639 Mitem/s │ 4.38 Mitem/s  │ 4.272 Mitem/s │         │
├─ 02_reverse_scan                           │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     6.993 ms      │ 10.18 ms      │ 7.291 ms      │ 7.644 ms      │ 100     │ 100
│     │                        1.429 Mitem/s │ 981.9 Kitem/s │ 1.371 Mitem/s │ 1.308 Mitem/s │         │
│     ├─ 2                     7.379 ms      │ 15.5 ms       │ 9.118 ms      │ 10.34 ms      │ 100     │ 100
│     │                        2.71 Mitem/s  │ 1.289 Mitem/s │ 2.193 Mitem/s │ 1.933 Mitem/s │         │
│     ├─ 3                     7.607 ms      │ 16 ms         │ 11.13 ms      │ 11.05 ms      │ 100     │ 100
│     │                        3.943 Mitem/s │ 1.874 Mitem/s │ 2.694 Mitem/s │ 2.712 Mitem/s │         │
│     ├─ 4                     8.493 ms      │ 19.2 ms       │ 12.98 ms      │ 12.99 ms      │ 100     │ 100
│     │                        4.709 Mitem/s │ 2.082 Mitem/s │ 3.079 Mitem/s │ 3.077 Mitem/s │         │
│     ├─ 5                     9.058 ms      │ 21.85 ms      │ 13.11 ms      │ 13.52 ms      │ 100     │ 100
│     │                        5.519 Mitem/s │ 2.287 Mitem/s │ 3.813 Mitem/s │ 3.697 Mitem/s │         │
│     ╰─ 6                     8.966 ms      │ 22.51 ms      │ 13.6 ms       │ 13.93 ms      │ 100     │ 100
│                              6.691 Mitem/s │ 2.665 Mitem/s │ 4.409 Mitem/s │ 4.305 Mitem/s │         │
├─ 03_clustered_scan                         │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.042 ms      │ 15.98 ms      │ 7.387 ms      │ 7.756 ms      │ 100     │ 100
│     │                        1.419 Mitem/s │ 625.5 Kitem/s │ 1.353 Mitem/s │ 1.289 Mitem/s │         │
│     ├─ 2                     7.796 ms      │ 15.96 ms      │ 10.73 ms      │ 10.76 ms      │ 100     │ 100
│     │                        2.565 Mitem/s │ 1.253 Mitem/s │ 1.862 Mitem/s │ 1.857 Mitem/s │         │
│     ├─ 3                     8.077 ms      │ 17.04 ms      │ 11.82 ms      │ 12.55 ms      │ 100     │ 100
│     │                        3.713 Mitem/s │ 1.76 Mitem/s  │ 2.536 Mitem/s │ 2.389 Mitem/s │         │
│     ├─ 4                     8.609 ms      │ 22.37 ms      │ 11.92 ms      │ 12.01 ms      │ 100     │ 100
│     │                        4.645 Mitem/s │ 1.787 Mitem/s │ 3.353 Mitem/s │ 3.328 Mitem/s │         │
│     ├─ 5                     9.298 ms      │ 23.53 ms      │ 13.54 ms      │ 14.12 ms      │ 100     │ 100
│     │                        5.377 Mitem/s │ 2.124 Mitem/s │ 3.691 Mitem/s │ 3.54 Mitem/s  │         │
│     ╰─ 6                     9.174 ms      │ 22.38 ms      │ 13.73 ms      │ 14.63 ms      │ 100     │ 100
│                              6.539 Mitem/s │ 2.68 Mitem/s  │ 4.369 Mitem/s │ 4.1 Mitem/s   │         │
├─ 04_sparse_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.11 ms       │ 9.998 ms      │ 7.397 ms      │ 7.547 ms      │ 100     │ 100
│     │                        1.406 Mitem/s │ 1 Mitem/s     │ 1.351 Mitem/s │ 1.324 Mitem/s │         │
│     ├─ 2                     7.546 ms      │ 15.73 ms      │ 8.947 ms      │ 9.627 ms      │ 100     │ 100
│     │                        2.65 Mitem/s  │ 1.271 Mitem/s │ 2.235 Mitem/s │ 2.077 Mitem/s │         │
│     ├─ 3                     8.565 ms      │ 16.74 ms      │ 11.88 ms      │ 12.45 ms      │ 100     │ 100
│     │                        3.502 Mitem/s │ 1.791 Mitem/s │ 2.524 Mitem/s │ 2.408 Mitem/s │         │
│     ├─ 4                     8.594 ms      │ 21.44 ms      │ 12.49 ms      │ 12.38 ms      │ 100     │ 100
│     │                        4.654 Mitem/s │ 1.865 Mitem/s │ 3.201 Mitem/s │ 3.23 Mitem/s  │         │
│     ├─ 5                     9.119 ms      │ 23.83 ms      │ 13.36 ms      │ 13.27 ms      │ 100     │ 100
│     │                        5.482 Mitem/s │ 2.097 Mitem/s │ 3.741 Mitem/s │ 3.765 Mitem/s │         │
│     ╰─ 6                     9.19 ms       │ 25.05 ms      │ 13.75 ms      │ 14.99 ms      │ 100     │ 100
│                              6.528 Mitem/s │ 2.394 Mitem/s │ 4.36 Mitem/s  │ 4.001 Mitem/s │         │
├─ 05_shared_prefix_scan                     │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     6.51 ms       │ 12.58 ms      │ 6.826 ms      │ 7.266 ms      │ 100     │ 100
│     │                        1.535 Mitem/s │ 794.6 Kitem/s │ 1.464 Mitem/s │ 1.376 Mitem/s │         │
│     ├─ 2                     6.911 ms      │ 14.4 ms       │ 7.941 ms      │ 8.571 ms      │ 100     │ 100
│     │                        2.893 Mitem/s │ 1.388 Mitem/s │ 2.518 Mitem/s │ 2.333 Mitem/s │         │
│     ├─ 3                     7.188 ms      │ 15.27 ms      │ 11.84 ms      │ 11.21 ms      │ 100     │ 100
│     │                        4.173 Mitem/s │ 1.963 Mitem/s │ 2.532 Mitem/s │ 2.676 Mitem/s │         │
│     ├─ 4                     8.037 ms      │ 20.46 ms      │ 11.69 ms      │ 11.81 ms      │ 100     │ 100
│     │                        4.976 Mitem/s │ 1.954 Mitem/s │ 3.419 Mitem/s │ 3.384 Mitem/s │         │
│     ├─ 5                     8.736 ms      │ 20.9 ms       │ 12.75 ms      │ 12.5 ms       │ 100     │ 100
│     │                        5.722 Mitem/s │ 2.391 Mitem/s │ 3.92 Mitem/s  │ 3.997 Mitem/s │         │
│     ╰─ 6                     8.57 ms       │ 21.53 ms      │ 13.08 ms      │ 14.36 ms      │ 100     │ 100
│                              7 Mitem/s     │ 2.785 Mitem/s │ 4.584 Mitem/s │ 4.177 Mitem/s │         │
├─ 06_suffix_differ_scan                     │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     9.97 ms       │ 14.57 ms      │ 10.97 ms      │ 11.34 ms      │ 100     │ 100
│     │                        1.002 Mitem/s │ 685.9 Kitem/s │ 910.9 Kitem/s │ 881.6 Kitem/s │         │
│     ├─ 2                     10.18 ms      │ 24.16 ms      │ 11.26 ms      │ 11.92 ms      │ 100     │ 100
│     │                        1.963 Mitem/s │ 827.8 Kitem/s │ 1.775 Mitem/s │ 1.676 Mitem/s │         │
│     ├─ 3                     10.21 ms      │ 24.75 ms      │ 15.42 ms      │ 15.12 ms      │ 100     │ 100
│     │                        2.936 Mitem/s │ 1.211 Mitem/s │ 1.945 Mitem/s │ 1.983 Mitem/s │         │
│     ├─ 4                     10.44 ms      │ 31.18 ms      │ 16.21 ms      │ 16.3 ms       │ 100     │ 100
│     │                        3.83 Mitem/s  │ 1.282 Mitem/s │ 2.466 Mitem/s │ 2.453 Mitem/s │         │
│     ├─ 5                     10.79 ms      │ 30.73 ms      │ 18.25 ms      │ 17.86 ms      │ 100     │ 100
│     │                        4.633 Mitem/s │ 1.626 Mitem/s │ 2.739 Mitem/s │ 2.798 Mitem/s │         │
│     ╰─ 6                     13.78 ms      │ 33 ms         │ 19.1 ms       │ 21.5 ms       │ 100     │ 100
│                              4.352 Mitem/s │ 1.817 Mitem/s │ 3.14 Mitem/s  │ 2.789 Mitem/s │         │
├─ 07_hierarchical_scan                      │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.179 ms      │ 10.04 ms      │ 7.364 ms      │ 7.548 ms      │ 100     │ 100
│     │                        1.392 Mitem/s │ 995.6 Kitem/s │ 1.357 Mitem/s │ 1.324 Mitem/s │         │
│     ├─ 2                     7.541 ms      │ 15.21 ms      │ 9.002 ms      │ 9.806 ms      │ 100     │ 100
│     │                        2.651 Mitem/s │ 1.314 Mitem/s │ 2.221 Mitem/s │ 2.039 Mitem/s │         │
│     ├─ 3                     7.565 ms      │ 19.19 ms      │ 9.987 ms      │ 10.89 ms      │ 100     │ 100
│     │                        3.965 Mitem/s │ 1.562 Mitem/s │ 3.003 Mitem/s │ 2.753 Mitem/s │         │
│     ├─ 4                     8.459 ms      │ 22.77 ms      │ 13.35 ms      │ 13.65 ms      │ 100     │ 100
│     │                        4.728 Mitem/s │ 1.756 Mitem/s │ 2.994 Mitem/s │ 2.929 Mitem/s │         │
│     ├─ 5                     8.927 ms      │ 22.81 ms      │ 14.04 ms      │ 14.05 ms      │ 100     │ 100
│     │                        5.6 Mitem/s   │ 2.191 Mitem/s │ 3.561 Mitem/s │ 3.556 Mitem/s │         │
│     ╰─ 6                     9.144 ms      │ 23.87 ms      │ 14.02 ms      │ 15.01 ms      │ 100     │ 100
│                              6.561 Mitem/s │ 2.512 Mitem/s │ 4.279 Mitem/s │ 3.996 Mitem/s │         │
├─ 08_adversarial_splits_scan                │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     6.466 ms      │ 13.16 ms      │ 6.631 ms      │ 7.045 ms      │ 100     │ 100
│     │                        1.546 Mitem/s │ 759.7 Kitem/s │ 1.507 Mitem/s │ 1.419 Mitem/s │         │
│     ├─ 2                     6.997 ms      │ 14.69 ms      │ 8.323 ms      │ 9.127 ms      │ 100     │ 100
│     │                        2.858 Mitem/s │ 1.361 Mitem/s │ 2.402 Mitem/s │ 2.191 Mitem/s │         │
│     ├─ 3                     7.527 ms      │ 15.21 ms      │ 10.97 ms      │ 11.39 ms      │ 100     │ 100
│     │                        3.985 Mitem/s │ 1.972 Mitem/s │ 2.733 Mitem/s │ 2.633 Mitem/s │         │
│     ├─ 4                     7.922 ms      │ 18.17 ms      │ 11.89 ms      │ 11.63 ms      │ 100     │ 100
│     │                        5.049 Mitem/s │ 2.201 Mitem/s │ 3.361 Mitem/s │ 3.437 Mitem/s │         │
│     ├─ 5                     9.1 ms        │ 21.67 ms      │ 12.5 ms       │ 13.16 ms      │ 100     │ 100
│     │                        5.494 Mitem/s │ 2.306 Mitem/s │ 3.999 Mitem/s │ 3.799 Mitem/s │         │
│     ╰─ 6                     12.17 ms      │ 19.03 ms      │ 12.74 ms      │ 13.14 ms      │ 100     │ 100
│                              4.926 Mitem/s │ 3.151 Mitem/s │ 4.706 Mitem/s │ 4.563 Mitem/s │         │
├─ 09_interleaved_scan                       │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.987 ms      │ 13.52 ms      │ 8.292 ms      │ 8.498 ms      │ 100     │ 100
│     │                        1.251 Mitem/s │ 739.4 Kitem/s │ 1.205 Mitem/s │ 1.176 Mitem/s │         │
│     ├─ 2                     8.484 ms      │ 14.7 ms       │ 9.246 ms      │ 9.685 ms      │ 100     │ 100
│     │                        2.357 Mitem/s │ 1.359 Mitem/s │ 2.162 Mitem/s │ 2.064 Mitem/s │         │
│     ├─ 3                     8.654 ms      │ 15.95 ms      │ 9.93 ms       │ 10.9 ms       │ 100     │ 100
│     │                        3.466 Mitem/s │ 1.88 Mitem/s  │ 3.02 Mitem/s  │ 2.752 Mitem/s │         │
│     ├─ 4                     9.589 ms      │ 24.88 ms      │ 14.59 ms      │ 13.62 ms      │ 100     │ 100
│     │                        4.171 Mitem/s │ 1.607 Mitem/s │ 2.74 Mitem/s  │ 2.936 Mitem/s │         │
│     ├─ 5                     10.12 ms      │ 25.64 ms      │ 15.03 ms      │ 14.91 ms      │ 100     │ 100
│     │                        4.938 Mitem/s │ 1.949 Mitem/s │ 3.325 Mitem/s │ 3.351 Mitem/s │         │
│     ╰─ 6                     12.25 ms      │ 22.61 ms      │ 15.31 ms      │ 15.65 ms      │ 100     │ 100
│                              4.897 Mitem/s │ 2.653 Mitem/s │ 3.918 Mitem/s │ 3.831 Mitem/s │         │
├─ 10_blink_stress_scan                      │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.165 ms      │ 9.724 ms      │ 7.449 ms      │ 7.648 ms      │ 100     │ 100
│     │                        1.395 Mitem/s │ 1.028 Mitem/s │ 1.342 Mitem/s │ 1.307 Mitem/s │         │
│     ├─ 2                     7.536 ms      │ 15.69 ms      │ 8.378 ms      │ 8.683 ms      │ 100     │ 100
│     │                        2.653 Mitem/s │ 1.274 Mitem/s │ 2.387 Mitem/s │ 2.303 Mitem/s │         │
│     ├─ 3                     7.865 ms      │ 15.46 ms      │ 10.63 ms      │ 10.89 ms      │ 100     │ 100
│     │                        3.813 Mitem/s │ 1.939 Mitem/s │ 2.82 Mitem/s  │ 2.752 Mitem/s │         │
│     ├─ 4                     8.788 ms      │ 18.8 ms       │ 12.01 ms      │ 11.74 ms      │ 100     │ 100
│     │                        4.551 Mitem/s │ 2.127 Mitem/s │ 3.327 Mitem/s │ 3.406 Mitem/s │         │
│     ├─ 5                     9.141 ms      │ 23.36 ms      │ 13.36 ms      │ 13.34 ms      │ 100     │ 100
│     │                        5.469 Mitem/s │ 2.139 Mitem/s │ 3.74 Mitem/s  │ 3.748 Mitem/s │         │
│     ╰─ 6                     9.589 ms      │ 22.08 ms      │ 13.85 ms      │ 14.4 ms       │ 100     │ 100
│                              6.256 Mitem/s │ 2.716 Mitem/s │ 4.33 Mitem/s  │ 4.164 Mitem/s │         │
├─ 11_random_keys_scan                       │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.113 ms      │ 12.23 ms      │ 7.465 ms      │ 7.742 ms      │ 100     │ 100
│     │                        1.405 Mitem/s │ 817.1 Kitem/s │ 1.339 Mitem/s │ 1.291 Mitem/s │         │
│     ├─ 2                     7.694 ms      │ 15.59 ms      │ 8.955 ms      │ 10.04 ms      │ 100     │ 100
│     │                        2.599 Mitem/s │ 1.282 Mitem/s │ 2.233 Mitem/s │ 1.991 Mitem/s │         │
│     ├─ 3                     7.828 ms      │ 20.39 ms      │ 11.12 ms      │ 11.42 ms      │ 100     │ 100
│     │                        3.832 Mitem/s │ 1.471 Mitem/s │ 2.697 Mitem/s │ 2.625 Mitem/s │         │
│     ├─ 4                     8.849 ms      │ 24.67 ms      │ 12.81 ms      │ 12.5 ms       │ 100     │ 100
│     │                        4.52 Mitem/s  │ 1.621 Mitem/s │ 3.122 Mitem/s │ 3.199 Mitem/s │         │
│     ├─ 5                     9.266 ms      │ 23.7 ms       │ 13.49 ms      │ 13.63 ms      │ 100     │ 100
│     │                        5.395 Mitem/s │ 2.108 Mitem/s │ 3.703 Mitem/s │ 3.668 Mitem/s │         │
│     ╰─ 6                     9.161 ms      │ 23.04 ms      │ 13.87 ms      │ 14.45 ms      │ 100     │ 100
│                              6.549 Mitem/s │ 2.603 Mitem/s │ 4.324 Mitem/s │ 4.151 Mitem/s │         │
├─ 12_long_keys_64b_scan                     │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     7.126 ms      │ 12.13 ms      │ 7.359 ms      │ 8.036 ms      │ 100     │ 100
│     │                        1.403 Mitem/s │ 824.1 Kitem/s │ 1.358 Mitem/s │ 1.244 Mitem/s │         │
│     ├─ 2                     7.681 ms      │ 15.33 ms      │ 8.982 ms      │ 9.868 ms      │ 100     │ 100
│     │                        2.603 Mitem/s │ 1.303 Mitem/s │ 2.226 Mitem/s │ 2.026 Mitem/s │         │
│     ├─ 3                     8.022 ms      │ 16.66 ms      │ 9.859 ms      │ 10.77 ms      │ 100     │ 100
│     │                        3.739 Mitem/s │ 1.8 Mitem/s   │ 3.042 Mitem/s │ 2.785 Mitem/s │         │
│     ├─ 4                     8.912 ms      │ 15.86 ms      │ 12.47 ms      │ 12.12 ms      │ 100     │ 100
│     │                        4.488 Mitem/s │ 2.521 Mitem/s │ 3.205 Mitem/s │ 3.298 Mitem/s │         │
│     ├─ 5                     9.097 ms      │ 22.49 ms      │ 13.6 ms       │ 13.64 ms      │ 100     │ 100
│     │                        5.495 Mitem/s │ 2.223 Mitem/s │ 3.675 Mitem/s │ 3.664 Mitem/s │         │
│     ╰─ 6                     9.393 ms      │ 22.8 ms       │ 14.31 ms      │ 14.85 ms      │ 100     │ 100
│                              6.387 Mitem/s │ 2.63 Mitem/s  │ 4.191 Mitem/s │ 4.04 Mitem/s  │         │
├─ 13_scan_while_insert                      │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 3                     8.021 ms      │ 16.08 ms      │ 11.4 ms       │ 11.67 ms      │ 100     │ 100
│     ├─ 4                     9.929 ms      │ 16.01 ms      │ 14.11 ms      │ 13.59 ms      │ 100     │ 100
│     ├─ 5                     11.32 ms      │ 23.78 ms      │ 13.88 ms      │ 13.94 ms      │ 100     │ 100
│     ╰─ 6                     11.85 ms      │ 23.33 ms      │ 14.9 ms       │ 15 ms         │ 100     │ 100
├─ 14_prefix_scan                            │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     20.51 ms      │ 23.29 ms      │ 21.51 ms      │ 21.5 ms       │ 20      │ 20
│     │                        4.875 Kitem/s │ 4.293 Kitem/s │ 4.647 Kitem/s │ 4.649 Kitem/s │         │
│     ├─ 2                     20.98 ms      │ 50.29 ms      │ 45.06 ms      │ 41.74 ms      │ 20      │ 20
│     │                        9.528 Kitem/s │ 3.976 Kitem/s │ 4.437 Kitem/s │ 4.79 Kitem/s  │         │
│     ├─ 3                     23.69 ms      │ 61.4 ms       │ 55.71 ms      │ 50.28 ms      │ 20      │ 20
│     │                        12.66 Kitem/s │ 4.885 Kitem/s │ 5.384 Kitem/s │ 5.965 Kitem/s │         │
│     ├─ 4                     36.74 ms      │ 62.48 ms      │ 56.8 ms       │ 55.71 ms      │ 20      │ 20
│     │                        10.88 Kitem/s │ 6.401 Kitem/s │ 7.041 Kitem/s │ 7.179 Kitem/s │         │
│     ├─ 5                     32.8 ms       │ 68.48 ms      │ 59.83 ms      │ 56.04 ms      │ 20      │ 20
│     │                        15.24 Kitem/s │ 7.3 Kitem/s   │ 8.356 Kitem/s │ 8.92 Kitem/s  │         │
│     ╰─ 6                     34.12 ms      │ 67.34 ms      │ 65.15 ms      │ 60.25 ms      │ 20      │ 20
│                              17.58 Kitem/s │ 8.909 Kitem/s │ 9.208 Kitem/s │ 9.957 Kitem/s │         │
├─ 15_full_scan_aggregate                    │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     35.44 ms      │ 39.1 ms       │ 36.21 ms      │ 36.47 ms      │ 100     │ 100
│     │                        2.821 Kitem/s │ 2.557 Kitem/s │ 2.761 Kitem/s │ 2.741 Kitem/s │         │
│     ├─ 2                     36.1 ms       │ 63.34 ms      │ 37.75 ms      │ 38.69 ms      │ 100     │ 100
│     │                        5.539 Kitem/s │ 3.157 Kitem/s │ 5.297 Kitem/s │ 5.169 Kitem/s │         │
│     ├─ 3                     36.61 ms      │ 57.63 ms      │ 38.62 ms      │ 43.36 ms      │ 100     │ 100
│     │                        8.193 Kitem/s │ 5.204 Kitem/s │ 7.765 Kitem/s │ 6.918 Kitem/s │         │
│     ├─ 4                     37.12 ms      │ 81.17 ms      │ 44 ms         │ 47.98 ms      │ 100     │ 100
│     │                        10.77 Kitem/s │ 4.927 Kitem/s │ 9.089 Kitem/s │ 8.336 Kitem/s │         │
│     ├─ 5                     37.75 ms      │ 86.74 ms      │ 62.71 ms      │ 58.47 ms      │ 100     │ 100
│     │                        13.24 Kitem/s │ 5.764 Kitem/s │ 7.972 Kitem/s │ 8.551 Kitem/s │         │
│     ╰─ 6                     38.92 ms      │ 88.99 ms      │ 65.54 ms      │ 65.9 ms       │ 100     │ 100
│                              15.41 Kitem/s │ 6.742 Kitem/s │ 9.153 Kitem/s │ 9.103 Kitem/s │         │
├─ 16_insert_heavy                           │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     1.359 ms      │ 2.859 ms      │ 1.957 ms      │ 1.984 ms      │ 100     │ 100
│     │                        7.353 Mitem/s │ 3.497 Mitem/s │ 5.108 Mitem/s │ 5.038 Mitem/s │         │
│     ├─ 2                     1.733 ms      │ 3.267 ms      │ 2.478 ms      │ 2.488 ms      │ 100     │ 100
│     │                        11.54 Mitem/s │ 6.12 Mitem/s  │ 8.067 Mitem/s │ 8.035 Mitem/s │         │
│     ├─ 3                     1.979 ms      │ 3.721 ms      │ 3.052 ms      │ 2.959 ms      │ 100     │ 100
│     │                        15.15 Mitem/s │ 8.06 Mitem/s  │ 9.826 Mitem/s │ 10.13 Mitem/s │         │
│     ├─ 4                     2.475 ms      │ 4.357 ms      │ 3.445 ms      │ 3.348 ms      │ 100     │ 100
│     │                        16.16 Mitem/s │ 9.18 Mitem/s  │ 11.6 Mitem/s  │ 11.94 Mitem/s │         │
│     ├─ 5                     2.701 ms      │ 4.742 ms      │ 3.607 ms      │ 3.577 ms      │ 100     │ 100
│     │                        18.5 Mitem/s  │ 10.54 Mitem/s │ 13.85 Mitem/s │ 13.97 Mitem/s │         │
│     ╰─ 6                     2.812 ms      │ 5.278 ms      │ 4.109 ms      │ 4.031 ms      │ 100     │ 100
│                              21.33 Mitem/s │ 11.36 Mitem/s │ 14.59 Mitem/s │ 14.88 Mitem/s │         │
├─ 17_hot_spot                               │               │               │               │         │
│  ╰─ masstree24                             │               │               │               │         │
│     ├─ 1                     1.174 ms      │ 2.249 ms      │ 1.692 ms      │ 1.688 ms      │ 100     │ 100
│     │                        8.517 Mitem/s │ 4.445 Mitem/s │ 5.909 Mitem/s │ 5.924 Mitem/s │         │
│     ├─ 2                     2.118 ms      │ 5.531 ms      │ 4.76 ms       │ 4.588 ms      │ 100     │ 100
│     │                        9.44 Mitem/s  │ 3.615 Mitem/s │ 4.201 Mitem/s │ 4.358 Mitem/s │         │
│     ├─ 3                     4.233 ms      │ 7.812 ms      │ 6.684 ms      │ 6.596 ms      │ 100     │ 100
│     │                        7.086 Mitem/s │ 3.839 Mitem/s │ 4.488 Mitem/s │ 4.548 Mitem/s │         │
│     ├─ 4                     5.435 ms      │ 10.05 ms      │ 9.131 ms      │ 9.095 ms      │ 100     │ 100
│     │                        7.359 Mitem/s │ 3.977 Mitem/s │ 4.38 Mitem/s  │ 4.397 Mitem/s │         │
│     ├─ 5                     7.41 ms       │ 13.37 ms      │ 11.29 ms      │ 11.31 ms      │ 100     │ 100
│     │                        6.747 Mitem/s │ 3.739 Mitem/s │ 4.426 Mitem/s │ 4.42 Mitem/s  │         │
│     ╰─ 6                     12.21 ms      │ 16.7 ms       │ 14.22 ms      │ 14.08 ms      │ 100     │ 100
│                              4.913 Mitem/s │ 3.592 Mitem/s │ 4.217 Mitem/s │ 4.258 Mitem/s │         │
╰─ 18_split_inducing_scan                    │               │               │               │         │
   ╰─ masstree24                             │               │               │               │         │
      ├─ 3                     8.205 ms      │ 16.82 ms      │ 9.328 ms      │ 10.08 ms      │ 100     │ 100
      ├─ 4                     9.384 ms      │ 26.14 ms      │ 15.18 ms      │ 14.41 ms      │ 100     │ 100
      ├─ 5                     9.56 ms       │ 27.79 ms      │ 15.55 ms      │ 14.74 ms      │ 100     │ 100
      ╰─ 6                     11.31 ms      │ 28.57 ms      │ 16.29 ms      │ 17.1 ms       │ 100     │ 100
```
