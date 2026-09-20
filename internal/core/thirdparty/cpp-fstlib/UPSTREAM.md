# cpp-fstlib

Source: https://github.com/yhirose/cpp-fstlib
Pinned commit: 7b5a8f93105ddfdbc278613435c1230c17e084b6
License: MIT (see LICENSE). fstlib.h is copied verbatim, without local patches.

C++17 minimal acyclic subsequential transducer, string -> uint32_t map.
The PoC adapter owns serialized bytes or an mmap and keeps them alive for the
non-owning reader. The format is private to this pinned dependency version.
Only exact matching and dictionary enumeration are integrated; upstream fuzzy
search is not claimed to implement Milvus FUZZY_MATCH semantics.
