# Arithmetic corpus run 20260815T221650Z

This is an operational run record, not a canonical publication manifest.
Generated JSON files remain ignored mutable outputs until their mathematical
claims and immutable hashes are deliberately promoted into
`results/manifest.json`.

## Deterministic layered constructions

- `layered-exact-300v-20260815T221650Z.json`
  - 2/2 strict types, each with V/E/F = 300/894/596
  - both analytic finite triangulations matched Qhull
  - SHA-256 `10847ff2cbbd054d57ebe3ede0a607edc4e30f91df4c131849a51ddd1c457955`
- `layered-exact-302v-20260815T221650Z.json`
  - 15/15 strict, pairwise graph-nonisomorphic factor types
  - maximum coordinate shear `5.417888360170764e-14`
  - Qhull matched 12/15; the remaining three are deliberately ill-conditioned
    stress cases whose analytic faces remain authoritative
  - SHA-256 `500c07324f8bfdd247d4ed7ccf34a47dbe16271d717e2d81cac4392ce2c6d642`

## Legalized 300-vertex campaign

- `legalized-300v-20260815T221650Z.json`
  - bounded at 32 accepted types, 64 attempts, 300 seconds, 6 workers
  - completed with 32 accepted among 40 attempts in 234.17 seconds
  - 32 distinct pynauty topology certificates; no duplicates or worker errors
  - every accepted boundary has V/E/F = 300/894/596
  - maximum coordinate shear `1.8105605903429023e-10`
  - maximum corner-angle error `1.48319578840983e-10` radians
  - maximum developing-map closure error `1.798468832990167e-12`
  - minimum interior and infinity-fan margins `5.694359181416253e-4`
    and `1.5202065743821835e-1` radians, respectively
  - flip count min/median/max = 5/11/21
  - 7 optimizer nonconvergences and 1 insufficient-margin rejection were
    checkpointed rather than silently discarded
  - SHA-256 `29bc7c6aae7b5912fe69dda5108fd602d1b5a362e90fa08191fa263498e9070f`

The exact integer certificate applies to the declared abstract zero-shear
structure. Layered coordinates additionally have a closed-form zero-shear
identity. Legalized coordinates are high-accuracy numerical witnesses and
still need interval/high-precision certification before theorem-level
coordinate claims in a paper.

All three files passed the `ideal-poly-arithmetic-corpus/1` JSON schema and
semantic validator. The codebase gate at launch was 308 passing tests; the
only warnings were the existing adaptive-quadrature warnings in the legacy
Lobachevsky reference path.
