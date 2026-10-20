# GitHub publication

The reviewed paper and supporting results are published through the repository's existing GitHub Pages configuration: the root of the `main` branch. The paper URL remains:

https://maxiruess.github.io/reasoning-distillation-grpo/assets/paper.pdf

The publication includes the current paper source, generated result fragments, figure, frozen protocols, saved evaluations, analysis and corrected experiment code. Trained model weights, credentials, caches and local working/build directories are excluded. Upstream model and dataset terms remain applicable.

The completed follow-up's public `execution-ledger.json` is a byte-for-byte copy of its completed local ledger. It supplies the saved runtime accounting needed to rebuild the paper. The renderer uses it when the ignored local `outputs/` ledger is absent. This fallback launches no cloud calls and changes no scientific endpoint, generation or score.

A clean copy containing only publication files successfully regenerates the paper without private `outputs/` files. Its extracted PDF text and figure match the reviewed version, with no final LaTeX warnings. The actual published PDF preserves the reviewed file's bytes rather than replacing it with the verification rebuild. Full local verification passes 84 CPU tests plus 11 subtests; scoped Ruff and whitespace checks pass. A scan for common credential-token patterns found no matches in the selected files.

The original experiment and manuscript closing records retain their historical hashes. This publication adds the build fallback and public links; it does not rewrite those records or run further GPU experiments. The main numerical 8K difference remains −1.30 points (conditional interval −5.73 to +2.86); the strict secondary difference is +2.60 (−2.60 to +8.07). Neither establishes an 8K advantage.
