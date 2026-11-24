"""Phase 0: verify GPU, CUDA arch (sm_120) and package versions. Writes results/env.json.

Ported from ~/vggtProject (vggt-sparse-recon): scripts/00_env_check.py.
"""

import sys

from countonce.env_check import main

if __name__ == "__main__":
    sys.exit(main())
