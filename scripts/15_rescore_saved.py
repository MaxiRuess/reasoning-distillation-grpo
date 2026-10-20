"""CPU-only, versioned rescoring of saved generations; never overwrites source results."""

import argparse
import json

from src.rescoring import rescore_saved_evaluation
from src.reward import SCORER_VERSION, SCORER_VERSIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", default="outputs/rescored")
    parser.add_argument("--scorer", choices=SCORER_VERSIONS, default=SCORER_VERSION)
    args = parser.parse_args()
    print(json.dumps(rescore_saved_evaluation(args.source, args.output_root, args.scorer), indent=2))
