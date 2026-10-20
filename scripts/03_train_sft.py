"""Train any configured SFT condition, including re_distill."""

from src.cli import training_main

if __name__ == "__main__":
    training_main("sft")
