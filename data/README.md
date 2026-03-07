IMDb sentiment dataset (prepared for labs)

Source:
- Large Movie Review Dataset (aclImdb v1.0), Maas et al. (2011)
  https://ai.stanford.edu/~amaas/data/sentiment/

Files:
- train.tsv : label<TAB>text  (balanced)
- dev.tsv   : label<TAB>text  (balanced, sampled from original train)
- test.tsv  : label<TAB>text  (original IMDb test split)
- toy.tsv   : tiny debug set
- lexicon_pos.txt / lexicon_neg.txt : small word lists for the lexicon baseline

Labels:
- 1 = positive
- 0 = negative

Preprocessing applied here:
- HTML <br /> replaced by spaces
- whitespace normalized
- each review truncated to first 60 words (to make texts short)

Splits:
- dev: 2500 pos + 2500 neg (sampled from original train, seed=13)
- train: remaining examples from original train
- test: original IMDb test split

Note:
This dataset contains IMDb-sourced text. Ensure your use/distribution is consistent with IMDb terms (non-commercial use for teaching).
