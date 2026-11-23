# Licence for trained models

**Models trained in this repository from BoardNet v4 onward are for
noncommercial use only, under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).**

The code and documentation stay under [AGPL-3.0-only](LICENSE).

## Why the models have a different licence

The project set out to release its models under the same licence as its code.
That changed on 23 November 2026:

- Without the noncommercial Lichess piece sets, the reader got 388 of 850
  screen boards drawn in those sets right. With them it gets 800.
- Those sets are licensed CC BY-NC-SA, which forbids commercial use and
  requires works made from them to carry the same terms.
- Whether trained weights count as a work made from their training images is
  not settled. This project treats them as if they do.

The sets are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Status

- **No model files are published.** This licence states the terms they will
  carry if they are.
- **Review is not finished.** The training data also includes piece sets
  under GPL and AGPL terms, and photographs of diagrams in copyrighted books.
  Whether one licence can properly cover weights trained on all of these is
  part of the review.
- **Inclusion in Mobile Maia is pending** that review.
- **A model with open terms can still be trained.** `python -m
  train.make_crops --sets open` generates training data without the
  noncommercial sets.
