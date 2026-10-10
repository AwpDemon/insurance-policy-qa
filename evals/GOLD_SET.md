# Gold set: how it was built

`gold_set.yaml` holds 120 questions: 105 answerable and 15 unanswerable. Each answerable item has
a reference answer, the key facts a correct answer must contain, and the clause ids that support it.

## Authoring criteria

1. **Written from the source.** Every answerable item was written while reading the cited clause in
   the FEMA PDF. The reference answer paraphrases that clause. Nothing comes from general insurance
   knowledge.
2. **Realistic phrasing.** Questions are worded the way a policyholder, agent, or adjuster would
   ask them ("My car was flooded in my driveway..."), not copied from the clause. That keeps
   lexical overlap low, so BM25 doesn't get an unfair advantage.
3. **Evidence at clause level.** `supporting` is a list of evidence *groups*. A group is satisfied
   when any one of its acceptable clause ids is retrieved. Alternatives are listed when the same
   rule appears word-for-word in several forms (for example, Suit Against Us is DF VII.O, GP VII.O,
   and RCBAP VIII.O) or is restated in a guide. A question that needs two separate facts has two
   groups (for example q076, which compares the GP and Dwelling special limits).
4. **Key facts are regexes.** Each one targets the fact that decides correctness, such as a dollar
   amount, a number of days, a percentage, a yes/no, or a named concept. `validate_gold()` checks
   that every regex matches its own reference answer, and the test suite runs that check.
5. **Coverage of the question types an insurance professional cares about.** The set covers the
   insuring agreement and definitions, property covered and not covered, exclusions (including
   exclusion-with-exception wording such as sewer backup "unless there is a flood in the area"),
   deductibles, conditions (proof of loss, appraisal, suit limitation), loss settlement (replacement
   cost, ACV, special loss settlement, coinsurance), comparisons across forms, and claims process
   from the consumer guides.
6. **Trick / false-premise items (7).** These questions carry a false assumption, such as a $5,000
   jewelry limit, a two-year suit window, a "valued policy" premise, or the General Property
   Form's $10,000 pollution cover attributed to the Dwelling Form. The correct behaviour is to
   correct the premise from the documents, not to refuse.
7. **Unanswerable items (15).** These are realistic questions the corpus does not answer: premiums,
   CRS discounts, WYO carriers, agent commissions, an HO-3 form, auto deductibles, appeal decision
   times. The hardest ones look answerable, and a model would know the answer from training data:
   the dollar maximum for contents (the policy text never states it), the GP building maximum, and
   the new-policy waiting period. For these the only correct output is "Not covered in the
   documents." Answering from memory counts as a failure, even when the remembered fact is true in
   the real world. The system's contract is to stay grounded in the documents.

## Known limitations of the gold set

- One author wrote it, and no second annotator checked it. Inter-annotator agreement was not
  measured. Ali should spot-check items before relying on the numbers.
- Regex key facts can be too strict (a correct paraphrase that avoids the number fails) or too
  loose (a yes/no regex can match a hedged answer). `evals/results/gen_*.jsonl` stores every answer,
  so you can audit any score, and the README failure analysis lists the misgrades found by hand.
- q120 (waiting period) is a deliberately hard negative. The Claims Handbook says policy *changes*
  "can take 30 days," but nothing in the corpus states the waiting period for a new policy.

## Held-out set

`holdout_set.yaml` has 25 more items (20 answerable, 5 unanswerable). They were written with the same criteria *after* prompt v2 had been designed from the gold-set failures, and *before* any system output on them was seen. They exist only to check that the v1-to-v2 prompt change generalises (see README). Do not tune against them. If you change the system, write a fresh held-out set.
