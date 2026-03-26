# judging-judges

When is a judge a good judge for Danish -- a work in progress!


# Usage

```bash
uv run python evaluate_llm_judge.py
```
which results in a CSV file with the judge's decisions and reasoning, as well as an overall accuracy score and breakdown by error type:
```
Accuracy: 80.0%
source
MT    0.8
Name: is_correct, dtype: float64
error_types
form                                           0.500000
mistakes                                       1.000000
other                                          1.000000
particles                                      1.000000
word choice                                    0.666667
word choice, SB congruence, missing element    1.000000
word choice, other                             1.000000
Name: is_correct, dtype: float64
```