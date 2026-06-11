import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nemotron_baseline.numeric_equation_loss_weights import build_char_weights  # noqa: E402


class NumericEquationLossWeightTests(unittest.TestCase):
    def weights_for(self, text: str) -> list[float]:
        return build_char_weights(text, high=2.0, base=1.0)

    def assert_span_weight(
        self,
        text: str,
        weights: list[float],
        span: str,
        expected: float,
        *,
        occurrence: int = 0,
    ) -> None:
        start = -1
        search_from = 0
        for _ in range(occurrence + 1):
            start = text.index(span, search_from)
            search_from = start + len(span)
        end = start + len(span)
        self.assertTrue(
            all(weight == expected for weight in weights[start:end]),
            msg=f"span {span!r} was not all weight {expected}",
        )

    def assert_span_base(
        self, text: str, weights: list[float], span: str, *, occurrence: int = 0
    ) -> None:
        self.assert_span_weight(text, weights, span, 1.0, occurrence=occurrence)

    def assert_span_high(
        self, text: str, weights: list[float], span: str, *, occurrence: int = 0
    ) -> None:
        self.assert_span_weight(text, weights, span, 2.0, occurrence=occurrence)

    def assert_span_critical(
        self, text: str, weights: list[float], span: str, *, occurrence: int = 0
    ) -> None:
        self.assert_span_weight(text, weights, span, 3.0, occurrence=occurrence)

    def test_common_scaffold_is_demoted_but_decision_payloads_are_weighted(self) -> None:
        text = """Same operator RHS values are 6 and 11
The RHS values mix length 1 and 2, so use subtraction or modular
Try BA_DC first
Try BA_DC with x-y for operator -
The current format is BA_DC|x-y
Example 34-73 = 6
BA DC BA-DC rev plain op_prefix
43 37 6 6 6 -6
Match
Common
rev
abs_rev

The format BA_DC|x-y|common supports all two same operator examples
Apply format BA_DC|x-y|common to the query

Query
12-34
BA DC BA-DC rev abs_rev
21 43 -22 -22 22
All common output formats agree on 22

Answer: \\boxed{22}"""
        weights = self.weights_for(text)

        self.assert_span_base(text, weights, "Same operator RHS values are 6 and 11")
        self.assert_span_base(text, weights, "The RHS values ")
        self.assert_span_high(text, weights, "mix length 1 and 2, so use subtraction or modular")
        self.assert_span_base(text, weights, "Try BA_DC first")
        self.assert_span_base(text, weights, "Try BA_DC with x-y for operator -")
        self.assert_span_base(text, weights, "The current format is BA_DC|x-y")
        self.assert_span_high(text, weights, "43 37 6 6 6 -6")
        self.assert_span_base(text, weights, "Match")
        self.assert_span_base(text, weights, "Common")

        common_list_start = text.index("Common\nrev\nabs_rev") + len("Common\n")
        self.assertTrue(all(weight == 2.0 for weight in weights[common_list_start:common_list_start + 3]))
        abs_rev_start = common_list_start + len("rev\n")
        self.assertTrue(all(weight == 2.0 for weight in weights[abs_rev_start:abs_rev_start + 7]))
        self.assert_span_high(text, weights, "BA_DC|x-y|common", occurrence=0)
        self.assert_span_high(text, weights, "BA_DC|x-y|common", occurrence=1)
        self.assert_span_critical(text, weights, "21 43 -22 -22 22")
        agree_value_start = text.index("All common output formats agree on 22") + len(
            "All common output formats agree on "
        )
        self.assertTrue(all(weight == 3.0 for weight in weights[agree_value_start:agree_value_start + 2]))
        self.assert_span_base(text, weights, "\\boxed{22}")

    def test_operator_absence_weights_only_query_mapping_and_vote_outputs(self) -> None:
        text = """Use symbol mapping for query operator !
+ -> x+y no
! -> x+y PASS
) -> x+y no

Candidate x-y
Query
76!23
BA DC BA-DC rev abs_rev
67 32 35 53 53
Output votes
53 has 2 votes
Highest vote count
2
Choose operator candidate with highest output agreement vote count, if tie choose the first candidate
x-y"""
        weights = self.weights_for(text)

        self.assert_span_base(text, weights, "Use symbol mapping for query operator !")
        self.assert_span_base(text, weights, "+ -> x+y no")
        self.assert_span_high(text, weights, "! -> x+y PASS")
        self.assert_span_base(text, weights, ") -> x+y no")

        self.assert_span_base(text, weights, "Candidate ")
        self.assert_span_high(text, weights, "x-y")
        self.assert_span_high(text, weights, "67 32 35 53 53")
        self.assert_span_critical(text, weights, "53 has 2 votes")
        vote_count_start = text.index("Highest vote count\n2") + len("Highest vote count\n")
        self.assertEqual(weights[vote_count_start], 3.0)
        chosen_start = text.rindex("\nx-y") + 1
        self.assertTrue(all(weight == 2.0 for weight in weights[chosen_start:chosen_start + 3]))


if __name__ == "__main__":
    unittest.main()
