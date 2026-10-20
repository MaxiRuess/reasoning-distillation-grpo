import unittest

from src.reward import answers_match, binary_reward_fn, extract_answer_auto, format_reward_fn


class RewardTests(unittest.TestCase):
    def test_complete_numbers(self):
        for text, expected in [
            ("The answer is 1,234.", "1234"),
            ("The answer is 1/2.", "0.5"),
            (r"\boxed{\frac{-3}{2}}", "-1.5"),
            ("#### -1,234.5", "-1234.5"),
        ]:
            with self.subTest(text=text):
                self.assertTrue(answers_match(extract_answer_auto(text), expected))

    def test_ignore_rejected_thinking(self):
        self.assertEqual(extract_answer_auto(r"<think>Try \boxed{10}, wrong.</think> Final answer: 12"), "12")

    def test_incomplete_reasoning_has_no_answer(self):
        self.assertIsNone(extract_answer_auto("<think>Try 40 then 42"))
        self.assertEqual(binary_reward_fn(None, ["<think>Try 42"], ["42"]), [0.0])

    def test_ambiguous_expression_and_nonfinite(self):
        for text in [r"\frac{1}{2}+7", "inf", "-inf", "nan", "1e309", "1,23", "1/0"]:
            with self.subTest(text=text):
                self.assertFalse(answers_match(text, "0.5"))
                self.assertFalse(answers_match(text, text))

    def test_empty_and_broken_format_not_full_credit(self):
        for text in [r"<think></think> \boxed{0}", r"<think>work \boxed{0}", r"<think>work</think> \boxed{"]:
            self.assertLess(format_reward_fn(None, [text])[0], 1.0)
        self.assertEqual(format_reward_fn(None, [r"<think>2+2=4</think> \boxed{4}"]), [1.0])

    def test_no_arbitrary_last_number(self):
        self.assertIsNone(extract_answer_auto("I considered 4 options and checked 2 equations."))


if __name__ == "__main__":
    unittest.main()


def test_prose_and_markdown_final_answers():
    for text in ["John gets $6 in change.", "The answer is **6**.", "It takes 6 hours."]:
        assert answers_match(extract_answer_auto(text), "6")


def test_conclusive_numeric_prose_survives_explanatory_tail():
    cases = [
        ("Maya earns $27 per shift. This is the amount remaining after expenses.", "27"),
        ("The package needs two colors. This results in a total of 7 rolls. The quantities are additive.", "7"),
        ("The tank needs 12 liters for its final fill. This ensures that the requested capacity is reached.", "12"),
        ("The distance is **180 meters per week**.", "180"),
        ("There are 9 marbles left.", "9"),
        ("She pays $1,250.50 in total.", "1250.5"),
        ("The final answer is 1/2 meter. This follows from the calculation.", "0.5"),
        ("First there were 10. After subtraction, the final answer is 6.", "6"),
        ("This results in a total of 7 rolls. The answer is straightforward because the quantities are additive.", "7"),
        ("**Answer:** Ari spent **$42** on the supplies.", "42"),
        ("Thus, an age 8 years from now is 39 years old.", "39"),
        ("The original price before the 20% discount was $50. The final price is $40 after that discount.", "50"),
        ("**Answer:** The board will be 6 centimeters shorter after cutting from 1:00 PM to 2:00 PM.", "6"),
        ("**Answer:**\nThe shipment has a total of 17 boxes. The answer is straightforward.", "17"),
        ("Combining these amounts: 12 + 15 = 27.", "27"),
        ("The orchard will become profitable after 3 years. Costs are covered by then.", "3"),
    ]
    for text, expected in cases:
        assert answers_match(extract_answer_auto(text), expected), text


def test_prose_policy_rejects_distractors_and_partial_numeric_expressions():
    for text in [
        "I considered 4 options and checked 2 equations.",
        "I considered 4 options.",
        "The answer is 6 or 7.",
        "The answer is 1/2 + 7.",
        "The answer is 1,23.",
        "She has 5 apples and 7 pears.",
        "How many of the 8 boxes are needed?",
        "<think>The answer is 12.",
        "<think>The answer is 12.</think> No final answer was given.",
    ]:
        assert extract_answer_auto(text) is None, text


def test_old_scoring_versions_remain_reproducible():
    from src.reward import SCORER_VERSION, score_response

    text = "Maya earns $27 per shift. This is the amount remaining after expenses."
    assert SCORER_VERSION == "boxed-final-v1"
    assert score_response(text, "27", scorer="final-response-v6").correct
    old = score_response(text, "27", scorer="final-response-v2")
    assert old.predicted is None and old.scorer_version == "final-response-v2"


def test_declarative_prose_and_restatement_in_answer_section():
    from src.reward import score_response

    for text, answer in [
        ("The program had 93 downloads over those months.", "93"),
        ("The truck carried 42 packages.", "42"),
        ("**Answer:** Each leg measures 50 and 70 miles. The total distance covered is 120 miles.", "120"),
    ]:
        assert score_response(text, answer, scorer="final-response-v6").correct
    assert not score_response(
        "The program had 93 downloads over those months.", "93", scorer="final-response-v3"
    ).correct
    assert extract_answer_auto("I checked 4 equations.") is None
    assert extract_answer_auto("We check 4 equations.") is None
    assert extract_answer_auto("The amount must be higher than 12.") is None
    assert extract_answer_auto("At least 5 boxes are needed.") is None


def test_cardinal_word_answers_are_numeric_and_versions_are_preserved():
    from src.reward import score_response

    for text, answer in [
        ("The answer is two.", "2"),
        ("Therefore, the number of packages ordered is two.", "2"),
        ("Final answer: twenty-one boxes.", "21"),
        ("The answer is one hundred and forty-two.", "142"),
        ("Answer: negative seven.", "-7"),
        ("Answer: two million three hundred thousand and five.", "2300005"),
    ]:
        assert score_response(text, answer, scorer="final-response-v6").correct, text
    assert not score_response("The answer is two.", "2", scorer="final-response-v4").correct
    for text in ["The answer is one or two.", "The answer is one two.", "The answer is twenty thirty."]:
        assert extract_answer_auto(text) is None
    assert extract_answer_auto("There are two inputs. I cannot determine the answer.") is None


def test_asserted_value_is_separate_from_conditional_context():
    from src.reward import score_response

    for text, answer in [
        ("Annual salary is $48,000 if she works 40 weeks a year and 30 hours per week.", "48000"),
        ("The answer is 36 given 6 groups of 6 items.", "36"),
        ("Final answer: twenty-one boxes, assuming 7 boxes in each of 3 groups.", "21"),
        ("The total is 12 because there are 3 sets of 4.", "12"),
        ("Combining these: 3 * 4 = 12 given 3 sets of 4.", "12"),
        ("The program had 93 downloads over the three months.", "93"),
        ("Final answer is 7 dozen eggs, which is the amount consumed in four weeks.", "7"),
        ("She needs 20 cups to ensure that each of the three animals is fed.", "20"),
    ]:
        assert score_response(text, answer, scorer="final-response-v6").correct, text
    assert extract_answer_auto("The answer is 6 or 7 if we change the inputs.") is None
    assert extract_answer_auto("The fraction is one over two.") is None


def test_explicit_final_answer_contract_prevents_input_quantity_selection():
    from src.reward import score_response

    response = r"There are 2 blue bolts and 1 white bolt. The total is 3. \boxed{3}"
    assert score_response(response, "3").correct
    missing = score_response("There are 2 blue bolts and 1 white bolt. The total is 3.", "3")
    assert missing.predicted is None and missing.status == "missing_final_answer"
    assert score_response(r"<think>Try \boxed{2}</think> \boxed{3}", "3").correct
    assert score_response(r"\boxed{two}", "2").correct
    assert not score_response(r"\boxed{2 or 3}", "3").correct
    assert score_response(r"\boxed{2} or \boxed{3}", "3").status == "ambiguous_final_answer"
    assert not score_response(r"\boxed{1000001}", "1000000").correct
    assert not score_response(r"\boxed{0.000000001}", "0").correct
