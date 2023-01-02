from typing import Optional, List, Dict, Any

import openai

PROMPT_INTRODUCTION = """
Imagine you are experienced software engineer working in TDD regime. 
Your task is to write tests for Python code, using Pytest library (not unittest).
You are given function with the following signature: 
"""
FUNCTION_DESCRIPTION_INTRODUCTION = "Function can be described by the following: "
TEST_CASES_INTRODUCTION = "You are supposed to cover the following cases with tests:"
TEST_REQUIREMENTS_INTRODUCTION = (
    "Additionally, you are given the following instructions:"
)
REQUIREMENTS_MAPPING = {
    "gwt": "All test cases follow given when then convention and appropriate python comments "
    "should be added to test body.",
    "stc": "All test cases are denoted in separate test functions with meaningful names (do not use parametrise).",
    "fixtures": "Use fixtures wherever appropriate.",
    "typing": "Provide typing for generated test cases",
}
DEFAULT_COMPLETION_CONFIG = {
    "model": "text-davinci-003",
    "max_tokens": 2048,
}


def generate_tests(
    tested_function_signature: str,
    tested_function_description: str,
    test_cases: List[str],
    tests_specifics: Optional[List[str]] = None,
    completion_config: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = _prepare_prompt(
        tested_function_signature=tested_function_signature,
        tested_function_description=tested_function_description,
        test_cases=test_cases,
        tests_specifics=tests_specifics,
    )
    if completion_config is None:
        completion_config = DEFAULT_COMPLETION_CONFIG
    return openai.Completion.create(prompt=prompt, **completion_config)["choices"][0][
        "text"
    ]


def _prepare_prompt(
    tested_function_signature: str,
    tested_function_description: str,
    test_cases: List[str],
    tests_specifics: Optional[List[str]],
) -> str:
    if len(test_cases) == 0:
        raise ValueError("Cannot generate tests without test cases description.")
    if tests_specifics is None:
        tests_specifics = []
    prompt = (
        f"{PROMPT_INTRODUCTION}`{tested_function_signature}`\n"
        f"{FUNCTION_DESCRIPTION_INTRODUCTION}{tested_function_description}\n"
        f"{TEST_CASES_INTRODUCTION}\n"
    )
    for test_case in test_cases:
        prompt = f"{prompt}{test_case}\n"
    if len(tests_specifics) == 0:
        return prompt
    prompt = f"{prompt}{TEST_REQUIREMENTS_INTRODUCTION}\n"
    for tests_specific in tests_specifics:
        if tests_specific.lower() in REQUIREMENTS_MAPPING:
            tests_specific = REQUIREMENTS_MAPPING[tests_specific.lower()]
        prompt = f"{prompt}{tests_specific}\n"
    return prompt
