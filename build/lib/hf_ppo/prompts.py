import textwrap


def strip_all(s: str) -> str:
    """
    Remove tabs, carriage returns, trailing and leading whitespaces.
    """
    return textwrap.dedent(s).replace('\n', ' ').strip()


def STAY_WITHIN_THE_TOKEN_LIMIT(limit: int = 1024) -> str:
    assert isinstance(limit, int), "Limit argument has wrong type."
    return strip_all(f"""
        Answer with no more than {limit} tokens. Any part of your
        response beyond this limit will be cut off.
    """)


def STAY_WITHIN_THE_TOKEN_LIMIT_TRAININIG_AWARE(limit: int = 1024) -> str:
    assert isinstance(limit, int), "Limit argument has wrong type."
    return strip_all(f"""
        You are being trained to improve your question-answering abilities.
        Each example you see has a token limit of {limit} for your answers.
        Any part of your response beyond this limit will be cut off and could
        result in a lower evaluation score. Please ensure your answer stays
        within the limit to receive accurate feedback and support the training
        process effectively.
    """)
