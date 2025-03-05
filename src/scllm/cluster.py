import random


def get_quote() -> dict:
    """
    Test function to check that tests are working
    """

    quotes = [
        {
            "quote": "A long descriptive name is better than a short "
            "enigmatic name. A long descriptive name is better "
            "than a long descriptive comment.",
            "author": "Robert C. Martin",
        },
        {
            "quote": "You should name a variable using the same "
            "care with which you name a first-born child.",
            "author": "Robert C. Martin",
        },
        {
            "quote": "Any fool can write code that a computer "
            "can understand. Good programmers write code"
            " that humans can understand.",
            "author": "Martin Fowler",
        },
    ]

    return quotes[random.randint(0, len(quotes) - 1)]
