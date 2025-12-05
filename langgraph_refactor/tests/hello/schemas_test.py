import pytest
from pydantic import ValidationError
from hello.schemas import GreetingsResponse


def test_valid_greetings_response():
    data = {
        "greeting": "Hello",
        "news": "The fly won the olimpic medal",
        "chat": "How can I help you?"
    }
    response = GreetingsResponse(**data)

    assert response.greeting == data["greeting"]
    assert response.news == data["news"]
    assert response.chat == data["chat"]


def test_missing_required_fields():
    """All fields are required, missing any should raise an error."""
    with pytest.raises(ValidationError):
        GreetingsResponse(greeting="Hi")  # missing news and chat


@pytest.mark.parametrize(
    "field, value",
    [
        ("greeting", 123),  # not a string
        ("news", None),     # not allowed
        ("chat", True),     # boolean instead of text
    ]
)
def test_invalid_field_types(field, value):
    """Ensure type enforcement for each field."""
    data = {
        "greeting": "Hi",
        "news": "News here",
        "chat": "Chat here"
    }
    data[field] = value
    with pytest.raises(ValidationError):
        GreetingsResponse(**data)



def test_extra_fields_are_ignored():
    """Extra fields should be accepted but ignored by the model."""
    response = GreetingsResponse(
        greeting="Hello",
        news="Something happened",
        chat="Let's talk",
        extra_field="Not allowed",
    )

    # Core fields are still there
    assert response.greeting == "Hello"
    assert response.news == "Something happened"
    assert response.chat == "Let's talk"

    # Extra field should not be set as an attribute
    assert not hasattr(response, "extra_field")
