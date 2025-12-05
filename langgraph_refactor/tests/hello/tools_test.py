import pytest
from hello.tools import get_local_news, get_user_location


class TestGetLocalNews:
    def test_returns_string(self):
        result = get_local_news.run("São Paulo")
        assert isinstance(result, str)

    def test_contains_expected_phrase(self):
        result = get_local_news.run("Rio de Janeiro")
        assert "The fly won the olimpic medal" in result

    @pytest.mark.parametrize("city", ["São Paulo", "Rio de Janeiro", "New York", "Tokyo"])
    def test_works_for_any_city_input(self, city):
        """Even though we ignore the city, ensure the tool doesn't break for different values."""
        result = get_local_news.run(city)
        assert isinstance(result, str)
        assert "The fly" in result


class TestGetUserLocation:
    def test_get_user_location_brazil(self):
        result = get_user_location.run("Oi")
        assert result == "Brazil"

    def test_get_user_location_usa_default_for_non_oi(self):
        result = get_user_location.run("Hello")
        assert result == "USA"

    @pytest.mark.parametrize(
        "greeting, expected",
        [
            ("Oi", "Brazil"),      # exact match
            ("Hello", "USA"),      # anything else => USA
            ("oi", "USA"),         # case-sensitive behavior
            ("", "USA"),           # empty string
            ("Bonjour", "USA"),    # other languages
        ],
    )
    def test_get_user_location_various_inputs(self, greeting, expected):
        result = get_user_location.run(greeting)
        assert result == expected
