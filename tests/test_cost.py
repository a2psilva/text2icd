from pathlib import Path
from types import SimpleNamespace

import pytest

from text2icd.llm.cost import (
    BudgetExceededError,
    CostMeter,
    Price,
    PriceError,
    Usage,
    load_prices,
)

PRICE = Price(
    provider="test",
    id="small",
    input=1.0,
    output=5.0,
    cache_read=0.1,
    cache_write_5m=1.25,
    cache_write_1h=2.0,
    long_context_above=1000,
    long_input=2.0,
    long_output=10.0,
    batch_discount=0.5,
)


def test_usage_from_a_response_object_or_dict() -> None:
    response = SimpleNamespace(
        input_tokens=120,
        output_tokens=8,
        cache_read_input_tokens=600,
        cache_creation_input_tokens=0,
        cache_creation=None,
    )
    usage = Usage.from_response(response)
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens) == (120, 8, 600)
    assert usage.prompt_tokens == 720
    as_dict = Usage.from_response(
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_creation_input_tokens": 700,
            "cache_creation": {"ephemeral_1h_input_tokens": 700, "ephemeral_5m_input_tokens": 0},
        }
    )
    assert as_dict.cache_write_1h_tokens == 700
    assert as_dict.as_dict() == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_creation_input_tokens": 700,
        "cache_write_1h_tokens": 700,
    }
    assert usage + as_dict == Usage(130, 10, 600, 700, 700)
    assert Usage.from_response(SimpleNamespace()) == Usage()


def test_cost() -> None:
    # 1,000 input at $1, 100 output at $5, 500 cache reads at $0.10 (per million tokens).
    usage = Usage(input_tokens=400, output_tokens=100, cache_read_input_tokens=500)
    assert PRICE.cost(usage) == pytest.approx((400 * 1.0 + 100 * 5.0 + 500 * 0.1) / 1e6)
    writes = Usage(input_tokens=100, cache_creation_input_tokens=800, cache_write_1h_tokens=300)
    assert PRICE.cost(writes) == pytest.approx((100 + 500 * 1.25 + 300 * 2.0) / 1e6)
    assert PRICE.cost(writes, batch=True) == pytest.approx(PRICE.cost(writes) / 2)


def test_long_context_prices() -> None:
    usage = Usage(input_tokens=900, output_tokens=10, cache_read_input_tokens=200)
    # 1,100 prompt tokens is over the 1,000 threshold: input and output double, cache scales too.
    assert PRICE.cost(usage) == pytest.approx((900 * 2.0 + 10 * 10.0 + 200 * 0.2) / 1e6)


def test_meter_and_budget() -> None:
    meter = CostMeter(PRICE, budget_usd=0.001)
    meter.check()
    first = meter.charge(Usage(input_tokens=500, output_tokens=50))
    assert first == pytest.approx(0.00075)
    meter.check()
    meter.charge(Usage(input_tokens=500))
    assert meter.calls == 2
    assert meter.usage == Usage(input_tokens=1000, output_tokens=50)
    with pytest.raises(BudgetExceededError, match=r"budget of \$0.00 spent"):
        meter.check()


def test_shipped_table() -> None:
    table = load_prices()
    haiku = table.get("anthropic", "claude-haiku-5-5")
    assert (haiku.input, haiku.output, haiku.cache_read) == (0.10, 0.50, 0.01)
    assert table.get("anthropic", "claude-opus-5-5").output == 20.0
    with pytest.raises(PriceError, match="no price for gpt on anthropic"):
        table.get("anthropic", "gpt")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("schema = 2\nchecked = 2026-11-01\n", "unsupported price table schema 2"),
        ("schema = 1\n", "needs a `checked` date"),
        (
            "schema = 1\nchecked = 2026-11-01\n[[model]]\nprovider = 'a'\nid = 'm'\ninput = 1\n",
            "bad price entry 'm'",
        ),
        (
            "schema = 1\nchecked = 2026-11-01\n[[model]]\nprovider = 'a'\nid = 'm'\ninput = -1\n"
            "output = 1\ncache_read = 0\ncache_write_5m = 0\ncache_write_1h = 0\n",
            "must be numbers of 0 or more",
        ),
    ],
)
def test_bad_tables(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(text)
    with pytest.raises(PriceError, match=message):
        load_prices(path)


def test_duplicate_models(tmp_path: Path) -> None:
    entry = (
        "[[model]]\nprovider = 'a'\nid = 'm'\ninput = 1\noutput = 1\ncache_read = 0\n"
        "cache_write_5m = 0\ncache_write_1h = 0\n"
    )
    path = tmp_path / "prices.toml"
    path.write_text("schema = 1\nchecked = 2026-11-01\n" + entry + entry)
    with pytest.raises(PriceError, match="listed twice"):
        load_prices(path)
