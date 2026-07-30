import textwrap
import pytest
from src.scenarios import load_scenarios, Scenario

def test_load_scenarios_computes_annual_cost_and_placeholders(tmp_path):
    yaml_content = textwrap.dedent("""
        single_taipei:
          monthly_costs_ntd:
            rent: 13000
            food: 9000
          placeholders:
            current_capital_ntd: 500000
            current_age: 40
            monthly_work_income_ntd: 40000
    """)
    config_path = tmp_path / "scenarios.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    scenarios = load_scenarios(str(config_path))

    assert set(scenarios.keys()) == {"single_taipei"}
    s = scenarios["single_taipei"]
    assert isinstance(s, Scenario)
    assert s.name == "single_taipei"
    assert s.annual_cost == (13000 + 9000) * 12
    assert s.current_capital == 500000
    assert s.current_age == 40
    assert s.monthly_work_income == 40000


def test_load_scenarios_missing_field_raises(tmp_path):
    yaml_content = textwrap.dedent("""
        single_taipei:
          monthly_costs_ntd:
            rent: 13000
          placeholders:
            current_age: 40
            monthly_work_income_ntd: 40000
    """)
    config_path = tmp_path / "scenarios.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="current_capital_ntd"):
        load_scenarios(str(config_path))
