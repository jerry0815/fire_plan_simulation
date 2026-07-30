from dataclasses import dataclass

import yaml


@dataclass
class Scenario:
    name: str
    annual_cost: float
    current_capital: float
    current_age: int
    monthly_work_income: float


def load_scenarios(path: str) -> dict[str, Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    scenarios: dict[str, Scenario] = {}
    for key, cfg in raw.items():
        monthly_costs = cfg.get("monthly_costs_ntd")
        if monthly_costs is None:
            raise ValueError(f"scenario '{key}' missing 'monthly_costs_ntd'")

        placeholders = cfg.get("placeholders")
        if placeholders is None:
            raise ValueError(f"scenario '{key}' missing 'placeholders'")

        for field in ("current_capital_ntd", "current_age", "monthly_work_income_ntd"):
            if field not in placeholders:
                raise ValueError(f"scenario '{key}' missing placeholders.{field}")

        annual_cost = sum(monthly_costs.values()) * 12
        scenarios[key] = Scenario(
            name=key,
            annual_cost=annual_cost,
            current_capital=placeholders["current_capital_ntd"],
            current_age=placeholders["current_age"],
            monthly_work_income=placeholders["monthly_work_income_ntd"],
        )
    return scenarios
