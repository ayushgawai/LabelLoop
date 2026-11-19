from dataclasses import dataclass, field


@dataclass
class Memory:
    seen: dict = field(default_factory=dict)      # code -> zone
    sorted: set = field(default_factory=set)
    rejected_count: int = 0
    current_code: str | None = None
    unsorted_estimate: int | None = None          # from the agent's survey
    bin_side: dict | None = None
