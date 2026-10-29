from dataclasses import dataclass, field


@dataclass
class Failure:
    rule: str  # human-readable rule name
    column: str | None
    detail: str
    count: int | None = None  # offending rows when known; never the values (LGPD)


@dataclass
class ContractResult:
    tool: str
    blocking: list[Failure] = field(default_factory=list)
    warnings: list[Failure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blocking
