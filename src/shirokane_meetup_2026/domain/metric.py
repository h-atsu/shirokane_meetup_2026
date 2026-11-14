from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Metrics:
    objective_value: float = field(
        metadata={"description": "配信スコア合計から不足ペナルティを引いた目的関数値"}
    )
    total_score: float = field(
        metadata={"description": "選択されたユーザー・コンテンツの配信スコア合計"}
    )
    total_penalty: float = field(
        metadata={"description": "配信不足回数にペナルティ係数を掛けた合計"}
    )
    total_shortage: int = field(
        metadata={"description": "全コンテンツの最低配信回数に対する不足回数合計"}
    )
    num_contents_below_target: int = field(
        metadata={"description": "実際の配信回数が最低配信回数を下回ったコンテンツ数"}
    )
    total_deliveries: int = field(
        metadata={"description": "全ユーザー・コンテンツに対する配信件数の合計"}
    )
    delivery_count_cv: float = field(
        metadata={
            "description": (
                "コンテンツ別配信回数の標準偏差を平均で割った値。"
                "0に近いほど配信回数が均等"
            )
        }
    )
    num_capacity_violations: int = field(
        metadata={
            "description": "表示容量制約に違反したユーザー数(実行可能解であれば0の想定)"
        }
    )
