# =======この例では型ヒントを使っているが，実際には型チェックは行われない=======

# price: int = 100
# tax: float = 1.1

# def calculate_price_including_tax(price: int, tax: float) -> int:
#     return int(price * tax)

# if __name__ == "__main__":
#     print(f'{calculate_price_including_tax(price, tax)}円')
# ====================================================================

# 以下も同様に型ヒントを使っているが，実際には型チェックは行われない
from typing import List, Dict

sample_list: List[int] = [1.1, 2.2, 3.3]
sample_dict: Dict[str, str] = {'username': 'haruto'}
