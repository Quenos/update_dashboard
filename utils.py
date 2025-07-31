from typing import Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar('T')


def find_closest(items: List[T], target: float, key: Callable[[T], float]) -> Tuple[Optional[T], Optional[T]]:
    # Initialize variables to store the closest items below and above the target value
    closest_below = None
    closest_above = None
    min_diff_below = float('inf')
    min_diff_above = float('inf')
    
    # Iterate through the items to find the closest ones
    for item in items:
        key_value = float(key(item))
        diff = key_value - target
        
        if diff == 0:
            # If the item is exactly the target, return it as both below and above
            return item, item
        elif diff < 0 and abs(diff) < min_diff_below:
            min_diff_below = abs(diff)
            closest_below = item
        elif diff > 0 and abs(diff) < min_diff_above:
            min_diff_above = abs(diff)
            closest_above = item
    
    return closest_below, closest_above
