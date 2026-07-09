from tqdm import tqdm
import re
import pandas as pd
import ast
from typing import List, Optional, Iterator, Tuple
from math import ceil

def extract_list(text: str) -> List[str]:
    """
    Extracts the first list-like structure from a string.
    
    Args:
        text (str): The input string containing a list-like structure.
        
    Returns:
        List[str]: The extracted list.
    """
    # Use regex to find the first occurrence of a list-like structure
    match = re.search(r"\[[^\[\]]+\]", text, re.DOTALL)
    if match:
        return ast.literal_eval(match.group(0))
    return []
    


def iter_windows(
    n: int,
    window_size: int,
    stride: int,
    verbose: bool = False,
    desc: Optional[str] = None
) -> Iterator[Tuple[int, int, int]]:
    """
    Iterates over windows of a given size and stride, in reverse order.

    Args:
        n (int): The total number of elements.
        window_size (int): The size of each window.
        stride (int): The stride for moving the window.
        verbose (bool): If True, show progress bar.
        desc (str): Description for the progress bar.

    Yields:
        Tuple[int, int, int]: Start index, end index, and window length.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")
    if stride <= 0:
        raise ValueError("Stride must be greater than 0.")
    if stride > window_size:
        raise ValueError("Stride must be less than or equal to window size.")
    """
    # Compute the last full‐window start
    max_start = ((n - window_size) // stride) * stride

    if window_size > n:
        # entire input fits in one window
        yield 0, n, n
    else:
        for start_idx in tqdm(
            range(max_start, -1, -stride),
            desc=desc,
            disable=not verbose,
            unit="window"
        ):
            end_idx = start_idx + window_size
            # if we somehow overshoot (shouldn't), cap at n
            if end_idx > n:
                end_idx = n
            window_len = end_idx - start_idx

            # always include the very first window, otherwise only if it’s not just a tiny reminder
            if start_idx == 0 or window_len > stride:
                yield start_idx, end_idx, window_len
    """
    
    num_windows = ceil((n - window_size) / stride) + 1
    starts = [i * stride for i in range(num_windows)]
    starts = [s for s in starts if s < n]
    
    for start_idx in tqdm(starts[::-1], desc=desc, disable=not verbose, unit="window"):
        end_idx = min(start_idx + window_size, n)
        window_len = end_idx - start_idx
        yield start_idx, end_idx, window_len

def save_nuggets(nuggets: pd.DataFrame, file: str) -> None:
    """
    Save nuggets to a file in TSV format.

    Args:
        nuggets (pd.DataFrame): DataFrame containing nuggets.
        file (str): Path to the output file.
    """
    essential = ["qid", "nugget_id", "nugget"]
    optional = ["importance", "assignment"]
    columns = nuggets.columns
    if any([x not in columns for x in essential]):
        raise ValueError(
            "Require at least {essential} columns to save, found {columns}"
        )

    for c in optional:
        if c not in columns:
            nuggets[c] = -1

    nuggets = nuggets[essential + optional]
    nuggets.to_csv(file, sep="\t", index=False, header=False)

def load_nuggets(file: str) -> pd.DataFrame:
    """
    Load nuggets from a TSV file.

    Args:
        file (str): Path to the input file.

    Returns:
        pd.DataFrame: DataFrame containing nuggets.
    """
    essential = ["qid", "nugget_id", "nugget", "importance", "assignment"]
    nuggets = pd.read_csv(file, sep="\t", index_col=False, names=essential)

    return nuggets


__all__ = ["extract_list", "iter_windows", "save_nuggets", "load_nuggets"]
