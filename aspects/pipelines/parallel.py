import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, Callable, List

from tqdm import tqdm


def parallelized_job(
        elements: Sequence,
        fn: Callable,
        desc: str = 'Running in parallel',
        batch_size: int = 1000,
        jobs: int = None,
) -> List:
    jobs = jobs or multiprocessing.cpu_count()
    elements = list(elements)

    with ProcessPoolExecutor(jobs) as pool:
        return list(
            tqdm(
                pool.map(fn, elements, chunksize=batch_size),
                total=len(elements),
                desc=desc
            )
        )
