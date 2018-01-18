"""Random Sampling
"""
from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip


class RandomSampling(QueryStrategy):

    r"""Random sampling

    This class implements the random query strategy. A random entry from the
    unlabeled pool is returned for each query.

    Parameters
    ----------
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of declaring a RandomSampling query_strategy object:

    .. code-block:: python

       from libact.query_strategies import RandomSampling

       qs = RandomSampling(
                dataset, # Dataset object
            )
    """

    def __init__(self, dataset, **kwargs):
        super(RandomSampling, self).__init__(dataset, **kwargs)

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)
        self.update_scores_list()

    def retrieve_score_list(self):
        unlabeled_entry_ids, _ = zip(*self.dataset.get_unlabeled_entries())
        # random score between (0.0, 1.0) for all unlabeled instances
        return dict(zip(unlabeled_entry_ids, self.random_state_.rand(len(unlabeled_entry_ids))))
