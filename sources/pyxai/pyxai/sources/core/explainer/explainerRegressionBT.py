import c_explainer
from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.explainer.explainerBT import ExplainerBT
from pyxai.sources.core.structure.type import ReasonExpressivity
from pyxai.sources.solvers.MIP.SufficientRegressionBT import SufficientRegression
from pyxai.sources.solvers.MIP.Range import Range
import time


class ExplainerRegressionBT(ExplainerBT):
    def __init__(self, boosted_trees, instance=None):
        self._lower_bound = None
        self._upper_bound = None
        super().__init__(boosted_trees, instance)


    def set_instance(self, instance):
        super().set_instance(instance)
        self._lower_bound = self.predict(instance)
        self._upper_bound = self._lower_bound


    @property
    def regression_boosted_trees(self):
        """
        The tree of the model
        """
        return self._boosted_trees


    def set_interval(self, lower_bound, upper_bound):
        """
        Set the interval for the reason. The prediction must be in the interval. The largest the interval is,
        the smallest the reason is.
        Args:
            lower_bound: lower bound of the interval
            upper_bound: upper bound of the interval
        """
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound


    @property
    def lower_bound(self):
        return self._lower_bound


    @lower_bound.setter
    def lower_bound(self, lower_bound):
        self._lower_bound = lower_bound


    @property
    def upper_bound(self):
        return self._upper_bound


    @upper_bound.setter
    def upper_bound(self, upper_bound):
        self._upper_bound = upper_bound


    def predict(self, instance):
        return self._boosted_trees.predict_instance(instance)


    def predict_implicant(self, bin_implicant):
        return self._boosted_trees.predict_implicant(bin_implicant

                                                     )


    def tree_specific_reason(self, *, n_iterations=50, time_limit=None, seed=0):
        """
        Compute a tree specific reason related to the given interval
        Args:
            n_iterations: the number of iterations done
            time_limit:
            seed: The ssed

        Returns: a tree specific reason
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        if self._upper_bound is None or self.lower_bound is None:
            raise RuntimeError("lower bound and upper bound must be set when computing a reason")
        if seed is None:
            seed = -1
        if time_limit is None:
            time_limit = 0
        reason_expressivity = ReasonExpressivity.Conditions
        if self.c_BT is None:
            # Preprocessing to give all trees in the c++ library
            self.c_BT = c_explainer.new_regression_BT()
            for tree in self._boosted_trees.forest:
                c_explainer.add_tree(self.c_BT, tree.raw_data_for_CPP())
            c_explainer.set_base_score(self.c_BT, self._boosted_trees.learner_information.extras["base_score"])
        c_explainer.set_excluded(self.c_BT, tuple(self._excluded_literals))
        if self._theory:
            c_explainer.set_theory(self.c_BT, tuple(self._boosted_trees.get_theory(self._binary_representation)))
        c_explainer.set_interval(self.c_BT, self._lower_bound, self._upper_bound)
        # 0 for prediction. We don't care of it. The interval is the important thing here
        result = c_explainer.compute_reason(self.c_BT, self._binary_representation, self._implicant_id_features, 0, n_iterations,
                                          time_limit,
                                          int(reason_expressivity), seed, 0)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.tree_specific_reason.__name__, result)
        return result

    def sufficient_reason(self, *, seed=0, time_limit=None):
        raise NotImplemented("sufficient reason is not yet implemented for regression boosted trees")
        if self._instance is None:
            raise ValueError("Instance is not set")

        cplex = SufficientRegression()
        reason, time_used = cplex.create_model_and_solve(self, self._lower_bound, self._upper_bound)
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def extremum_range(self):
        """
        The extremum range for predictions. Computed in polynomial time, but the real extremum range can be smaller.
        Use range_for_partial_instance(self, [None for _ in range(explainer.instance]) to extract the real one (can be time consuming)
        Returns: a tuple (min_value, max_value)

        """
        min_weights = []
        max_weights = []
        for tree in self._boosted_trees.forest:
            leaves = tree.get_leaves()
            min_weights.append(min([l.value for l in leaves]))
            max_weights.append(max([l.value for l in leaves]))
        return (sum(min_weights), sum(max_weights))



    def range_for_partial_instance(self, partial_instance, *, time_limit=None):
        """
            Given a partial instance, extract the range for all possible extension of this range
            the partial instance is defined with None value on undefined values
            Return a tuple (min_value, max_value)
            """
        starting_time = -time.process_time()
        range = Range()
        partial = self._boosted_trees.instance_to_binaries(partial_instance)
        min_prediction = range.create_model_and_solve(self, None if self._theory is False else self._theory_clauses(), partial, True, time_limit)
        max_prediction = range.create_model_and_solve(self, None if self._theory is False else self._theory_clauses(), partial, False, time_limit)
        time_used = starting_time + time.process_time()
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        base_score = self._boosted_trees.learner_information.extras["base_score"]
        return [None, None] if self._elapsed_time == Explainer.TIMEOUT else [min_prediction + base_score, max_prediction+base_score]



    def is_implicant(self, abductive):
        min_weights = []
        max_weights = []
        base_score = self.regression_boosted_trees.learner_information.extras["base_score"]

        for tree in self._boosted_trees.forest:
            weights = self.compute_weights(tree, tree.root, abductive)
            min_weights.append(min(weights))
            max_weights.append(max(weights))
        return base_score + sum(min_weights) >= self._lower_bound and base_score + sum(max_weights) <= self._upper_bound
