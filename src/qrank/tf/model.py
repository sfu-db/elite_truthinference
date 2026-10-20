"""Reliability-weighted voting: exact MAP-EM over correct-cluster states, plus a structure ranker."""

import base64
from itertools import combinations
import logging
import pickle

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

from .structure import cluster_features

logger = logging.getLogger(__name__)
FORMAT = "tf_v3"
# Selection: among clusters whose log posterior is within RANKER_WINDOW nats of the most
# probable cluster, the highest log posterior + RANKER_WEIGHT * ranker score.
RANKER_WEIGHT = 2.
RANKER_WINDOW = 2.


def crp_log_normalizer(log_concentration, count):
    """For n = 0..count-1: sum_{s=1}^{n-1} log(alpha + s), and its first two derivatives in u = log(alpha)."""
    alpha = np.exp(log_concentration)
    steps = alpha + np.arange(1, max(count - 1, 0))
    pad = lambda values: np.concatenate(([0., 0.], np.cumsum(values)))[:count]
    return pad(np.log(steps)), pad(alpha / steps), pad(alpha * (steps - alpha) / steps ** 2)


class ReliabilityVote:
    def __init__(self, condition_on_empty=True, dialect="sqlite", route_budget=0., structure=True):
        self.condition_on_empty = bool(condition_on_empty)
        self.dialect = dialect
        # Without the structure ranker (non-SQL candidates) selection is the most probable cluster.
        self.structure = bool(structure)
        self.route_budget = float(route_budget)
        self.route_threshold = 0.

    @property
    def kinds(self):
        return 2 if self.condition_on_empty else 1

    def _pack(self, questions):
        index = {s: i for i, s in enumerate(self.sources)}
        n_q = len(questions)
        n_c = max(1, max(len(q.valid) for q in questions))
        n_t = max(len(q.clusters) + 1 for q in questions)
        states = np.zeros((n_q, n_t, n_c))
        valid = np.zeros((n_q, n_c), dtype=bool)
        reliability = np.zeros((n_q, n_c), dtype=int)
        state_mask = np.zeros((n_q, n_t), dtype=bool)
        blocks = np.zeros((n_q, n_t))
        sizes = np.zeros((n_q, n_t), dtype=int)
        lgamma = np.zeros((n_q, n_t))
        for j, q in enumerate(questions):
            n = len(q.valid)
            valid[j, :n] = True
            kind = np.array(q.nonempty, dtype=int) if self.condition_on_empty else np.zeros(n, dtype=int)
            reliability[j, :n] = np.array([index[s] for s in q.sources], dtype=int) * self.kinds + kind
            state_mask[j, :len(q.clusters) + 1] = True
            for t in range(len(q.clusters) + 1):
                wrong = [c for k, c in enumerate(q.clusters) if k != t - 1]
                blocks[j, t] = max(len(wrong) - 1, 0)
                sizes[j, t] = sum(map(len, wrong))
                lgamma[j, t] = gammaln([len(c) for c in wrong]).sum()
                if t:
                    states[j, t, list(q.clusters[t - 1])] = 1
        return dict(states=states, valid=valid, reliability=reliability, state_mask=state_mask,
                    blocks=blocks, sizes=sizes, lgamma=lgamma)

    def _expectation(self, packed):
        p = self.source_probability.ravel()[packed["reliability"]]
        joint = (np.einsum("qtc,qc->qt", packed["states"], np.log(p) - np.log1p(-p))
                 + (packed["valid"] * np.log1p(-p)).sum(axis=1)[:, None])
        normalizer, _, _ = crp_log_normalizer(self.log_concentration, packed["sizes"].max() + 1)
        joint += self.log_concentration * packed["blocks"] + packed["lgamma"] - normalizer[packed["sizes"]]
        joint = np.where(packed["state_mask"], joint, -np.inf)
        evidence = logsumexp(joint, axis=1)
        return joint - evidence[:, None], evidence

    def _concentration_step(self, blocks, masses):
        """Newton's method on the concave objective u*blocks - sum_n masses_n log Z_n(u) - u^2/2."""
        u = self.log_concentration

        def value(x):
            return x * blocks - masses @ crp_log_normalizer(x, len(masses))[0] - .5 * x * x

        for _ in range(100):
            _, first, second = crp_log_normalizer(u, len(masses))
            gradient = blocks - masses @ first - u
            curvature = -(masses @ second) - 1.
            if abs(gradient) <= 1e-10 * (1 + blocks):
                return u
            step = -gradient / curvature
            while value(u + step) < value(u) - 1e-12 * (1 + abs(value(u))):
                step /= 2
            u += step
        raise RuntimeError("Concentration update did not converge")

    def _maximization(self, packed, mass):
        expected = np.einsum("qt,qtc->qc", mass, packed["states"])
        ids, valid = packed["reliability"], packed["valid"]
        size = self.source_probability.size
        counts = np.bincount(ids[valid], minlength=size)
        successes = np.bincount(ids[valid], weights=expected[valid], minlength=size)
        # Beta(2, 2) MAP: interior estimates without clipping.
        self.source_probability = ((successes + 1) / (counts + 2)).reshape(self.source_probability.shape)
        blocks = float((mass * packed["blocks"]).sum())
        masses = np.bincount(packed["sizes"].ravel(), weights=mass.ravel())
        self.log_concentration = self._concentration_step(blocks, masses)

    def _log_prior(self):
        p = self.source_probability.ravel()
        return float(np.sum(np.log(p) + np.log1p(-p)) - .5 * self.log_concentration ** 2)

    def _fit_ranker(self, questions, log_mass):
        """Pairwise ranker on cluster-feature differences, with the EM posterior as soft targets (no labels).

        Every cluster pair (a, b) with posterior r_a > r_b gives the rows (F_a - F_b, 1) and (F_b - F_a, 0),
        both weighted by r_a - r_b.
        """
        rows, targets, weights = [], [], []
        for j, q in enumerate(questions):
            if len(q.clusters) < 2:
                continue
            features = cluster_features(q, self.dialect)
            posterior = np.exp(log_mass[j, 1:len(q.clusters) + 1])
            for a, b in combinations(range(len(q.clusters)), 2):
                if posterior[a] == posterior[b]:
                    continue
                win, lose = (a, b) if posterior[a] > posterior[b] else (b, a)
                difference = features[win] - features[lose]
                rows += [difference, -difference]
                targets += [1, 0]
                weights += [abs(posterior[a] - posterior[b])] * 2
        if not rows:
            raise ValueError("The cluster ranker needs training questions with at least two execution clusters")
        rows = np.array(rows)
        self.ranker_scale = rows.std(axis=0) + 1e-9
        self.ranker = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                                                     early_stopping=False, random_state=0)
        self.ranker.fit(rows / self.ranker_scale, np.array(targets), sample_weight=np.array(weights))

    def _ranker_scores(self, questions):
        """Per question with at least two clusters: each cluster's mean log-odds of beating every other cluster."""
        blocks, spans = [], {}
        for j, q in enumerate(questions):
            n = len(q.clusters)
            if n < 2:
                continue
            features = cluster_features(q, self.dialect) / self.ranker_scale
            spans[j] = (sum(len(b) for b in blocks), n)
            blocks.append(np.array([features[a] - features[b] for a in range(n) for b in range(n) if b != a]))
        if not blocks:
            return {}
        probability = np.clip(self.ranker.predict_proba(np.vstack(blocks))[:, 1], 1e-12, 1 - 1e-12)
        log_odds = np.log(probability) - np.log1p(-probability)
        return {j: log_odds[start:start + n * (n - 1)].reshape(n, n - 1).mean(axis=1) for j, (start, n) in spans.items()}

    def fit(self, questions, max_iter=500, tol=1e-7):
        self.sources = tuple(sorted({s for q in questions for s in q.sources}))
        self.source_probability = np.full((len(self.sources), self.kinds), .5)
        self.log_concentration = 0.
        packed = self._pack(questions)
        log_mass, evidence = self._expectation(packed)
        score = float(evidence.sum() + self._log_prior())
        self.history = [score]
        for iteration in range(1, max_iter + 1):
            self._maximization(packed, np.exp(log_mass))
            log_mass, evidence = self._expectation(packed)
            updated = float(evidence.sum() + self._log_prior())
            if not np.isfinite(updated) or updated < score - 1e-8 * (1 + abs(score)):
                raise RuntimeError("EM did not preserve a finite, nondecreasing MAP objective")
            self.history.append(updated)
            if iteration % 10 == 0:
                logger.info("EM iteration %d: log posterior %.6f", iteration, updated)
            if abs(updated - score) <= tol * (1 + abs(score)):
                if self.structure:
                    self._fit_ranker(questions, log_mass)
                # Label-free routing cutoff: the route_budget quantile of training confidence.
                self.route_threshold = 0.
                confidence = np.array([p["confidence"] for p in self.predict(questions)])
                self.route_threshold = (float(np.quantile(confidence, self.route_budget, method="higher"))
                                        if self.route_budget else 0.)
                return self
            score = updated
        raise RuntimeError(f"EM did not converge in {max_iter} iterations")

    def predict(self, questions):
        log_mass, evidence = self._expectation(self._pack(questions))
        ranker = self._ranker_scores(questions) if self.structure else {}
        predictions = []
        for j, q in enumerate(questions):
            logs = log_mass[j, :len(q.clusters) + 1]
            clusters = [[q.valid[i] for i in c] for c in q.clusters]
            scores = ranker.get(j, np.zeros(len(clusters)))
            winners = []
            if clusters:
                eligible = logs[1:] >= logs[1:].max() - RANKER_WINDOW
                selection = np.where(eligible, logs[1:] + RANKER_WEIGHT * scores, -np.inf)
                best = selection.max()
                winners = [k for k in range(len(clusters)) if abs(selection[k] - best) <= 1e-12]
            # Posterior probability of the selected cluster; zero without valid candidates.
            confidence = float(np.exp(logs[winners[0] + 1])) if winners else 0.
            predictions.append({
                "question_id": q.question_id, "db_id": q.db_id, "clusters": clusters,
                "probabilities": np.exp(logs).tolist(), "log_probabilities": logs.tolist(),
                "ranker_scores": scores.tolist(), "decision_clusters": winners,
                # Forced choice among observed clusters, even when all-wrong is most probable.
                "selected_index": clusters[winners[0]][0] if winners else None,
                "confidence": confidence,
                # Forward to a stronger solver (e.g. an LLM call); the forced selection is kept as a fallback.
                "route": confidence < self.route_threshold,
                "log_evidence": float(evidence[j])})
        return predictions

    def to_dict(self):
        state = {"format": FORMAT, "condition_on_empty": self.condition_on_empty, "dialect": self.dialect,
                 "route_budget": self.route_budget, "route_threshold": self.route_threshold,
                 "structure": self.structure, "sources": list(self.sources),
                 "source_probability": self.source_probability.tolist(), "log_concentration": self.log_concentration,
                 "history": self.history}
        if self.structure:
            state["ranker_scale"] = self.ranker_scale.tolist()
            # The fitted scikit-learn ranker, pickled; load only model files you produced.
            state["ranker"] = base64.b64encode(pickle.dumps(self.ranker)).decode("ascii")
        return state

    @classmethod
    def from_dict(cls, state):
        if state["format"] != FORMAT:
            raise ValueError(f"Unknown model format {state['format']!r}")
        model = cls(state["condition_on_empty"], state["dialect"], state["route_budget"], state["structure"])
        model.route_threshold = float(state["route_threshold"])
        model.sources = tuple(state["sources"])
        model.source_probability = np.asarray(state["source_probability"], dtype=float)
        model.log_concentration = float(state["log_concentration"])
        if model.structure:
            model.ranker_scale = np.asarray(state["ranker_scale"], dtype=float)
            model.ranker = pickle.loads(base64.b64decode(state["ranker"]))
        model.history = state["history"]
        return model
