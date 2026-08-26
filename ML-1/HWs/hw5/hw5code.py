import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини нужно выбирать минимальный сплит
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты 
    можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    mask = np.diff(x) != 0

    if not np.any(mask):
        return np.array([]), np.array([]), None, None

    thresholds = (x[:-1][mask] + x[1:][mask]) / 2
    idxs = np.where(mask)[0] + 1

    unique = np.unique(y)
    Y = np.vstack([(y == k).astype(int) for k in unique])
    pref = np.cumsum(Y, axis=1)

    lc = pref[:, idxs - 1]
    rc = pref[:, -1][:, None] - lc
    ls = lc.sum(axis=0)
    rs = rc.sum(axis=0)
    n = len(y)

    Hl = 1 - np.sum((lc / ls) ** 2, axis=0)
    Hr = 1 - np.sum((rc / rs) ** 2, axis=0)
    ginis = -(ls / n) * Hl - (rs / n) * Hr

    i = np.argmax(ginis)
    gini_best = ginis[i]
    threshold_best = thresholds[i]

    return thresholds, ginis, threshold_best, gini_best



class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        sub_y = np.asarray(sub_y)
        if np.all(sub_y == sub_y[0]):  # ==
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        # стоп по глубине
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):  # без 1
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                main_class = Counter(sub_y).most_common(1)[0][0]
                clicks = Counter(sub_X[sub_y == main_class, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # как в лекции
                sorted_categories = [key for key, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))

            else:
                raise ValueError

            if len(Counter(feature_vector)) < 2:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            if thresholds.size == 0:
                continue

            for threshold, gini in zip(thresholds, ginis):
                curr_split = feature_vector < threshold
                if self._min_samples_leaf is not None:
                    if np.sum(curr_split) < self._min_samples_leaf or np.sum(~curr_split) < self._min_samples_leaf:
                        continue

                if gini_best is not None and gini <= gini_best:
                    continue

                feature_best = feature
                gini_best = gini
                split = curr_split

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = set(threshold_best)
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif self._feature_types[feature] == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

class LinearRegressionTree:
    """Compatibility stub kept out of the contest solution.

    The homework and contest tasks use ``DecisionTree`` for classification.
    This class was present in the draft file but has no completed statement in
    the assignment, so it is intentionally not exposed as a silent no-op.
    """

    def __init__(self, *args, **kwargs):
        raise ValueError("LinearRegressionTree is not part of HW5 tasks; use DecisionTree instead.")
