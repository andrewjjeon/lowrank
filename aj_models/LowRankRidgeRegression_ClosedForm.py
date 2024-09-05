import numpy as np
import sklearn.linear_model

def _fit_rrr_no_intercept_all_ranks(X: np.ndarray, Y: np.ndarray, alpha: float, solver: str):
    ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False, solver=solver)  # alpha is lambda, ridge is the W_hat
    beta_ridge = ridge.fit(X, Y).coef_  # beta_ridge is B_Ridge
    Lambda = np.eye(X.shape[1]) * np.sqrt(alpha)
    X_star = np.concatenate((X, Lambda))
    Y_star = X_star @ beta_ridge.T
    _, _, Vt = np.linalg.svd(Y_star, full_matrices=False)
    return beta_ridge, Vt

def _fit_rrr_no_intercept(X: np.ndarray, Y: np.ndarray, alpha: float, rank: int, solver: str, memory=None):
    memory = sklearn.utils.validation.check_memory(memory)
    fit = memory.cache(_fit_rrr_no_intercept_all_ranks)
    beta_ridge, Vt = fit(X, Y, alpha, solver)
    return Vt[:rank, :].T @ (Vt[:rank, :] @ beta_ridge)  # this is the closed-form low rank ridge regression solution




class ReducedRankRidge(sklearn.base.MultiOutputMixin, sklearn.base.RegressorMixin, sklearn.linear_model._base.LinearModel):
    def __init__(self, alpha=1.0, fit_intercept=True, rank=None, ridge_solver='auto', memory=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rank = rank
        self.ridge_solver = ridge_solver
        self.memory = memory

    def fit(self, X, y):
        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            y_offset = np.average(y, axis=0)
            # doesn't modify inplace, unlike -=
            X = X - X_offset
            y = y - y_offset
        self.coef_ = _fit_rrr_no_intercept(X, y, self.alpha, self.rank, self.ridge_solver, self.memory)
        self.rank_ = np.linalg.matrix_rank(self.coef_)
        if self.fit_intercept:
            self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.intercept_ = np.zeros(y.shape[1])
        return self





def reduced_rank_regression(X, Y, alpha, max_m=15, fit_intercept=True,
                            nfolds=10, shuffle=True, return_min_score=False,
                            verbose=False):
    """
    reduced rank regression cf. Semedo et al. methods
    inferred dimensionality is smallest rank which is within one SEM of full rank regression
    """
    #ntrials = len(X)
    #ones_to_append = np.zeros((ntrials, 1))
    #X = np.concatenate((ones_to_append, X), axis=1)
    test_cv = KFold(n_splits=nfolds, shuffle=shuffle,
                    random_state=test_random_state)
    memory = joblib.Memory(verbose=0)
    ridge_model = ReducedRankRidge(fit_intercept=fit_intercept, alpha=alpha,
                                   memory=memory)
    scores = np.zeros((max_m))
    full_scores = cross_val_score(ridge_model, X=X, y=Y, cv=test_cv)
    best_score = np.mean(full_scores)
    best_score_sem = np.std(full_scores)/np.sqrt(nfolds)
    for i in range(max_m):
        ridge_model.rank = i + 1
        scores[i] = cross_val_score(ridge_model, X=X, y=Y, cv=test_cv).mean()
    min_acceptable_score = best_score - best_score_sem
    passes = np.argwhere(scores > min_acceptable_score).reshape(-1)
    if len(passes) == 0:
        if verbose:
            print("no scores within one SEM of best score ({}): {}".format(
                min_acceptable_score, scores))
        chosen_ind = max_m
    else:
        chosen_ind = passes[0]
    chosen_m = chosen_ind + 1
    memory.clear(warn=False)
    if return_min_score:
        return scores, chosen_m, min_acceptable_score
    else:
        return scores, chosen_m