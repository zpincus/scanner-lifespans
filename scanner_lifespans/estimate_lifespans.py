import numpy
from sklearn import neighbors, grid_search

from zplib.scalar_stats import hmm, smoothing, kde


def annotate_dead(scores, scores_live, scores_dead, max_iters=5):
    prev_states = numpy.zeros(scores.shape, dtype=numpy.uint16)
    for i in range(max_iters):
        if i == 0:
            # state 0 = dead, state 1 =live
            p_transition = [[1, 0], [0.5, 0.5]]
            p_initial = [0.1, 0.9]
        else:
            p_initial, p_transition = hmm.estimate_hmm_params(prev_states, pseudocount=1, time_sigma=3)
            p_transition[:,0] = [1, 0] # sternly envorce that the dead stay dead.
        estimator = hmm.ObservationProbabilityEstimator([scores_dead, scores_live])
        p_obses = [estimator(score_series) for score_series in scores]
        states = numpy.array([hmm.viterbi(p_obs, p_transition, p_initial) for p_obs in p_obses])
        diffs = (states != prev_states).sum()
        prev_states = states
        print(i, diffs)
        if diffs == 0:
            break
    return states, p_initial, p_transition

def annotate_dead2(scores, ages, p_transition, p_initial, reference_ages, reference_live, reference_dead, p_t_time_sigma, p_o_time_sigma, max_iters=5):
    for i in range(max_iters):
        p_obses = estimate_p_obs(ages, scores, reference_ages, reference_live, reference_dead, time_sigma=p_o_time_sigma)
        states = numpy.array([hmm.viterbi(p_obs, p_transition, p_initial) for p_obs in p_obses])
        if i > 0:
            diffs = (states != prev_states).sum()
            print(i, diffs)
            if diffs == 0:
                break
        prev_states = states
        p_initial, p_transition = hmm.estimate_hmm_params(states, pseudocount=1, time_sigma=p_t_time_sigma)
        p_transition[:,0] = [1, 0] # sternly enforce that the dead stay dead.
        reference_ages, reference_live, reference_dead = make_reference_score_sets(states, scores, ages)
    return states, p_initial, p_transition


class ParameterEstimator:
    def __init__(self, doa_rate, lifespans, ages, scores, states, score_sigma, age_sigma, lifespan_sigma=None, dead_sigma=None):
        """
        Parameters:
        doa_rate: fraction of wells with already-dead worms or simply empty wells
        lifespans: list of lifespans from a comparable population of worms, which
            will be used to estimate the transition probabilities of the HMM --
            i.e. what is the proability of death over any given age interval?
        ages, scores, states: each a list of length n (does not need to be the
            same length as the lifespans array), giving movement scores for
            living (state=1) and dead (state=0) worms at different ages. Ideally
            this will cover a range of ages and both living and dead. These will
            be used to estimate the observation probabilities as a function of
            age, using Gaussian KDE.
        score_sigma: standard deviation for KDE smoothing of the scores of living
            worms, for estimating p(score | age, alive)
        age_sigma: standard deviation for KDE smoothing of the ages of living
            worms, for estimating p(score | age, alive)
        lifespan_sigma: standard deviation for KDE smoothing of the worm lifespans
            for estimating p(lifespan). If None, fit from data.
        dead_sigma: standard deviation for KDE smoothing of the scores of dead
            worms, for estimating p(score | dead). If None, fit from data.
        """
        lifespans = numpy.asarray(lifespans)
        ages = numpy.asarray(ages)
        scores = numpy.asarray(scores)
        states = numpy.asarray(states)
        assert len(ages) == len(scores) == len(states)

        self.doa_rate = doa_rate
        if lifespan_sigma is None:
            lifespan_sigma = self.find_best_bandwidth(lifespans)
        self.lifespan_kde = kde.FixedBandwidthKDE(lifespans, lifespan_sigma) # p(lifespan)
        dead_mask = states == 0
        dead_scores = scores[dead_mask]
        if dead_sigma is None:
            dead_sigma = self.find_best_bandwidth(dead_scores)
        self.dead_kde = kde.FixedBandwidthKDE(dead_scores, dead_sigma) # p(score | dead)
        live_mask = ~dead_mask
        live_data = numpy.array([ages[live_mask], scores[live_mask]]) # shape = 2, n
        self.live_kde = kde.FixedBandwidthKDE(live_data, [age_sigma, score_sigma]) # p(score, age | live)
        self.age_kde = kde.FixedBandwidthKDE(ages[live_mask], age_sigma) # p(age | live), which is the marginal of self.live_kde
        #note: p(score | age, live) = p(score, age | live) / p(age | live)

    @staticmethod
    def find_best_bandwidth(data):
        n = len(data)
        scotts_factor = n**(-1/5)*numpy.std(data) # starting bandwidth via Scott's rule of thumb
        log_scotts = numpy.log2(scotts_factor)
        params = {'bandwidth': np.logspace(log_scotts-2, log_scotts+2, 20, base=2)}
        grid = grid_search.GridSearchCV(neighbors.KernelDensity(), params)
        grid.fit(self.lifespans)
        return grid.best_estimator_.bandwidth

    def p_obs(self, scores, ages):
        """Estimate the probability of observing the given scores from worms
        of the given ages, either if the worm is alive or dead.

        Parameters:
        scores: array of shape (num_worms, num_timepoints), recording the score of
            of each of a set of worms at each age.
        ages: list of ages at which worms were scored. len(ages) must be num_timepoints

        Returns: array of shape (num_worms, num_timepoints, 2), giving for each
            observation at each timepoint the proabilities of the observation given
            that the worm is dead and given that the worm is alive.
        """
        num_worms, num_timepoints = scores.shape
        assert len(ages) == num_worms
        flat_scores = scores.flatten()

        p_dead = self.dead_kde(flat_scores).reshape(scores.shape)
        all_ages = numpy.tile(ages, num_worms) # repeat ages for each worm. all_ages.shape == flat_scores.shape
        ages_and_scores = numpy.array([all_ages, flat_scores]) # shape = (2, num_worms*num_timepoints)
        p_live_and_age = self.live_kde(ages_and_scores)
        p_ages = self.age_kde(ages)
        p_all_ages = numpy.tile(p_ages, num_worms)
        p_live_given_age = (p_live_and_age / p_all_ages).reshape(scores.shape)
        p_obs = numpy.dstack([p_dead, p_live_given_age]) # shape (num_worms, num_timepoints, 2)
        return p_obs

    def p_transition(self, ages):
        """Estimate the HMM transition probability matrix.
        Parameters:
        ages: list of ages at which worms were scored.

        Returns: array of shape (len(ages)-1, 2, 2) where [t, si, sj] is the
            probability of transition from state si to state sj between age[t]
            and age[t+1].
        """
        p_transition = numpy.empty((len(ages)-1, 2, 2))
        p_transition[:,0] = [1, 0] # sternly enforce that the dead stay dead at all times.
        p_dies_in_interval = []
        for start, end in zip(ages[:-1], ages[1:]):
            p_dies_in_interval.append(self.lifespan_kde.integrate_box_1d(start, end))
        p_dies_in_interval = numpy.array(p_dies_in_interval)
        min_prob = p_dies_in_interval.max() / 100
        p_dies_in_interval[p_dies_in_interval < min_prob] = min_prob # make sure there's always a chance of death-
        p_transition[:,1,0] = p_dies_in_interval # probability of going from live (state=1) to dead (state=0)
        p_transition[:,1,1] = 1 - p_dies_in_interval
        return p_transition

    def p_initial(self):
        return numpy.array([self.doa_rate, 1-self.doa_rate])


def estimate_p_obs(ages, scores, reference_ages, reference_live, reference_dead, time_sigma):
    """Given a set of reference scores for live and dead worms at different ages,
    estimate the probability of observing the given scores at the given ages.
    The probability estimates for the scores at a give age will be a weighted
    average of the scores for the reference ages, where the weight is a gaussian
    function of the difference in ages.

    Parameters:
        ages: list of ages at which worms were scored. Let num_timepoints = len(ages)
        scores: array of shape (num_worms, num_timepoints), recording the score of
            of each of a set of worms at each age.
        reference_ages: list of ages at which reference live/dead scores are provided.
        reference_live: list of arrays of scores of live worms at different ages.
            Must be same length as reference_ages. The length of each set of
            scores can differ.
        reference_dead: list of arrays of scores of dead worms at different ages.
            Must be same length as reference_ages. The length of each set of
            scores can differ.
        time_sigma: standard deviation of the gaussian used to weight the
            age differences. Larger values give more smoothing over time.

    Returns: array of shape (num_worms, num_timepoints, 2), giving for each
        observation at each timepoint the proabilities of the observation given
        that the worm is dead and given that the worm is alive.
    """
    reference_estimators = []
    reference_ages = numpy.asarray(reference_ages)
    for scores_live, scores_dead in zip(reference_live, reference_dead):
        estimator = hmm.ObservationProbabilityEstimator([scores_dead, scores_live])
        reference_estimators.append(estimator)
    scores = numpy.asarray(scores)
    num_worms, num_timepoints = scores.shape
    assert len(ages) == num_timepoints
    p_obses = numpy.empty((num_worms, num_timepoints, 2), dtype=float)
    for i, age in enumerate(ages):
        timepoint_data = scores[:, i]
        estimates = numpy.array([estimator(timepoint_data) for estimator in reference_estimators])
        # estimates.shape == (len(reference_ages), num_worms, 2)
        weights = smoothing._gaussian(reference_ages, mu=age, sigma=time_sigma)
        total_weight = weights.sum()
        if total_weight > 0:
            weights /= weights.sum()
        else: # if all weights are zero
            weights = numpy.zeros_like(weights)
            weights[numpy.abs(age - reference_ages).argmin()] = 1 # put all weight on the closest
        weighted_estimates = estimates * weights[:, numpy.newaxis, numpy.newaxis]
        weighted_mean = weighted_estimates.sum(axis=0) # shape == (num_worms, 2)
        p_obses[:,i,:] = weighted_mean
    return p_obses

def states_to_last_alive_indices(states):
    # states.shape = num_worms, num_timepoints
    last_alive_indices = []
    indices = numpy.arange(states.shape[1])
    for worm_states in states:
        if worm_states[0] == 0:
            i = None
        else:
            i = (worm_states * indices).argmax()
        last_alive_indices.append(i)
    return numpy.array(last_alive_indices)

def last_alive_indices_to_states(last_alive_indices, num_timepoints):
    states = []
    for i in last_alive_indices:
        worm_states = numpy.zeros(num_timepoints, dtype=int)
        if i != None:
            worm_states[:i+1] = 1
        states.append(worm_states)
    return numpy.array(states, dtype=int)

def last_alive_indices_to_lifespans(last_alive_indices, ages):
    lifespans = []
    for i in last_alive_indices:
        if last_alive_i == None:
            lifespan = -1 # well was empty / DOA
        else:
            if last_alive_i < indices[-1]: # worm has died
                lifespan = (ages[last_alive_i] + ages[last_alive_i+1]) / 2 # assume death was between last live observation and first dead observation
            else:
                lifespan = numpy.nan # worm was never observed to be dead
        lifespans.append(lifespan)
    return numpy.array(lifespans)

def lifespan_to_last_alive_indices(lifespans, ages):
    last_alive_indices = []
    ages = numpy.asarray(ages)
    for lifespan in lifespans:
        if numpy.isnan(lifespan):
            i = len(ages) - 1
        else:
            timepoints_alive = (lifespan > ages).sum()
            if timepoints_alive > 0:
                i = timepoints_alive - 1
            else:
                i = None
        last_alive_indices.append(i)
    return numpy.array(last_alive_indices, dtype=int)

def lifespan_to_states(lifespans, ages):
    states = []
    ages = numpy.asarray(ages)
    for lifespan in lifespans:
        if numpy.isnan(lifespan):
            seq = numpy.ones(len(ages), dtype=int)
        else:
            seq = (ages < lifespan).astype(int) # 0 means dead
        states.append(seq)
    return numpy.array(states, dtype=int)

def make_reference_score_sets(states, scores, ages):
    # states, scores shape is (n_worms, n_timepoints)
    # reference_live and reference_dead are lists of n_timepoints length with scores for live and dead worms
    reference_live = []
    reference_dead = []
    reference_ages = []
    for states_at_t, scores_at_t, age in zip(states.T, scores.T, ages):
        live = states_at_t.astype(bool)
        num_live = live.sum()
        num_dead = len(live) - num_live
        if num_live < 10 or num_dead < 10:
            continue
        reference_live.append(scores_at_t[live])
        reference_dead.append(scores_at_t[~live])
        reference_ages.append(age)
    return reference_ages, reference_live, reference_dead

