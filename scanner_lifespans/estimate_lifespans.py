import numpy
from sklearn import neighbors, grid_search, base

from zplib.scalar_stats import hmm, smoothing, kde

def estimate_lifespans(scores, ages, reference_states, reference_scores, reference_ages):
    states, p_initial, p_transition = annotate_dead(scores, reference_scores[reference_states==1], reference_scores[reference_states==0])
    return states

def annotate_dead(scores, scores_live, scores_dead, transition_time_sigma=3, max_iters=5):
    estimator = hmm.ObservationProbabilityEstimator([scores_dead, scores_live])
    p_obses = [estimator(score_series) for score_series in scores]
    prev_states = numpy.zeros(scores.shape, dtype=numpy.uint16)
    for i in range(max_iters):
        if i == 0:
            # state 0 = dead, state 1 =live
            p_transition = [[1, 0], [0.5, 0.5]]
            p_initial = [0.1, 0.9]
        else:
            p_initial, p_transition = hmm.estimate_hmm_params(prev_states, pseudocount=1, time_sigma=transition_time_sigma)
            p_transition[:,0] = [1, 0] # sternly envorce that the dead stay dead.
        states = numpy.array([hmm.viterbi(p_obs, p_transition, p_initial) for p_obs in p_obses])
        diffs = (states != prev_states).sum()
        prev_states = states
        #print(i, diffs)
        if diffs == 0:
            break
    return states, p_initial, p_transition

def simple_hmm(scores, ages, ref_lifespans, ref_ages, ref_scores, ref_states, lifespan_sigma=None):
    scores_live = ref_scores[ref_states==1]
    scores_dead = ref_scores[ref_states==0]
    estimator = hmm.ObservationProbabilityEstimator([scores_dead, scores_live])
    p_obses = [estimator(score_series) for score_series in scores]
    estimator = SimpleHMMEstimator(ref_lifespans, lifespan_sigma)
    p_initial = estimator.p_initial(ages[0])
    p_transition = estimator.p_transition(ages)
    states = numpy.array([hmm.viterbi(p_obs, p_transition, p_initial) for p_obs in p_obses])
    return states, p_initial, p_transition

def em_hmm(scores, ages, ref_lifespans, ref_ages, ref_scores, ref_states, age_sigma, score_sigma=None, lifespan_sigma=None, dead_sigma=None, em_iters=5, update_p_obs=True, update_p_trans=True):
    estimator = HMMParameterEstimator(ref_lifespans, ref_ages, ref_scores, ref_states, age_sigma, score_sigma, lifespan_sigma, dead_sigma)
    p_obses = estimator.p_obs(scores, ages)
    p_initial = estimator.p_initial(ages[0])
    p_transition = estimator.p_transition(ages)
    if not (update_p_trans or update_p_obs):
        em_iters = 1
    for i in range(em_iters):
        states = numpy.array([hmm.viterbi(p_obs, p_transition, p_initial) for p_obs in p_obses])
        if i > 0:
            diffs = (states != prev_states).sum()
            # print(i, diffs)
            if diffs == 0:
                break
        prev_states = states
        if update_p_obs or update_p_trans:
            lifespans = cleanup_lifespans(states_to_lifespans(states, ages), ages)
            ref_ages, ref_scores, ref_states = make_ref(ages, scores, states)
            estimator = HMMParameterEstimator(lifespans, ref_ages, ref_scores, ref_states, age_sigma, score_sigma=None, lifespan_sigma=None, dead_sigma=None)
            if update_p_obs:
                p_obses = estimator.p_obs(scores, ages)
            if update_p_trans:
                p_initial = estimate_p_initial(states)
                p_transition = estimator.p_transition(ages)
    return states, p_initial, p_transition

def make_ref(ages, scores, states):
    num_worms, num_timepoints = scores.shape
    assert len(ages) == num_timepoints
    ref_ages = numpy.tile(ages, num_worms) # repeat ages for each worm. all_ages.shape == flat_scores.shape
    ref_scores = scores.flatten()
    ref_states = states.flatten()
    return ref_ages, ref_scores, ref_states

def estimate_p_initial(state_sequences, pseudocount=1):
    n, t = state_sequences.shape
    s = state_sequences.max() + 1 # number of states
    initial_counts = numpy.bincount(state_sequences[:,0], minlength=s) + pseudocount
    p_initial = initial_counts / (n + s*pseudocount)
    return p_initial

class _KDE_for_sklearn(kde.FixedBandwidthKDE, base.BaseEstimator):
    def __init__(self, bandwidth=1):
        self.bandwidth = bandwidth

    def fit(self, X, y=None):
        # X.shape = (n_samples, dimension) -- OPPOSITE convention of scipy kde!
        self.dataset = numpy.atleast_2d(numpy.transpose(X))
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape
        self.set_bandwidth(self.bandwidth)
        return self

    def score(self, X, y=None):
        p = self(numpy.transpose(X))
        if numpy.any(p==0):
            return -numpy.inf
        return numpy.log(p).sum()

class SimpleHMMEstimator:
    def __init__(self, lifespans, lifespan_sigma=None):
        """
        Parameters:
        lifespans: list of lifespans from a comparable population of worms, which
            will be used to estimate the transition probabilities of the HMM --
            i.e. what is the proability of death over any given age interval?
        """
        lifespans = numpy.asarray(lifespans)
        assert sum(lifespans == -1) == sum(numpy.isnan(lifespans)) == 0
        if lifespan_sigma is None:
            lifespan_sigma = self.find_best_bandwidth(lifespans)
        self.lifespan_kde = kde.FixedBandwidthKDE(lifespans, lifespan_sigma) # p(lifespan)

    @staticmethod
    def find_best_bandwidth(data):
        n = len(data)
        scotts_factor = n**(-1/5)*numpy.std(data) # starting bandwidth via Scott's rule of thumb
        log_scotts = numpy.log2(scotts_factor)
        params = {'bandwidth': numpy.logspace(log_scotts-2, log_scotts+2, 20, base=2)}
        grid = grid_search.GridSearchCV(_KDE_for_sklearn(), params)
        if n > 5000:
            data = data[::n//2500]
        grid.fit(data)
        return grid.best_estimator_.bandwidth


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
        # now, we need the probability that an animal dies in an interval, given that it was alive right before
        # p(dies at t | alive at t-1) = p(dies at t and alive at t-1) / p(alive at t-1)
        # as p(dies at t and alive at t-1) is obviously equal to p(dies at t) (how could it be otherwise??):
        p_dies_in_interval = []
        for start, end in zip(ages[:-1], ages[1:]):
            num = self.lifespan_kde.integrate_box_1d(start, end) # probability of a death in that interval
            denom = 1 - self.lifespan_kde.integrate_box_1d(0, start) # probability of surviving to start of interval

            if denom == 0:
                p = 1
            else:
                p = num / denom
            p_dies_in_interval.append(p)
        min_prob = 1e-4
        max_prob = 1 - min_prob
        p_dies_in_interval = numpy.clip(p_dies_in_interval, min_prob, max_prob) # make sure there's always a chance of death/survival...
        p_transition[:,1,0] = p_dies_in_interval # probability of going from live (state=1) to dead (state=0)
        p_transition[:,1,1] = 1 - p_dies_in_interval
        return p_transition

    def p_initial(self, first_age):
        # assume worms that died in the last day might show up as DOA
        doa_rate = self.lifespan_kde.integrate_box_1d(first_age-1, first_age)
        return numpy.array([doa_rate, 1-doa_rate])


class HMMParameterEstimator:
    def __init__(self, lifespans, ages, scores, states, age_sigma, score_sigma=None, lifespan_sigma=None, dead_sigma=None):
        """
        Parameters:
        lifespans: list of lifespans from a comparable population of worms, which
            will be used to estimate the transition probabilities of the HMM --
            i.e. what is the proability of death over any given age interval?
        ages, scores, states: each a list of length n (does not need to be the
            same length as the lifespans array), giving movement scores for
            living (state=1) and dead (state=0) worms at different ages. Ideally
            this will cover a range of ages and both living and dead. These will
            be used to estimate the observation probabilities as a function of
            age, using Gaussian KDE.
        age_sigma: standard deviation for KDE smoothing of the ages of living
            worms, for estimating p(score | age, alive)
        score_sigma: standard deviation for KDE smoothing of the scores of living
            worms, for estimating p(score | age, alive). If None, fit from data.
        lifespan_sigma: standard deviation for KDE smoothing of the worm lifespans
            for estimating p(lifespan). If None, fit from data.
        dead_sigma: standard deviation for KDE smoothing of the scores of dead
            worms, for estimating p(score | dead). If None, fit from data.
        """
        ages = numpy.asarray(ages)
        scores = numpy.asarray(scores)
        states = numpy.asarray(states)
        assert len(ages) == len(scores) == len(states)
        lifespans = numpy.asarray(lifespans)
        assert sum(lifespans == -1) == sum(numpy.isnan(lifespans)) == 0
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
        if score_sigma is None:
            score_sigma = self.find_best_score_bandwidth(live_data, age_sigma)
        self.live_kde = kde.FixedBandwidthKDE(live_data, [age_sigma, score_sigma]) # p(score, age | live)
        self.age_kde = kde.FixedBandwidthKDE(ages[live_mask], age_sigma) # p(age | live), which is the marginal of self.live_kde
        #note: p(score | age, live) = p(score, age | live) / p(age | live)

    @staticmethod
    def find_best_bandwidth(data):
        n = len(data)
        scotts_factor = n**(-1/5)*numpy.std(data) # starting bandwidth via Scott's rule of thumb
        log_scotts = numpy.log2(scotts_factor)
        params = {'bandwidth': numpy.logspace(log_scotts-2, log_scotts+2, 20, base=2)}
        grid = grid_search.GridSearchCV(_KDE_for_sklearn(), params)
        if n > 5000:
            data = data[::n//2500]
        grid.fit(data)
        return grid.best_estimator_.bandwidth

    @staticmethod
    def find_best_score_bandwidth(live_data, age_sigma):
        live_data = live_data.T # shape n, 2
        n = len(live_data)
        scotts_factor = n**(-1/6)*numpy.std(live_data[:,1]) # starting bandwidth via Scott's rule of thumb
        log_scotts = numpy.log2(scotts_factor)
        score_sigmas = numpy.logspace(log_scotts-2, log_scotts+2, 20, base=2)
        params = {'bandwidth': [(age_sigma, score_sigma) for score_sigma in score_sigmas]}
        grid = grid_search.GridSearchCV(_KDE_for_sklearn(), params)
        if n > 5000:
            live_data = live_data[::n//2500]
        grid.fit(live_data)
        return grid.best_estimator_.bandwidth[1]

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
        assert len(ages) == num_timepoints
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
        # now, we need the probability that an animal dies in an interval, given that it was alive right before
        # p(dies at t | alive at t-1) = p(dies at t and alive at t-1) / p(alive at t-1)
        # as p(dies at t and alive at t-1) is obviously equal to p(dies at t) (how could it be otherwise??):
        p_dies_in_interval = []
        for start, end in zip(ages[:-1], ages[1:]):
            num = self.lifespan_kde.integrate_box_1d(start, end) # probability of a death in that interval
            denom = 1 - self.lifespan_kde.integrate_box_1d(0, start) # probability of surviving to start of interval

            if denom == 0:
                p = 1
            else:
                p = num / denom
            p_dies_in_interval.append(p)
        min_prob = 1e-4
        max_prob = 1 - min_prob
        p_dies_in_interval = numpy.clip(p_dies_in_interval, min_prob, max_prob) # make sure there's always a chance of death/survival...
        p_transition[:,1,0] = p_dies_in_interval # probability of going from live (state=1) to dead (state=0)
        p_transition[:,1,1] = 1 - p_dies_in_interval
        return p_transition

    def p_initial(self, first_age):
        # assume worms that died in the last day might show up as DOA
        doa_rate = self.lifespan_kde.integrate_box_1d(first_age-1, first_age)
        return numpy.array([doa_rate, 1-doa_rate])


def states_to_lifespans(states, ages):
    last_alive_indices = states_to_last_alive_indices(states)
    lifespans = last_alive_indices_to_lifespans(last_alive_indices, ages)
    return lifespans

def states_to_last_alive_indices(states):
    # states.shape = num_worms, num_timepoints
    last_alive_indices = []
    indices = numpy.arange(states.shape[1])
    for worm_states in states:
        if worm_states[0] == 0: # worm was never observed to be alive: well empty / DOA
            i = None
        else: # worm was alive for one or more observations
            i = (worm_states * indices).argmax()
        last_alive_indices.append(i)
    return last_alive_indices

def last_alive_indices_to_states(last_alive_indices, num_timepoints):
    states = []
    for i in last_alive_indices:
        worm_states = numpy.zeros(num_timepoints, dtype=int)
        if i != None: # worm was alive for one or more observations
            worm_states[:i+1] = 1
        states.append(worm_states)
    return numpy.array(states, dtype=int)

def last_alive_indices_to_lifespans(last_alive_indices, ages):
    lifespans = []
    for i in last_alive_indices:
        if i == None:
            lifespan = -1 # worm was never observed to be alive: well empty / DOA
        else:
            if i == len(ages) - 1: # worm was never observed to be dead
                lifespan = numpy.nan
            else: # worm was observed both alive and dead
                lifespan = (ages[i] + ages[i+1]) / 2 # assume death was between last live observation and first dead observation
        lifespans.append(lifespan)
    return numpy.array(lifespans)

def lifespans_to_last_alive_indices(lifespans, ages):
    last_alive_indices = []
    ages = numpy.asarray(ages)
    for lifespan in lifespans:
        if numpy.isnan(lifespan): # worm was never observed to be dead
            i = len(ages) - 1
        else:
            timepoints_alive = (lifespan > ages).sum()
            if timepoints_alive == 0: # worm was never observed to be alive: well empty / DOA
                i = None
            else: # worm was observed both alive and dead
                i = timepoints_alive - 1
        last_alive_indices.append(i)
    return last_alive_indices

def lifespans_to_states(lifespans, ages):
    states = []
    ages = numpy.asarray(ages)
    for lifespan in lifespans:
        worm_states = numpy.ones(len(ages), dtype=int)
        if not numpy.isnan(lifespan): # worm was dead for one or more observations
            worm_states[ages > lifespan] = 0 # 0 means dead
        states.append(worm_states)
    return numpy.array(states, dtype=int)

def cleanup_lifespans(lifespans, ages):
    """Remove NANs from lifespan list by assuming that the animals died over the
    next-possible interval after the last recorded observation. Remove -1s from
    the lifespan list by assuming that DOA animals died right before observation."""
    lifespans = numpy.array(lifespans)
    next_age = ages[-1] + (ages[-1] - ages[-2])
    death_time = (ages[-1] + next_age) / 2
    lifespans[numpy.isnan(lifespans)] = death_time
    lifespans[lifespans == -1] = ages[0]
    return lifespans
