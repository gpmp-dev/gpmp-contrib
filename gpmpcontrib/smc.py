# --------------------------------------------------------------
# Authors:
#   Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
#   Julien Bect <julien.bect@centralesupelec.fr>
# Copyright (c) 2023, 2024 CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import time
from numpy.random import default_rng
import scipy.stats as stats
import gpmp.num as gnp


class ParticlesSetError(BaseException):
    def __init__(self, param_s, lower, upper):
        message = (
            "ParticlesSet: scaling parameter param_s in MH step out of range "
            "(value: {}, lower bound: {}, upper bound: {}).".format(
                param_s, lower, upper
            )
        )
        super().__init__(message)


class ParticlesSet:
    """
    A class representing a set of particles for Sequential Monte
    Carlo (SMC) simulation.

    This class provides elementary operations for initializing,
    reweighting, resampling, and moving particles.

    Parameters
    ----------
    box : array_like
        The domain box in which the particles are initialized.
    n : int, optional, default: 1000
        Number of particles.
    initial_distribution: str, optional, default: "randunif"
        Initial distribution for the particles.
    rng : numpy.random.Generator
        Random number generator.

    Attributes
    ----------
    n : int
        Number of particles.
    x : ndarray
        Current positions of the particles.
    logpx : ndarray
        Log-probabilities of the particles at their current positions.
    w : ndarray
        Weights of the particles.
    logpdf_function : callable
        Function to compute the log-probability density.
    param_s : float
        Scaling parameter for the perturbation step.
    resample_scheme : str
        Scheme for resampling ('multinomial' or 'residual').
    rng : numpy.random.Generator
        Random number generator.

    Methods
    -------
    particles_init(box, n)
        Initialize particles within the given box.
    set_logpdf(logpdf_function)
        Set the log-probability density function.
    reweight()
        Reweight the particles based on the log-probability density function.
    ess()
        Calculate the effective sample size (ESS) of the particles.
    resample()
        Resample the particles based on their weights.
    multinomial_resample()
        Resample using multinomial resampling.
    residual_resample()
        Resample using residual resampling.
    perturb()
        Perturb the particles by adding random noise.
    move()
        Perform a Metropolis-Hastings step and compute the acceptation rate.

    """

    def __init__(self, box, n=1000, initial_distribution="randunif", rng=default_rng()):
        """
        Initialize the ParticlesSet instance.
        """
        self.n = n  # Number of particles
        self.dim = len(box[0])
        self.logpdf_function = None
        self.rng = rng

        # Dictionary to hold parameters for the particle set
        self.particles_set_params = {
            "initial_distribution": initial_distribution,
            "resample_scheme": "residual",
            "param_s_initial_value": 0.5,  # Initial scaling parameter for MH perturbation
            "param_s_upper_bound": 10**4,
            "param_s_lower_bound": 10 ** (-3),
            # Jitter added to pertubation covariance matrix when it's not PSD
            "jitter_initial_value": 1e-16,
            "jitter_max_iterations": 10,
        }
        self.param_s = self.particles_set_params["param_s_initial_value"]
        self.resample_scheme = self.particles_set_params["resample_scheme"]

        # Initialize the particles.  Returns a tuple containing the
        # positions, log-probabilities, and weights of the particles
        self.x = None
        self.logpx = None
        self.w = None
        self.particles_init(
            box, n, method=self.particles_set_params["initial_distribution"]
        )

    def particles_init(self, box, n, method="randunif"):
        """Initialize particles within the given box.

        Parameters
        ----------
        box : array_like
            The domain box in which the particles are to be initialized.
        n : int
            Number of particles.
        method : str, optional
            Method for initializing particles. Currently, only
            'randunif' (uniform random) is supported. The option 'qmc'
            (quasi Monte-Carlo) will be supported in future versions.

        Returns
        -------
        tuple
            A tuple containing the positions, log-probabilities, and
            weights of the initialized particles.

        FIXME
        -----
        Implement more general initial densities

        """
        assert self.dim == len(
            box[0]
        ), "Box dimension does not match particles dimension"
        self.n = n

        # Initialize positions
        if method == "randunif":
            self.x = ParticlesSet.randunif(self.dim, self.n, box, self.rng)
        else:
            raise NotImplementedError(
                f"The method '{method}' is not supported. Currently, only 'randunif' is available."
            )

        # Initialize log-probabilities and weights
        self.logpx = gnp.zeros((n,))
        self.w = gnp.full((n,), 1 / n)

    def set_logpdf(self, logpdf_function):
        """
        Set the log-probability density function for the particles.

        Parameters
        ----------
        logpdf_function : callable
            Computes the log-probability density at given positions.
        """
        self.logpdf_function = logpdf_function

    def reweight(self):
        logpx_new = self.logpdf_function(self.x)
        self.w = self.w * gnp.exp(logpx_new - self.logpx)
        self.logpx = logpx_new

    def ess(self):
        """https://en.wikipedia.org/wiki/Effective_sample_size"""
        return gnp.sum(self.w) ** 2 / gnp.sum(self.w**2)

    def resample(self, debug=False):
        """
        Resample the particles based on the chosen resampling scheme.

        The resample method routes to either multinomial_resample or
        residual_resample according to self.resample_scheme.
        """
        if self.resample_scheme == "multinomial":
            self.multinomial_resample(debug=debug)
        elif self.resample_scheme == "residual":
            self.residual_resample(debug=debug)
        else:
            raise ValueError("Unknown resample scheme: {}".format(self.resample_scheme))

    def multinomial_resample(self, debug=False):
        """
        Resample using multinomial resampling.

        This method assigns offspring counts to particles according
        to a multinomial distribution.
        """
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        p = self.w / gnp.sum(self.w)
        try:
            counts = self.multinomial_rvs(self.n, p, self.rng)
        except Exception:
            extype, value, tb = __import__("sys").exc_info()
            __import__("traceback").print_exc()
            __import__("pdb").post_mortem(tb)

        if debug:
            print(
                f"Multinomial resample: proportion discarded = {gnp.sum(counts==0) / self.n} "
            )

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled = gnp.set_row2(x_resampled, i, self.x[j, :])
                logpx_resampled = gnp.set_elem1(logpx_resampled, i, self.logpx[j])
                counts = gnp.set_elem1(counts, j, counts[j] - 1)
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = gnp.full((self.n,), 1 / self.n)

    def residual_resample(self, debug=False):
        """
        Resample using residual resampling.

        This method reduces variance by first assigning a deterministic
        number of copies to each particle and then assigning the remainder
        via multinomial sampling.
        """
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        p = self.w / gnp.sum(self.w)
        N = self.n

        # Deterministic assignment: floor of expected counts
        counts_det = gnp.asint(gnp.floor(N * p))
        N_det = int(gnp.sum(counts_det))

        # Compute residuals
        residuals = N * p - counts_det
        N_residual = N - N_det

        # Multinomial step on residuals
        if N_residual > 0:
            try:
                counts_res = self.multinomial_rvs(
                    N_residual, residuals / N_residual, self.rng
                )
            except Exception:
                extype, value, tb = __import__("sys").exc_info()
                __import__("traceback").print_exc()
                __import__("pdb").post_mortem(tb)
        else:
            counts_res = gnp.zeros_like(counts_det)

        # Total counts
        counts = counts_det + counts_res

        if debug:
            print(
                f"Residual resample: proportion discarded = {gnp.sum(counts==0) / self.n} "
            )

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled = gnp.set_row2(x_resampled, i, self.x[j, :])
                logpx_resampled = gnp.set_elem1(logpx_resampled, i, self.logpx[j])
                counts = gnp.set_elem1(counts, j, counts[j] - 1)
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = gnp.full((self.n,), 1 / self.n)

    def perturb(self):
        """Perturb the particles by adding Gaussian noise.

        This method perturbs the current positions of the particles by
        applying a Gaussian random perturbation. The covariance matrix
        is computed from the particles' current positions and is
        scaled by the perturbation parameter `param_s`. This
        covariance matrix defines the spread of the Gaussian noise
        used to move the particles.

        """

        param_s_lower = self.particles_set_params["param_s_lower_bound"]
        param_s_upper = self.particles_set_params["param_s_upper_bound"]

        # Check if param_s is within bounds
        if self.param_s > param_s_upper or self.param_s < param_s_lower:
            raise ParticlesSetError(self.param_s, param_s_lower, param_s_upper)

        # Covariance matrix of the pertubation noise
        C = self.param_s * gnp.cov(self.x.reshape(self.n, -1).T)

        # Call ParticlesSet.multivariate_normal_rvs(C, self.n, self.rng)
        # with control on the possible degeneracy of C
        try:
            eps = ParticlesSet.multivariate_normal_rvs(C, self.n, self.rng)
            success = True
        except ValueError as e:
            # If the covariance matrix is not PSD, apply jittering to fix it
            print(f"Non-PSD covariance matrix encountered: {e}")
            success = False
            for i in range(
                self.particles_set_params["jitter_max_iterations"]
            ):  # Try iterations of jittering
                jitter = self.particles_set_params["jitter_initial_value"] * (10**i)
                C_jittered = C + jitter * np.eye(C.shape[0])  # Add jitter
                try:
                    eps = ParticlesSet.multivariate_normal_rvs(
                        C_jittered, self.n, self.rng
                    )
                    success = True
                    break
                except ValueError as inner_e:
                    print(f"Jittering attempt {i} failed: {inner_e}")

        if not success:
            raise RuntimeError(
                "Failed to generate samples after "
                + f"{self.particles_set_params['jitter_max_iterations']} jittering attempts. "
                + "Covariance matrix might still be non-PSD."
            )

        return self.x + eps.reshape(self.n, -1)

    def move(self):
        """
        Perform a Metropolis-Hastings step and compute the acceptation rate.

        This method perturbs the particles, computes the acceptation probabilities, and
        decides whether to move the particles to their new positions.

        Returns
        -------
        float
            Acceptation rate of the move.
        """
        # Perturb the particles
        y = self.perturb()
        logpy = self.logpdf_function(y)

        # Compute acceptation probabilities
        rho = gnp.minimum(1, gnp.exp(logpy - self.logpx))

        accepted_moves = 0  # Counter for accepted moves
        for i in range(self.n):
            if ParticlesSet.rand(self.rng) < rho[i]:
                # Update the particle position and log probability if the move is accepted
                self.x = gnp.set_row2(self.x, i, y[i, :])
                self.logpx = gnp.set_elem1(self.logpx, i, logpy[i])
                accepted_moves += 1

        # Compute the acceptation rate
        acceptation_rate = accepted_moves / self.n

        return acceptation_rate

    @staticmethod
    def rand(rng):
        return rng.uniform()

    @staticmethod
    def multinomial_rvs(n, p, rng):
        return gnp.asarray(stats.multinomial.rvs(n=n, p=p, random_state=rng))

    @staticmethod
    def multivariate_normal_rvs(C, n, rng):
        return gnp.asarray(
            stats.multivariate_normal.rvs(cov=C, size=n, random_state=rng)
        )

    @staticmethod
    def randunif(dim, n, box, rng):
        return gnp.asarray(stats.qmc.scale(rng.uniform(size=(n, dim)), box[0], box[1]))


class SMC:
    """Sequential Monte Carlo (SMC) sampler class.

    This class drives the SMC process using a set of particles,
    employing a strategy as described in
    Bect, J., Li, L., & Vazquez, E. (2017). "Bayesian subset simulation",
    SIAM/ASA Journal on Uncertainty Quantification, 5(1), 762-786.
    Available at: https://arxiv.org/abs/1601.02557

    Parameters
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int, optional, default: 1000
        Number of particles.
    initial_distribution: str, optional, default: "randunif"
        Initial distribution for the particles.
    rng : numpy.random.Generator
        Random number generator.

    Attributes
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int
        Number of particles.
    particles : ParticlesSet
        Instance of ParticlesSet class to manage the particles.

    Methods
    -------
    step(logpdf_parameterized_function, logpdf_param)
        Perform a single SMC step.
    move_with_controlled_acceptation_rate()
        Adjust the particles' movement to control the acceptation rate.

    """

    def __init__(self, box, n=2000, initial_distribution="randunif", rng=default_rng()):
        """
        Initialize the SMC sampler.
        """
        self.box = box
        self.n = n
        self.initial_distribution = initial_distribution
        self.particles = ParticlesSet(box, n, initial_distribution, rng)

        # Dictionary to hold MH algorithm parameters
        self.mh_params = {
            "mh_steps": 10,
            "acceptation_rate_min": 0.2,
            "acceptation_rate_max": 0.4,
            "adjustment_factor": 1.4,
            "adjustment_max_iterations": 50,
        }

        # Logging
        self.log = []  # Store the state logs
        self.stage = 0
        self.logging_current_ess = None
        self.logging_current_logpdf_param = None
        self.logging_target_logpdf_param = None
        self.logging_restart_iteration = 0
        self.logging_logpdf_param_sequence = []  # Sequence of logpdf_params in restart
        self.logging_acceptation_rate_sequence = []

    def _log_data(
        self,
        logpdf_param=None,
        ess=None,
        acceptation_rate=None,
        log_current_state_and_reinitialize=False,
    ):
        """
        Helper function to log data during the SMC process. It logs both incremental data
        like logpdf_param, ESS, and acceptation rate, as well as the full state at the end of a stage.

        Parameters
        ----------
        logpdf_param : float, optional
            The current logpdf parameter value being used in the SMC step.
        ess : float, optional
            Effective sample size (ESS) of the current particle set.
        acceptation_rate : float, optional
            Acceptation rate of the particle move step.
        log_current_state_and_reinitialize : bool, optional
            If True, logs the full current state of the SMC process.
        """
        # Incremental data logging
        if logpdf_param is not None:
            self.logging_current_logpdf_param = logpdf_param
        if ess is not None:
            self.logging_current_ess = ess
        if acceptation_rate is not None:
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

        # If log_current_state_and_reinitialize is True, log the full state
        if log_current_state_and_reinitialize:
            state = {
                "timestamp": time.time(),
                "stage": self.stage,
                "num_particles": self.n,
                "current_scaling_param": self.particles.param_s,
                "target_logpdf_param": self.logging_target_logpdf_param,
                "current_logpdf_param": self.logging_current_logpdf_param,
                "ess": self.logging_current_ess,
                "restart_iteration": self.logging_restart_iteration,
                "logpdf_param_sequence": self.logging_logpdf_param_sequence.copy(),
                "acceptation_rate_sequence": self.logging_acceptation_rate_sequence.copy(),
            }
            self.log.append(state)
            # Reinitialize acceptation_rate_sequence for the next stage
            self.logging_acceptation_rate_sequence = []

    def step(self, logpdf_parameterized_function, logpdf_param, debug=False):
        """
        Perform a single step of the SMC process.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density at
            given positions.
        logpdf_param: float
            Parameter value for the logpdf function (typically, a threshold).

        """

        # Set target density
        def logpdf(x):
            return logpdf_parameterized_function(x, logpdf_param)

        self.particles.set_logpdf(logpdf)

        # Reweight
        self.particles.reweight()

        # Log current logpdf_param and ESS
        self._log_data(logpdf_param=logpdf_param, ess=self.particles.ess())

        # Resample / move
        self.particles.resample(debug)

        self.move_with_controlled_acceptation_rate()

        for _ in range(self.mh_params["mh_steps"] - 1):
            # Additional moves if required
            acceptation_rate = self.particles.move()
            self._log_data(acceptation_rate=acceptation_rate)

        # Log state at the end of the step
        self._log_data(log_current_state_and_reinitialize=True)

        # Debug plot, if needed
        if debug:
            self.plot_particles()

    def step_with_possible_restart(
        self,
        logpdf_parameterized_function,
        initial_logpdf_param,
        target_logpdf_param,
        min_ess_ratio,
        p0,
        debug=False,
    ):
        """Perform an SMC step with the possibility of restarting the process.

        This method checks if the effective sample size (ESS) falls
        below a specified ratio, and if so, initiates a restart. The
        restart process reinitializes particles and recalculates
        logpdf_params to better target the desired distribution.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density of a
            given position.
        initial_logpdf_param : float
            The starting logpdf_param value for the restart process.
        target_logpdf_param : float
            The desired target logpdf_param value for the log-probability
            density.
        min_ess_ratio : float
            The minimum acceptable ratio of ESS to the total number of
            particles. If the ESS falls below this ratio, a restart is
            initiated.
        p0 : float
            The prescribed probability used in the restart method to
            compute the new logpdf_param.
        debug : bool, optional
            If True, prints debug information during the
            process. Default is False.
        """
        # Logging
        self.stage += 1
        self.logging_current_logpdf_param = target_logpdf_param
        self.logging_target_logpdf_param = target_logpdf_param

        # Set target density
        def logpdf(x):
            return logpdf_parameterized_function(x, target_logpdf_param)

        self.particles.set_logpdf(logpdf)

        # reweight
        self.particles.reweight()
        self.logging_current_ess = self.particles.ess()

        # restart?
        if self.logging_current_ess / self.n < min_ess_ratio:
            self.restart(
                logpdf_parameterized_function,
                initial_logpdf_param,
                target_logpdf_param,
                p0,
                debug=debug,
            )
            # Note: Logging will occur inside the restart method.

        else:
            # Resample
            self.particles.resample()

            # Move with control on acceptation rate
            self.move_with_controlled_acceptation_rate()

            # Additional moves if required
            for _ in range(self.mh_params["mh_steps"] - 1):
                acceptation_rate = self.particles.move()
                self.logging_acceptation_rate_sequence.append(acceptation_rate)

            # Logging
            self._log_data(log_current_state_and_reinitialize=True)

    def restart(
        self,
        logpdf_parameterized_function,
        initial_logpdf_param,
        target_logpdf_param,
        p0,
        debug=False,
    ):
        """
        Perform a restart method in SMC.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric probability density
        initial_logpdf_param : float
            Starting param value.
        target_logpdf_param : float
            Target param value.
        p0 : float
            Prescribed probability
        debug : bool
            If True, print debug information.
        """
        if debug:
            print("---- Restarting SMC ----")

        self._log_data(log_current_state_and_reinitialize=True)

        self.particles.particles_init(
            self.box, self.n, method=self.initial_distribution
        )

        current_logpdf_param = initial_logpdf_param

        self.logging_logpdf_param_sequence = [initial_logpdf_param]

        while current_logpdf_param != target_logpdf_param:
            next_logpdf_param = self.compute_next_logpdf_param(
                logpdf_parameterized_function,
                current_logpdf_param,
                target_logpdf_param,
                p0,
                debug,
            )

            self.logging_restart_iteration += 1
            self.logging_logpdf_param_sequence.append(next_logpdf_param)

            self.step(logpdf_parameterized_function, next_logpdf_param, debug=debug)

            current_logpdf_param = next_logpdf_param

        # Logging reinitialization
        self.logging_logpdf_param_sequence = []
        self.logging_restart_iteration = 0

    def move_with_controlled_acceptation_rate(self, debug=False):
        """
        Adjust the particles' movement to maintain the acceptation
        rate within specified bounds.  This method dynamically adjusts
        the scaling parameter based on the acceptation rate to ensure
        efficient exploration of the state space.

        """
        iteration_counter = 0
        self.logging_acceptation_rate_sequence = []  # Logging
        while iteration_counter < self.mh_params["adjustment_max_iterations"]:
            iteration_counter += 1

            acceptation_rate = self.particles.move()

            # Logging
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

            if debug:
                print(f"Acceptation rate = {acceptation_rate}")

            if acceptation_rate < self.mh_params["acceptation_rate_min"]:
                self.particles.param_s /= self.mh_params["adjustment_factor"]
                continue

            if acceptation_rate > self.mh_params["acceptation_rate_max"]:
                self.particles.param_s *= self.mh_params["adjustment_factor"]
                continue

            break

    def _compute_p_value(self, logpdf_function, new_logpdf_param, current_logpdf_param):
        """
        Compute the mean value of the exponentiated difference in
        log-probability densities between two logpdf_params.

        .. math::

            \\frac{1}{n} \\sum_{i=1}^{n} \\exp(logpdf_function(x_i, new_logpdf_param)
            - logpdf_function(x_i, current_logpdf_param))

        Parameters
        ----------
        logpdf_function : callable
            Function to compute log-probability density.
        new_logpdf_param : float
            The new logpdf_param value.
        current_logpdf_param : float
            The current logpdf_param value used as a reference.

        Returns
        -------
        float
            Computed mean value.

        """
        return gnp.mean(
            gnp.exp(
                logpdf_function(self.particles.x, new_logpdf_param)
                - logpdf_function(self.particles.x, current_logpdf_param)
            )
        )

    def compute_next_logpdf_param(
        self,
        logpdf_parameterized_function,
        current_logpdf_param,
        target_logpdf_param,
        p0,
        debug=False,
    ):
        """
        Compute the next logpdf_param using a dichotomy method.

        This method is part of the restart strategy. It computes a
        logpdf_param for the parameter of the
        logpdf_parameterized_function, ensuring a controlled migration
        of particles to the next stage. The parameter p0 corresponds
        to the fraction of moved particles that will be in the support
        of the target density.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric log-probability density.
        current_logpdf_param : float
            Starting logpdf_param value.
        target_logpdf_param : float
            Target logpdf_param value.
        p0 : float
            Prescribed probability.
        debug : bool
            If True, print debug information.

        Returns
        -------
        float
            Next computed logpdf_param.

        """
        tolerance = 0.05
        low = current_logpdf_param
        high = target_logpdf_param

        # Check if target_logpdf_param can be reached with p >= p0
        p_target = self._compute_p_value(
            logpdf_parameterized_function, target_logpdf_param, current_logpdf_param
        )
        if p_target >= p0:
            if debug:
                print("Target logpdf_param reached.")
            return target_logpdf_param

        while True:
            mid = (high + low) / 2
            p = self._compute_p_value(
                logpdf_parameterized_function, mid, current_logpdf_param
            )

            if debug:
                print(
                    f"Search: p = {p:.2f} / p0 = {p0:.2f}, "
                    + f"test logpdf_param = {mid}, "
                    + f"current = {current_logpdf_param}, "
                    + f"target = {target_logpdf_param}"
                )

            if abs(p - p0) < tolerance:
                break

            if p < p0:
                high = mid
            else:
                low = mid

        return mid

    def plot_state(self):
        """Plot the state of the SMC process over different stages.

        It includes visualizations of logpdf_params, effective sample
        size (ESS), and acceptation rates.
        """

        import matplotlib.pyplot as plt

        log_data = self.log

        def make_stairs(y):
            x_stairs = []
            y_stairs = []
            for i in range(len(y)):
                x_stairs.extend([i, i + 1])
                y_stairs.extend([y[i], y[i]])
            return x_stairs, y_stairs

        # Initializing lists to store data
        stages = []
        target_logpdf_params = []
        current_logpdf_params = []
        ess_values = []
        acceptation_rates = []
        stage_changes = []  # To mark the stages where change occurs

        # Extracting and replicating data according to the length of 'acceptation_rate_sequence' in each log entry
        for idx, entry in enumerate(log_data):
            ar_length = len(entry["acceptation_rate_sequence"])
            if ar_length == 0:
                entry["acceptation_rate_sequence"] = [0.0]
                ar_length = 1

            stages.extend([entry["stage"]] * ar_length)
            target_logpdf_params.extend([entry["target_logpdf_param"]] * ar_length)
            current_logpdf_params.extend([entry["current_logpdf_param"]] * ar_length)
            ess_values.extend([entry["ess"]] * ar_length)
            acceptation_rates.extend(entry["acceptation_rate_sequence"])

        # Plotting
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Time")
        ax1.set_ylabel("logpdf_param", color=color)
        t, target_logpdf_params = make_stairs(target_logpdf_params)
        t, current_logpdf_params = make_stairs(current_logpdf_params)
        ax1.plot(
            t,
            target_logpdf_params,
            label="Target logpdf_param",
            color="red",
            linestyle="dashed",
        )
        ax1.plot(
            t,
            current_logpdf_params,
            label="Current logpdf_param",
            color="red",
            linestyle="solid",
        )
        (ymin, ymax) = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 1.2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend(loc="upper left")

        # Adding vertical lines for stage changes
        last_stage = 0
        for idx, stage in enumerate(stages):
            if stage > last_stage:
                plt.axvline(x=idx, color="gray", linestyle="dashed")
                last_stage = stage

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("ESS", color=color)
        t, ess_values = make_stairs(ess_values)
        ax2.plot(t, ess_values, label="ESS", color=color)
        ax2.set_ylim(0.0, self.n)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="upper right")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color = "tab:green"
        ax3.set_ylabel("Acceptation Rate", color=color)
        ax3.plot(
            acceptation_rates, label="Acceptation Rate", color=color, linestyle="dotted"
        )
        ax3.set_ylim(0.0, 1.0)
        ax3.tick_params(axis="y", labelcolor=color)
        ax3.legend(loc="lower right")

        fig.tight_layout()
        plt.title("SMC Process State Over Stages")
        plt.show()

    def plot_particles(self):

        from gpmpcontrib.plot.visualization import plotmatrix

        plotmatrix(
            gnp.hstack((self.particles.x, self.particles.logpx.reshape(-1, 1))),
            self.particles.logpx,
        )
