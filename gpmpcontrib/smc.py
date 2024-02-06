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
    rng : numpy.random.Generator
        Random number generator.
    n : int, optional
        Number of particles (default is 1000).

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
    perturb()
        Perturb the particles by adding random noise.
    move()
        Perform a Metropolis-Hastings step and compute the acceptation rate.

    """

    def __init__(self, box, rng=default_rng(), n=1000):
        """
        Initialize the ParticlesSet instance.
        """
        self.n = n  # Number of particles
        self.dim = len(box[0])
        self.logpdf_function = None
        self.rng = rng

        # Initialize the particles.  Returns a tuple containing the
        # positions, log-probabilities, and weights of the particles
        self.x = None
        self.logpx = None
        self.w = None
        self.particles_init(box, n)

        # Metropolis-Hastings ingredients
        self.param_s = 0.05  # Default scaling parameter for perturbation


    def particles_init(self, box, n):
        """Initialize particles within the given box.

        Parameters
        ----------
        box : array_like
            The domain box in which the particles are to be initialized.
        n : int
            Number of particles.

        Returns
        -------
        tuple
            A tuple containing the positions, log-probabilities, and
            weights of the initialized particles.
        """
        assert(self.dim == len(box[0]))
        self.n = n
        self.x = gnp.asarray(self.randunif(box))
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

    def resample(self):
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        p = self.w / gnp.sum(self.w)
        try:
            counts = self.multinomial_rvs(p)
        except:
            extype, value, tb = __import__("sys").exc_info()
            __import__("traceback").print_exc()
            __import__("pdb").post_mortem(tb)

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled = gnp.set_row2(x_resampled, i, self.x[j, :])
                logpx_resampled = gnp.set_elem1(logpx_resampled, i, self.logpx[j])
                counts[j] -= 1
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = gnp.full((self.n,), 1 / self.n)

    def perturb(self):
        C = self.param_s * gnp.cov(self.x.reshape(self.x.shape[0], -1).T)
        eps = gnp.asarray(self.multivariate_normal_rvs(C))

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
            if gnp.asarray(self.rng.uniform())[0] < rho[i]:
                # Update the particle position and log probability if the move is accepted
                self.x = gnp.set_row2(self.x, i, y[i, :])
                self.logpx = gnp.set_elem1(self.logpx, i, logpy[i])
                accepted_moves += 1

        # Compute the acceptation rate
        acceptation_rate = accepted_moves / self.n

        return acceptation_rate

    def multinomial_rvs(self, p):
        return stats.multinomial.rvs(
            n=self.n, p=p, random_state=self.rng)

    def multivariate_normal_rvs(self, C):
        return stats.multivariate_normal.rvs(
            cov=C, size=self.n, random_state=self.rng)

    def randunif(self, box):
        return stats.qmc.scale(
            self.rng.uniform(size=(self.n, self.dim)), box[0], box[1])

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
    n : int, optional
        Number of particles (default is 1000).
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
    step(logpdf_parameterized_function, u_target)
        Perform a single SMC step.
    move_with_controlled_acceptation_rate()
        Adjust the particles' movement to control the acceptation rate.

    """

    def __init__(self, box, n=1000, rng=default_rng()):
        """
        Initialize the SMC sampler.
        """
        self.box = box
        self.n = n
        self.particles = ParticlesSet(box, rng, n)

        # Dictionary to hold MH algorithm parameters
        self.mh_params = {
            "mh_steps": 5,
            "acceptation_rate_min": 0.4,
            "acceptation_rate_max": 0.6,
            "adjustment_factor": 1.4,
            "adjustment_max_iterations": 50,
        }

        # Logging
        self.log = []  # Store the state logs
        self.stage = 0
        self.logging_current_ess = None
        self.logging_current_threshold = None
        self.logging_target_threshold = None
        self.logging_restart_iteration = 0
        self.logging_threshold_sequence = []  # Sequence of thresholds in restart
        self.logging_acceptation_rate_sequence = []

    def step(self, logpdf_parameterized_function, u_target):
        """
        Perform a single step of the SMC process.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density of a
            given position.
        u_target: float
            Parameter value for the logpdf function.

        """
        # Set target density
        logpdf = lambda x: logpdf_parameterized_function(x, u_target)
        self.particles.set_logpdf(logpdf)

        # Reweight
        self.particles.reweight()
        self.logging_current_ess = self.particles.ess()

        # Resample / move
        self.particles.resample()
        self.move_with_controlled_acceptation_rate()
        for _ in range(self.mh_params["mh_steps"] - 1):
            # Additional moves if required
            acceptation_rate = self.particles.move()
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

        # Logging
        self.logging_current_threshold = u_target
        self.log_state()

        # Debug plot, if needed
        debug = False
        if debug:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(self.particles.x, self.particles.logpx, ".")
            plt.show()

    def step_with_possible_restart(
        self,
        logpdf_parameterized_function,
        initial_threshold,
        target_threshold,
        min_ess_ratio,
        p0,
        debug=False,
    ):
        """Perform an SMC step with the possibility of restarting the process.

        This method checks if the effective sample size (ESS) falls
        below a specified ratio, and if so, initiates a restart. The
        restart process reinitializes particles and recalculates
        thresholds to better target the desired distribution.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density of a
            given position.
        initial_threshold : float
            The starting threshold value for the restart process.
        target_threshold : float
            The desired target threshold value for the log-probability
            density.
        min_ess_ratio : float
            The minimum acceptable ratio of ESS to the total number of
            particles. If the ESS falls below this ratio, a restart is
            initiated.
        p0 : float
            The prescribed probability used in the restart method to
            compute the new threshold.
        debug : bool, optional
            If True, prints debug information during the
            process. Default is False.
        """
        # Logging
        self.stage += 1
        self.logging_current_threshold = target_threshold
        self.logging_target_threshold = target_threshold

        # Set target density
        logpdf = lambda x: logpdf_parameterized_function(x, target_threshold)
        self.particles.set_logpdf(logpdf)

        # reweight
        self.particles.reweight()
        self.logging_current_ess = self.particles.ess()

        # restart?
        if self.logging_current_ess / self.n < min_ess_ratio:
            self.restart(
                logpdf_parameterized_function, initial_threshold, target_threshold, p0
            )
            # Note: Logging will occur inside the restart method.
        else:
            # resample / move
            self.particles.resample()
            self.move_with_controlled_acceptation_rate()
            for _ in range(self.mh_params["mh_steps"] - 1):
                # Additional moves if required
                acceptation_rate = self.particles.move()
                self.logging_acceptation_rate_sequence.append(acceptation_rate)
            # Logging
            self.log_state()

    def restart(
        self,
        logpdf_parameterized_function,
        initial_threshold,
        final_threshold,
        p0,
        debug=False,
    ):
        """
        Perform a restart method in SMC.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric probability density
        initial_threshold : float
            Starting threshold value.
        final_threshold : float
            Target threshold value.
        p0 : float
            Prescribed probability
        debug : bool
            If True, print debug information.
        """
        if debug:
            print("---- Restarting SMC ----")

        self.log_state()  # Log current and target thresholds, ess

        self.particles.particles_init(self.box, self.n)
        current_threshold = initial_threshold

        self.logging_threshold_sequence = [initial_threshold]

        while current_threshold != final_threshold:
            next_threshold = self.compute_next_threshold(
                logpdf_parameterized_function,
                current_threshold,
                final_threshold,
                p0,
                debug,
            )
            self.logging_restart_iteration += 1
            self.logging_threshold_sequence.append(next_threshold)

            self.step(logpdf_parameterized_function, next_threshold)

            current_threshold = next_threshold

        # Logging reinitialization
        self.logging_threshold_sequence = []
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

    def _compute_p_value(self, logpdf_function, threshold, initial_threshold):
        """
        Compute the mean value of the exponentiated difference in
        log-probability densities between two thresholds.

        .. math::

            \\frac{1}{n} \\sum_{i=1}^{n} \\exp(logpdf_function(x_i, threshold)
            - logpdf_function(x_i, initial_threshold))

        Parameters
        ----------
        logpdf_function : callable
            Function to compute log-probability density.
        threshold : float
            The current threshold value.
        initial_threshold : float
            The initial threshold value used as a reference.

        Returns
        -------
        float
            Computed mean value.

        """
        return gnp.mean(
            gnp.exp(
                logpdf_function(self.particles.x, threshold)
                - logpdf_function(self.particles.x, initial_threshold)
            )
        )

    def compute_next_threshold(
        self,
        logpdf_parameterized_function,
        initial_threshold,
        final_threshold,
        p0,
        debug=False,
    ):
        """
        Compute the next threshold using a dichotomy method.

        This method is part of the restart strategy. It computes a
        threshold for the parameter of the
        logpdf_parameterized_function, ensuring a controlled migration
        of particles to the next stage. The parameter p0 corresponds
        to the fraction of moved particles that will be in the support
        of the target density.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric log-probability density.
        initial_threshold : float
            Starting threshold value.
        final_threshold : float
            Target threshold value.
        p0 : float
            Prescribed probability.
        debug : bool
            If True, print debug information.

        Returns
        -------
        float
            Next computed threshold.

        """
        tolerance = 0.05
        low = initial_threshold
        high = final_threshold

        # Check if final_threshold can be reached with p >= p0
        p_final = self._compute_p_value(
            logpdf_parameterized_function, final_threshold, initial_threshold
        )
        if p_final >= p0:
            if debug:
                print("Final threshold reached.")
            return final_threshold

        while True:
            mid = (high + low) / 2
            p = self._compute_p_value(
                logpdf_parameterized_function, mid, initial_threshold
            )

            if debug:
                print(
                    f"Search: p = {p}, "
                    + f"current threshold = {mid}, "
                    + f"initial = {initial_threshold}, "
                    + f"target = {final_threshold}"
                )

            if abs(p - p0) < tolerance:
                break

            if p < p0:
                high = mid
            else:
                low = mid

        return mid

    def log_state(self):
        """
        Log the current state of the SMC process.
        """
        state = {
            "timestamp": time.time(),
            "stage": self.stage,
            "num_particles": self.n,
            "current_scaling_param": self.particles.param_s,
            "target_threshold": self.logging_target_threshold,
            "current_threshold": self.logging_current_threshold,
            "ess": self.logging_current_ess,
            "restart_iteration": self.logging_restart_iteration,
            "threshold_sequence": self.logging_threshold_sequence.copy(),
            "acceptation_rate_sequence": self.logging_acceptation_rate_sequence.copy(),
        }
        self.log.append(state)
        # reinitialize acceptation_rate_sequence
        self.logging_acceptation_rate_sequence = []

    def plot_state(self):
        """Plot the state of the SMC process over different stages.

        It includes visualizations of thresholds, effective sample
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
        target_thresholds = []
        current_thresholds = []
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
            target_thresholds.extend([entry["target_threshold"]] * ar_length)
            current_thresholds.extend([entry["current_threshold"]] * ar_length)
            ess_values.extend([entry["ess"]] * ar_length)
            acceptation_rates.extend(entry["acceptation_rate_sequence"])

        # Plotting
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Threshold", color=color)
        t, target_thresholds = make_stairs(target_thresholds)
        t, current_thresholds = make_stairs(current_thresholds)
        ax1.plot(
            t,
            target_thresholds,
            label="Target Threshold",
            color="red",
            linestyle="dashed",
        )
        ax1.plot(
            t,
            current_thresholds,
            label="Current Threshold",
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
