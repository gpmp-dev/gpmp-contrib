"""
Multi-output deterministic or stochastic computer experiments

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import gpmp.num as gnp


class ComputerExperiment:
    def __init__(self, input_dim, input_box, output_dim, f, fname=None):
        """Computer experiment setting

        Parameters
        ----------
        input_dim: integer
            Dimension of the input space
        input_box: list of list
            Input domain
            [[x^min_1, ..., x^min_{input_dim}], [x^max_1, ..., x^max_{input_dim}]]
        self.output_dim
            Number of outputs
        f: function, list of functions
            Functions to be evaluated
        fname: string
            Function name
        foptargs: list
            Optional parameters
        """
        self.input_dim = input_dim
        self.input_box = input_box
        self.output_dim = output_dim
        self.f = f
        self.fname = fname

    def eval(self, x):

        n = x.shape[0]

        if isinstance(self.f, list):
            z_ = []
            for i in range(self.output_dim):
                z_.append(self.f[i](x).reshape((n, 1)))
            z = gnp.numpy.hstack(z_)

        else:
            z = self.f(x).reshape((n, self.output_dim))

        return z


class StochasticComputerExperiment(ComputerExperiment):
    def __init__(
        self, input_dim, input_box, output_dim, f, simulated_noise_variance=None
    ):

        # problem setting
        super().__init__(
            input_dim,  # dim search space
            input_box,  # box
            output_dim,  # dim output
            f,  # function or list of functions
        )

        # simulated homoscedastic noise variance for each output
        self.simulated_noise_variance = simulated_noise_variance  # ndarray (output_dim)

    def eval(self, x, simulated_noise_variance=True, batch_size=1):
        """
        Call a particular problem given an input

        Arguments
            x: input
            simulated_noise_variance: None, True, ndarray
        """
        n = x.shape[0]

        if simulated_noise_variance is True:
            simulated_noise_variance = self.simulated_noise_variance

        # Build z with repetitions -- ndarray(n x output_dim x batch_size)

        if simulated_noise_variance is not None:
            z = gnp.numpy.tile(
                    super().eval(x).reshape(n, self.output_dim, 1), (1, 1, batch_size)
            )  # n x output_dim x batch_size

            # For each output i, add noise if simulated_noise_variance[i] > 0
            for i in range(self.output_dim):
                if simulated_noise_variance[i] > 0.0:
                    z[:, i, :] = z[:, i, :] + \
                        gnp.numpy.sqrt(self.simulated_noise_variance[i]) * gnp.numpy.random.randn(n, batch_size)
        else:
            raise RuntimeError(
                "Dealing with true stochastic simulator is not implemented yet"
            )

        if batch_size == 1:
            return z.reshape(n, self.output_dim)  # drop last dimension if batch_size==1
        else:
            return z
