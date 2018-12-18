"""
This is an example to train a task with TNPG algorithm.

Here it runs CartpoleEnv on TNPG with 40 iterations.
Note this is not a suggested hyperparameter setting for CartpoleEnv.

Results:
    AverageDiscountedReturn: 51
    RiseTime: itr 2
"""
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.experiment import run_experiment
from garage.tf.algos import TNPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def run_task(*_):
    """Wrap TNPG training task in the run_task function."""
    env = TfEnv(normalize(CartpoleEnv()))

    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TNPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        optimizer_args=dict(reg_coeff=5e-2))
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
