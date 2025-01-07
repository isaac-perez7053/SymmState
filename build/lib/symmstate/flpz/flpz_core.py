from symmstate._symm_state_core import SymmStateCore


class FlpzCore(SymmStateCore):
    """
    The base class to the Flpz subprogram
    """

    def __init__(
        self,
        name="catio3",
        num_datapoints="10",
        abi_file="abifile.abi",
        min_amp=0,
        max_amp=0.5,
    ):
        self.name = name
        self.num_datapoints = num_datapoints
        self.abi_file = abi_file
        self.min_amp = min_amp
        self.max_amp = max_amp
