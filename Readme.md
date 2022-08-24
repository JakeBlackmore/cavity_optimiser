# Cavity Optimiser
python module to optimise micro-cavities for the QCS & Oxford University cavity-ion network project.

This code is a re-write of the version originally produced by Shaobo Gao, it includes several changes to improve the physicality of the results.

The main user-land functions are the Cavity object and the optimizer function both contained within `core`.

Full [documentation](https://cavity_optimiser.readthedocs.io) is available.

[Example](EXAMPLE) optimiser scripts are included.

# Installation

The code can be executed entirely with the `core.py` module, which can be imported using the common syntax:
`import core`

Alternatively the code can be installed globally using the [included](dist) .whl and .tar.gz files and imported as
`from cavity import core`

# Requirements
- python > 3.7
- numpy > 1.20
- scipy > 1.8

# License

This code is licensed under the GNU GPLv3. A copy can be found [here](LICENSE).

If you use our work for academic purposes please cite:

S. Gao *et al.*, Optimisation of Scalable Ion-Cavity Interfaces for Quantum Photonic Networks, [Arxiv *e-prints* 2112.05795](https://arxiv.org/abs/2112.05795)
