This directory contains:

1) OPLSUA_ff.lt -- OPLS-UA parameterization of the system
2) run_simulation.py -- python run script for the CBMC simulation.
3) system.data -- data file for an initially random 50% PEG 50% DDT trial on a 2.8 nm Au NP. Can be visualized with Ovito.
4) system.in -- defines which files to read for the Init section, Atom Definition section, and Settings section. The run section is handled by run_simulation.py and the CBMC model.
5) system.in.init -- contains LAMMPS system initialization parameters.
6) system.in.settings -- contains pair, bond, angle, and dihedral coeffs
7) system.lt -- input file for moltemplate used in the creation of the system.data file


LAMMPS version stable_29Oct2020 was used
