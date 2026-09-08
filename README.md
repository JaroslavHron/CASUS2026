# CASUS2026
Introduction to finite elements for multiphase reaction diffusion systems using Firedrake

https://www.firedrakeproject.org/  +  https://petsc.org/

copy the tutorial files `cp -r /home/tut10/hron/CASUS2026 ./` or clone the repository `https://github.com/JaroslavHron/CASUS2026`

How to run:
- prepared enviroment on `rosi`: login to `rosi` and run `source /home/tut10/hron/CASUS2026/setup.sh`
- google colab: To use Colab, go to https://colab.research.google.com
  and then paste the code in colab.txt into a cell.
- other options - see https://www.firedrakeproject.org/install.html

to run on rosi with the prepared enviroment use `srun -n 1 -u python 00_simple_poisson.py`
