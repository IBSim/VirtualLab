# VirtualLab

**VirtualLab** is a modular platform which enables the user to run simulations of physical laboratory experiments, i.e., their 'virtual counterparts'.

The motivation for creating a virtual laboratory is manyfold, for example:

* Planning and optimisation of physical experiments.
* Ability to directly compare experimental and simulation data, useful to better understand both physical and virtual methods.
* Augment sparse experimental data with simulation data for increased insight.
* Generating synthetic data to train machine learning models.

The software is mostly written in python, and is fully parametrised such that it can be run in 'batch mode', i.e., non-interactively, via the command line. This is in order to facilitate automation and so that many virtual experiments can be conducted in parallel.

Due to the modularity of the platform, by nature, **VirtualLab** is continually expanding. The bulk of the 'virtual experiments' currently included are carried out in the FE solver [Code_Aster](https://www.code-aster.org/). However, there are also modules to simulate [X-ray computed tomography](https://gvirtualxray.fpvidal.net/), [irradiation damage of materials](https://github.com/giacomo-po/MoDELib) and [electromagnetics](https://ruben-otin.blogspot.com/2015/04/ruben-otin-software-ruben-otin-april-19.html).

For installation instructions please see the [documentation pages](https://virtuallab.readthedocs.io/).

   Copyright 2020 IBSim Group.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
