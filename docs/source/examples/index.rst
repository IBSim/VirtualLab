Tutorials
=========

The tutorials in this section provide an overview in to running a 'virtual experiment' using **VirtualLab**.

These examples give an overview of:

 * how meshes and simulations can be created parametrically without the need for a graphical user interface (GUI).
 * the available options during simulations that give the user a certain degree of flexibility.
 * methods of debugging.
 * **VirtualLab**'s in-built pre and post-processing capabilities.

There is a tutorial for each of **VirtualLab**'s `virtual experiments <virtual_exp.html>`_.

Before starting the tutorials, it is advised to first read the `Code Structure <structure.html>`_ and `Running VirtualLab <runsim/index.html>`_ sections for an overview of **VirtualLab**. Then it is best to work through the tutorials in order as each will introduce new tools that **VirtualLab** has to offer.

These tutorials assume a certain level of pre-existing knowledge about the finite element method (FEM) as a prerequisite. Additionally, these tutorial do not aim to teach users on how to use the **Code_Aster** software itself, only its implementation as part of **VirtualLab**. For **Code_Aster** tutorials we recommend the excellent website `feaforall.com <https://feaforall.com/salome-meca-code-aster-tutorials/>`_. Because **VirtualLab** can be run completely from scripts, without opening the **Code_Aster** graphical user interface (GUI), **VirtualLab** can be used without being familiar with **Code_Aster**.

'Setting up data for visualisation' is outside the scope of these tutorials. The **ParaVis** module within **SALOME** is based on another piece of open-source software called **ParaView**. If you would like to learn more about how to visualise datasets with **SALOME** it is recommended that you follow the tutorials available on `feaforall.com <https://feaforall.com/salome-meca-code-aster-tutorials/>`_ and `paraview.org <https://www.paraview.org/Wiki/The_ParaView_Tutorial>`_.

Each tutorial is structured as follows: firstly the experimental test sample (i.e. geometry domain) is introduced followed by an overview of the boundary conditions and constraints being applied to the sample to emulate the physical experiment. Then a series of tasks are described to guide the user through various stages with specific learning outcomes.

Simulations are initiated by launching **VirtualLab** in the command line with a `RunFile <runsim/runfile.html>`_ specified using the flag ``-f``::

	VirtualLab -f </PATH/TO/RUNFILE>

If running **VirtualLab** with a container, follow the `appropriate guidance <runsim/launch.html#containers>`_ for your setup.

:file:`Run.py` in the **VirtualLab** top level directory is a template of a *RunFile* which is used to launch **VirtualLab**. Additional examples of *RunFiles* are available in the `RunFiles <structure.html#runfiles>`_ directory, where the file :file:`RunTutorials.py` is located which will be used for these tutorials.

.. note::

    To help with following the tutorials, certain `keyword arguments <https://docs.python.org/3/glossary.html>`_ (referred to as kwargs)  have been changed from their default values in :file:`RunTutorials.py`. In `VirtualLab.Settings <runsim/runfile.html#virtuallab-settings>`_ *Mode* has been changed to 'Interactive', while in `VirtualLab.Sim <runsim/runfile.html#virtuallab-sim>`_ *ShowRes* is set to :code:`True`.

.. tip::

    You may wish to save a backup of :file:`RunTutorials.py` such that you may return to the default template without needing to re-download it.

.. toctree::
    :maxdepth: 2

    tensile
    lfa
    hive
    ibsim
