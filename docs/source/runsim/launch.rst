Launching VirtualLab
====================

Command Line Interface
######################

If **VirtualLab** has been installed correctly, the main program will have been added to your system :bash:`<path>`. In this case, it is possible to call **VirtualLab** from the terminal (also known as command line interface, CLI) or a bash script from any location in your system. To facilitate automation, **VirtualLab** has purposefully been designed to run without a graphical user interface (GUI).

.. _usage:

Usage of 'VirtualLab':
  VirtualLab -f <path>

Options:
   | :bash:`-f <path>` Where <path> points to the location of the python *Run* file (relative to the current working directory).
   | :bash:`-k <Name=Value>` Overwrite the value specified for variables/keyword arguments specified in the *Run* file.
   | :bash:`-h` Display the help menu.

* The default behaviour is to exit if no :bash:`<path>` is given.

The `RunFiles <structure.html#runfiles>`_ directory contains the *Run* files used to specify the settings of virtual experiments. These files contain the relevant information required to run analysis using **VirtualLab**, such as the type of virtual experiment or the mode in which it is run.

Containers
##########

Launching **VirtualLab** with a container can be carried out in several ways:

* Interactively
* Non-interactively
* Batch mode

Containers are write-protected. This means that, although they have an internal file structure, the user must tell the container where to read input files and write outputs by mounting (or binding) directories. The examples shown here demonstrate how to achieve this.

Interactive
~~~~~~~~~~~

These commands are examples of how to shell in to the **VirtualLab** container for an interactive session. Once inside the container, then **VirtualLab** may be launched as normally from a CLI command, detailed in :ref:`'Usage' <usage>`.

**Docker** ::

  sudo docker run -it \
  --mount type=bind, \
  /tmp=/tmp, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  ibsim/virtuallab:latest

**Singularity** ::

  singularity shell --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  VirtualLab.sif

.. note::
  :bash:`$USER` should be replaced by the username of the host user.

  Code_Aster requires write access to the :file:`/tmp` and :file:`../flasheur` directories. The user should create a local :file:`../flasheur` directory before launching the container.

  The :file:`..Input` directory is where the user's custom simulation files should be kept and :file:`..Output` is where the simulation results are written. These can be customised as desired on the host system.

Non-Interactive
~~~~~~~~~~~~~~~

To launch **VirtualLab** from outside the container, CLI commands as detailed in :ref:`'Usage' <usage>` must be sent as arguments.

**Docker** ::

  sudo docker run -it \
  --mount type=bind, \
  /tmp=/tmp, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  ibsim/virtuallab:latest \
  "VirtualLab.sif -f /home/$USER/Input/RunFile.py"

**Singularity** ::

  singularity exec --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  VirtualLab.sif -f /home/$USER/Input/RunFile.py

Batch Mode
~~~~~~~~~~

In batch mode, rather than launching the command directly it is normally entered within a script which is sent to a job scheduler (or workload manager). The command is then out in a queue to be executed when the requested resources become available. Singularity is often the platform of choice for shared HPC resources because it can be used without the user needing admin privileges. This is a Singularity example for the `slurm <https://slurm.schedmd.com/>`_ job scheduler on Supercomputing Wales's sunbird system.

**Singularity** ::

  #!/bin/bash --login
  #SBATCH --job-name=VirtualLab
  #SBATCH --output=VL.out.%J
  #SBATCH --error=VL.err.%J
  #SBATCH --time=0-00:20
  #SBATCH --ntasks=16
  #SBATCH --mem-per-cpu=1000
  #SBATCH --ntasks-per-node=16

  module load singularity/3.6.3

  singularity exec --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  VirtualLab.sif -f /home/$USER/Input/RunFile.py

Using a GUI
~~~~~~~~~~~

Although **VirtualLab** is predominantly set up to be used without a GUI, the user may sometimes wish to use the GUI for reasons such as checking meshes or simulation results interactively. In this event, it is possible to use either platform to launch the relevant software from within the container and interact with it as if it were installed on the local machine.

These commands are examples of how to launch the GUI interface of salome with the **VirtualLab** container.

**Docker** ::

  sudo docker run \
  --mount type=bind, \
  /tmp=/tmp, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  --net=host --env="DISPLAY" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  ibsim/virtuallab:latest salome

**Singularity** ::

  singularity exec --contain --bind \
  /tmp, \
  /dev, \
  /home/$USER/flasheur:/home/ibsim/flasheur, \
  /home/$USER/Input:/home/ibsim/VirtualLab/Intput, \
  /home/$USER/Output:/home/ibsim/VirtualLab/Output \
  salome

Virtual Machines
################

Once logged into the VM the user is presented with an Ubuntu desktop environment which can be used identically to a native Linux installation. That is, with the use of the CLI in a terminal **VirtualLab** may be launched as detailed in :ref:`'Usage' <usage>`.
