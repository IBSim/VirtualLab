.. role:: bash(code)
   :language: bash

Adding to VirtualLab
====================

**VirtualLab** has been designed so that adding work is as easy as possible. There are four ways in which new work can be added:
 1. New analysis scripts,
 2. New methods of performing analysis,
 3. New virtual experiments,
 4. Containers with new software or code.

Description on how work can be added to these are discussed below, along with the best practice for adding your work to the **VirtualLab** repository.

Scripts
*******

The easiest way to add to **VirtualLab** is by creating scripts for experiments and methods that already exist. For example, to create a new mesh script 'NewComponent' for the tensile experiment one would need to create the file :file:`Scripts/Experiments/Tensile/Mesh/NewComponent.py` which describes the steps that **SALOME** must follow to create a CAD geometry and mesh. This file can then be used by specifying it as the 'file' attribute to the 'Mesh' namespace, e.g. Mesh.File = 'NewComponent' in the parameter file.

Similarly 'Sim' and 'DA' scipts can be created and placed in the relevant directories in the experiment directory.

Experiments
***********

Adding a new experiment to **VirtualLab** will require creating a new experiment directory :file:`Scripts/Experiments/#ExpName`, where #ExpName is the name of the experiment. Within this experiment directory, sub-directories for the methods required to perform the analysis are needed. For example, if a mesh of a component is required using **SALOME** then a directory named :file:`Mesh` is required within the experiment directory which contains a relevant **SALOME** file.

A relevant directory would also need to be created within the input directory, i.e. :file:`Input/#ExpName`. The parameter file(s) which passes information to the relevant methods and files used are to be included within this directory.

Containers and Methods:
***********************
In **VirtualLab**, 'Containers' and 'Methods' are closely linked and are the heart of how **VirtualLab** can pull together many different pieces of software.
The **VirtualLab** executable actually starts out as a tcp (networking) sever running on the host machine defined by the script :file:`VL_server.py`. The server first spawns a manager container, **VL_Manager**, and passes in the RunFile. **VL_Manager** then executes the RunFile in a python environment. The RunFile itself begins by creating
an instance of the VLSetup class. This then acts to spawn, control and co-ordinate all the other containers that will run the software to perform the actual analysis.

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VL_Worflowpng.png?inline=false
  :width: 400
  :alt: Diagram of VirtualLab container setup
  :align: center

The Containers are spawned using various Methods which are functions assigned to the **VirtualLab** class to perform different types analysis, e.g. Mesh and Sim.
When a method is called by the **VL_Manager** a second container is spawned which is setup to perform the perform the required analysis. 

For example, a call to :bash:`VirtualLab.Mesh()` will spawn a container which has SalomeMeca installed inside. This will then run a script that will perform the actual analysis
using the parameters supplied by **VL_Manager**. The full list of different methods can be found in the methods directory :file:`Scripts/Methods`.

Each method file has a base class called 'Method' within it. These classes have a function called 'Setup' where information from the parameter file(s) are passed to build up the work to perform analysis, e.g., the information attached to the namespace 'Mesh' in the parameter file(s) is available in the Setup function of the method 'Mesh'. 

The 'Method' class must also have two other functions 'Spawn' and 'Run' which change how the method should behave when called, e.g., VirtualLab.Mesh().
The first function 'Spawn' is selected when the method is called by the **VL_Manager** container. This is handled automatically in the base method class.
'Spawn' as the name suggests configures a number of parameters and then communicates with the server on the host to spawn the container linked to the method 
and pass in the parameters for the analysis in question.

The second function 'Run' is selected when the method is called within a container other than **VL_Manager**, again this is handled transparently. 
'Run' is the function that will perform the required analysis with the supplied parameters.

Although not compulsory, these classes usually have a function called PoolRun which helps performs the analysis in parallel. For example, in the 'Mesh' method the meshes are created using **SALOME** in the PoolRun function. Placing the analysis in a seperate function enables the use of **VirtualLab**'s parallelisation package. This allows multiple pieces of analysis to be performed in parallel using either the pathos (single intra-node) or pyina (multi inter-node) packages. Please see one of the available methods to understand how this is achieved.

.. note::
    Any file in the methods directory starting with '_' will be ignored.

Amending Available Methods
**************************

Ammendments can be made to the methods available by using the :file:`config.py` file in the relevant methods directory. For example, due to the HIVE experiment being a multi-stage multi-physics experiment, 'Sim' needs to include a few additional steps. These are added in the file :file:`Scripts/Experiments/HIVE/Sim/config.py`. There is a similar config file for the meshing routine of HIVE also.

Adding New Methods:
*******************

To create a new method you will need a few things. Firstly, you will need a script to place in the methods directory. You may create a copy of the file :file:`_Template.py` in the methods directory and save it as #MethodName.py, where #MethodName the name of the new method type. Edit this file to perform the steps you desire. Not forgetting to edit the 'Spawn' function to associate your new
method with a new or existing container. #MethodName will then be available to add information to in the parameter file(s) and to perform analysis using VirtualLab.#MethodName() 
in the run file.

Next, you will need an apptainer Container configured with the appropriate software to run you analysis. This can either be one of our existing containers, found in the Containers directory, or a custom one you have created (see `adding new containers <contributing.html#adding-new-containers>`_). You will also need to create both a bash and python script to start the container and 
perform the analysis respectively. We have templates for both of these in the bin and bin/python directories.

Finally, you will need to add your method to the config file :file:`Config/VL_Modules.json`. Currently this only requires one parameter, a namespace to associate with 
your method. This is the name that is used in the the parameters file for **VirtualLab** and allows you to use a different name if you wish. 
For example Cad2vox uses the method 'Voxelise' but the namespace 'Vox' as it's easier to type. 

.. note:: 
   Each method can only have a single namespace, however, namespaces do not need to be unique to particular methods. 

Say for example you have several methods which share parameters they can share the same namespace. This is the case for CIL and GVXR where they share the 'GVXR' namespace since they share many of the same parameters.

Adding New Containers:
**********************

To build new containers for **VirtualLab** you will first need to `Install Docker <https://docs.docker.com/get-docker/>`_. We use Docker for development of containers as opposed to Apptainer because Dockerhub provides a convenient way of hosting and updating containers which Apptainer can pull from natively. The next step is to create your DockerFile configured with the software that you wish to use. We wont go into detail how to do this because it's out of the scope of this document. However, most popular software already have pre-made DockerFiles you can use as a starting point or failing that there are already plenty of resources online to get you started.

Once you have a DockerFile you will need to convert it to Apptainer. Annoyingly, Apptainer can't build directly from a Docker file instead you need to point it to a repository on a docker registry. 
The easiest way to do this is to use `DockerHub  <https://hub.docker.com/>`_. You will first need to create an account. Once this is done you will need to log into the DockerHub website then click 
on the blue "Create Repository" button (see screenshots). 

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/dockerhub_1.png
   :alt: insert screenshot of Dockerhub here.

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/dockerhub_2.png
   :alt: insert screenshot of Dockerhub here.

From there you will need to give your repository a name and decide if you want it to be public or private (Note: DockerHub only allows you have 1 private repository for free).

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/dockerhub_3.png
   :alt: insert screenshot of Dockerhub here.

Once this is complete you will need to push your docker image to the repository. this can be easily achieved at the command line.

First build your image locally, if you have not done so already. Replacing <image-name>, <tag-name> and <my_dockerfile> with whatever image name, tag and DockerFile you want to use.

:bash:`Docker build -t <image-name>:<tag-name> -f <my_dockerfile>`

Next login to DockerHub with the account you created.

:bash:`docker login`

Next we need to tag the image in a particular way to tell docker to point it to your repository. In this case <user-name> and <repo-name> are your username on DockerHub and the name of the repository
you wish to push to.

:bash:`docker tag <image-name>:<tag-name> <user-name>/<repo-name>:<tag-name>`

Finally we can push the image with

:bash:`docker push <user-name>/<repo-name>:<tag-name>`

With that done we can finally convert our Docker image to Apptainer with the following command. Replacing <MyContainer>.sif with whatever name you'd like to give the Apptainer sif file.

:bash:`apptainer build <My_container>.sif docker://<user-name>/<repo-name>:<tag-name>`

.. admonition:: Using a local Docker Repository

    Whilst DockerHub is free to use and a convenient solution it may not be the best solution for your situation. If privacy is your concern you could use an alternative registry like 
    `singularity hub <https://singularityhub.github.io/>`_ or even `host your own <https://www.c-sharpcorner.com/article/setup-and-host-your-own-private-docker-registry/>`_. 
    
    However, Say you are doing lots of testing and have a slow or limited internet connection. It's conceivable you may have to wait several minutes for upload your container to DockerHub only to re-download 
    it through Apptainer. Fortunately, it is entirely possible to host a Docker registry on your local machine. Unfortunately, there are a number of caveats to consider:

    1. It's quite fiddly and unintuitive to actually set up
    2. You are essentially doubling the amount of space needed to store docker images as you will have both a local and remote copy of the image to deal with.
    3. You won't be able to share these images with anyone else as they will be local to your machine.

    With those caveats in mind, if you are still undeterred a good set of instructions can be `found here <https://rcherara.ca/docker-registry/>`_.


Now that we have an apptainer file making it available as a module in **VirtualLab** is a fairly straightforward process. First place the sif file in the Containers directory of **VirtualLab**. You will then need to edit
the modules Config file :file:`Config/VL_Modules.json` to make the container available as **VirtualLab** module.

This file contains all the parameters to allow for the configuration of the various containers used by **VirtualLab**. The outer keys are the Module name used in the 'Spawn' method and the inner keys 
are the various parameters.

.. note:: 
    A single apptainer file can be associated to multiple Modules. This name is only used to identify how to setup the container 
    when 'Spawn is called by a particular method.  Thus you can use a single container for multiple different 
    methods that share the same software. Each method will simply need its own bash and pythons scripts to tell the 
    container what needs to be done.   

The following keys are required to define a module:

* Docker_url: The name of the image on DockerHub (that is "docker://<user-name>/<repo-name>" you used earlier)
* Tag: The image tag, again <tag-name> from earlier do not include the semi-colon
* Apptainer_file: Path to the sif file used for Apptainer 
* Startup_cmd: Command to run at container startup.

You also have the following optional keys:

* cmd_args: custom command line arguments, only useful if using your own scripts to start the container.

.. admonition:: Using custom startup scripts and custom_args

    The default arguments used by the template script are:'-m param_master -v param_var -s Simulation -p Project -I container_id'. 
    If cmd_args is set it will override these. You can also set it to a empty string (i.e. "") to specify no arguments.  

An optional final step you can take is to link you Container to the official ibsim repo on DockerHub. We keep all our DockerFiles in a separate 
`git repoisitory <https://github.com/IBSim/VirtualLab://github.com/IBSim/VirtualLab>`_ this is linked to DockerHub such that all we have to do is push our updated DockerFiles to that repo and it will
automatically update and re-build the container on DockerHub. If you wish to access this please contact Llion Evans.

Contributing to VirtualLab
**************************

To submit changes to **VirtualLab** we use the following procedure. This allows us to collaborate effectively without treading on each others toes.

Branch Structure
################
The current setup for **VirtualLab** is as follows:
 1. **Main:** Public facing branch, only changes made to this are direct merges from the dev branch.
 2. **Dev:** Main branch for the development team to pull and work from. We do not work directly on this branch, the only changes to this are direct merges from temporary branches.
 3. **Temporary branches:** Branches for new or work in progress features and bug fixes.

 Each developer should create a branch from **dev** when they want to create a new feature or bug fix.
 The branch name can be anything you like although preferably it should be descriptive of what the branch is for. Branch names should also be prepended with the developer's initials (to show who's leading the effort). Once the work is complete These branches can be merged back into **dev** with a merge request and then deleted.

Creating a new branch should be done roughly as follows::

    # First ensure you are on the dev branch
    git checkout dev
    # Create a new branch with a name and your initails
    git branch INITIALS_BRANCH-NAME
    # change onto the newly created branch
    git checkout BRANCHNAME-INT
    git push --set-upstream origin INITIALS_BRANCH-NAME

Now that we have a new temporary branch development can continue on this branch as usual with commits happening when desired by the user. The temp branch can be also pushed to GitLab without creating a merge request if working with collaborators (and also for backing up work in the cloud). To do this the collaborator just needs to ensure they have all the latest changes from all the branches of the code from GitLab using ``git pull --all`` then change over to your branch using ``git checkout INITIALS_BRANCH-NAME``.

Creating a merge request
########################

Once work on the temporary branch is complete and and ready to be merged into the dev branch we need to first ensure we have pushed our changes over to the remote GitLab repo.::

    # first ensure we have the latest changes
    git pull
    # push our changes to the GitLab repo
    git push

once this is complete we can go to the **VirtualLab** repo on `gitlab.com <https://gitlab.com/ibsim/virtuallab>`_ and ensure we are loged into GitLab.

To create the request, from the left hand side of the page click on "merge requests".

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GitLab.png
   :alt: insert screenshot of GitLab here.

Then on the right hand side of the next page click "create merge request".

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GitLab2.png
   :alt: insert screenshot of GitLab here.

From here set the source branch as your temporary branch and the taget branch as dev then click compare branches and continue.

.. image:: https://gitlab.com/ibsim/media/-/raw/master/images/docs/screenshots/GitLab3.png
   :alt: insert screenshot of GitLab here.

The final step is to use the form to create the merge request:

* First give your merge request a title and a brief description of what features you have added or what changes have been made.
* For **Assignees** select "Assign to me".
* For **Reviewers** select one of either Ben Thorpe, Llion Evans or Rhydian Lewis.
* For **milestone** select no Milestone.
* For **Labels** select one if appropriate.
* For **Merge options** select "Delete source branch when merge request is accepted".

Once this is complete click "create merge request" this will then notify whoever you selected as reviewer to approve the merge.

Tidying up
##########

Once the merge has been accepted, The final step is to pull in the latest changes to dev and delete your local copy of the temporary branch ::

    # first ensure we have the latest changes
    git checkout dev
    git pull
    # delete our local copy of the temporary branch
    git branch -d INITIALS_BRANCH-NAME
