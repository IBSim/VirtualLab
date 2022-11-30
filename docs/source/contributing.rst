Adding to VirtualLab
====================

VirtualLab has been designed so that adding work is as easy as possible. There are 4 ways in which new work can be added:
 1. New analysis scripts,
 2. New methods of performing analysis,
 3. New virtual experiments,
 4. Containers with new software or code.

Description on how work can be added to these are discussed below, along with the best practice for adding your work on the VirtualLab repository.

Scripts
*******

The easiest way to add to VirtualLab is by creating scripts for experiments and methods that already exist. For example, to create a new mesh script 'NewComponent' for the tensile experiment one would need to create the file :file:`Scripts/Experiments/Tensile/Mesh/NewComponent.py` which describes the steps that **SALOME** must follow to create a CAD geometry and mesh. This file can then be used by specifying it as the 'file' attribute to the 'Mesh' namespace, e.g. Mesh.File = 'NewComponent' in the parameter file.

Similarly Sim and DA scipts can be created and placed in the relevant directories in the experiment directory.

Methods
*******

Methods are the functions assigned to the VirtualLab class to perform different types analysis, e.g. Mesh and Sim. The full list of different methods can be found in the methods directory :file:`Scripts/Methods`.

Each method file has a class called 'Method' within it. These classes have a function called 'Setup' where information from the parameter file(s) are passed to build up the work to perform analysis, e.g. the information attached to the namespace 'Mesh' in the parameter file(s) is available in the Setup function of the method 'Mesh'. The 'Method' class must also have a function called 'Run' which is what's called in the Run file, e.g VirtualLab.Mesh() in the run file will execute the function 'Run' from method 'Mesh'.

Although not compulsory, these classes usually have a function called PoolRun which performs the analysis. For example, in the 'Mesh' method the meshes are created using **SALOME** in the PoolRun function. Placing the analysis in a seperate function enables the use of VirtualLab's parallelisation package. This allows multiple pieces of analysis to be performed in analysis using either the pathos (single node) or pyina (multi-node) packages. Please see one of the available methods to understand how this is achieved.

To create a new method create a copy of the file :file:`_Template.py` in the methods directory and save it as #MethodName.py, where #MethodName the name of the new method type. Edit this file to perform the steps you desire. #MethodName will now be available to add information to in the parameter file(s) and to perform analysis using VirtualLab.#MethodName() in the run file.

.. note::
    Any file in the methods directory starting with '_' will be ignored.

Ammendments can be made to the methods available by using the :file:`config.py` file in the relevant methods directory. For example, as the HIVE experiment is a multi-physics experiment 'Sim' needs to include a few additional steps. These are added in the file :file: `Scripts/Experiments/HIVE/Sim/config.py`. There is a similar config file for the meshing routine of HIVE also.

Experiments
***********

Adding a new experiment to **VirtualLab** will require creating a new experiment directory :file:`Scripts/Experiments/#ExpName`, where #ExpName is the name of the experiment. Within this experiment directory sub-directories for the methods required to perform the analysis are needed. For example, if a mesh of a component is required using **SALOME** then a directory names :file:`Mesh` is required within the experiment directory which contains a relevant **SALOME** file.

A relevant directory would also need to be created within the input directory, i.e. :file:`Input/#ExpName`. Within this directory the parameter file(s) are included which passes information to the relevant methods and files used.

Containers
**********
#ToDo

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
* For **Reviewers** select one of either Ben, Llion or Rhydian.
* For **milestone** select no Milestone.
* For **Labels** select one if appropriate.
* For **Merge options** select "Delete source branch when merge request is accepted".

Once this is complete click "create merge request" this will then notify whoever you selected as reviewer to aprove the merge.

Tidying up
##########

Once the merge has been accepted, The final step is to pull in the latest changes to dev and delete your local copy of the temporary branch ::

    # first ensure we have the latest changes
    git checkout dev
    git pull
    # delete our local copy of the temporary branch
    git branch -d INITIALS_BRANCH-NAME
