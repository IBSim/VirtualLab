Adding Code
===========
To submit changes to **VirtualLab** we use the following procedure. This allows us to collaborate effectively without treading on each others toes.

Branch Structure
****************
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

Creating a merge request into the dev branch
********************************************

Once work on the temporary branch is complete and and ready to be merged into dev we need to first ensure we have pushed our changes over to the remote GitLab repo.::

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
**********

Once the merge has been accepted, The final step is to pull in the latest changes to dev and delete your local copy of the temporary branch ::
  
    # first ensure we have the latest changes
    git checkout dev
    git pull
    # delete our local copy of the temporary branch
    git branch -d INITIALS_BRANCH-NAME
