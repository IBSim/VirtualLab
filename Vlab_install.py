#!/usr/bin/env python
# Script to install/update VirtualLab
# =======================================================================

# Import the modules needed to run the script.
import sys, os
from pathlib import Path
import subprocess
from zipfile import ZipFile
import shutil

# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu (execution starts here)
def main_menu():
    os.system("clear")

    print("Welcome to the VirtualLab Installer,\n")
    print("Please choose an option:")
    print("1. Install VirtualLab")
    print("2. Update existing VirtualLab Install")
    print("\n0. Quit")
    choice = input(" >>  ")
    exec_menu(choice)

    return


# Execute menu
def exec_menu(choice):
    os.system("clear")
    ch = choice.lower()
    if ch == "":
        menu_actions["main_menu"]()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            menu_actions["main_menu"]()
    return


# Menu 1
def Install_menu():
    print("Installing VirtualLab !\n")
    print("Please select where to Install VirtualLab to.\n")
    if Platform == "Windows":
        install_path = "C:/Program Files/VirtualLab"
    else:
        install_path = f"{Path.home()}/VirtualLab"
    print(f"1. Default location: {install_path}")
    print("2. Custom location")
    print("9. Back")
    print("0. Quit")
    choice = input(" >>  ")
    if choice == "1":
        install_Vlab(install_path)
    elif choice == "2":
        custom_dir()
    if choice == "":
        # return to install menu
        menu_actions["1"]()
    else:
        exec_menu(choice)
    return


# Back to main menu
def back():
    menu_actions["main_menu"]()


# Exit program
def exit():
    sys.exit()


def yes_no():
    print("Are you sure you want to Continue? y/n")
    choice = input(">>")
    if choice.lower() != "y":
        print("Quiting")
        sys.exit()
    return


def custom_dir():
    print("Please enter full path for where to Install VirtualLab.\n")
    print("Or type 0 to go back.\n")
    choice = input(" >>  ")
    if choice == "0":
        # return to install menu
        menu_actions["1"]()
    elif not os.path.isabs(choice):
        # choice is not a valid path
        print("Invalid path, please try again.\n")
        custom_dir()
    install_Vlab(choice)


def install_Vlab(install_path,non_interactive=False,shell_num=1):
    os.system("clear")
    print(f"Installing VirtualLab to {install_path}")
    if os.path.isdir(install_path):
        # if it exists check if it is empty
        dir_list = os.listdir(install_path)
        # Checking if the list is empty or not
        if len(dir_list) != 0:
            print("******************************************\n")
            print(f"The Directory {install_path} is not Empty\n")
            print("Warning: contents will be overwriten.\n")
            print("******************************************\n")
            shutil.rmtree(install_path)
            os.mkdir(install_path)
        if not non_interactive:
            yes_no()
    else:
        print(f"New directory {install_path} will be created")
        os.mkdir(install_path)
        if not non_interactive:
            yes_no()
    # Docker = check_container_tool()
    Docker = False
    print("Downloading VirtualLab\n")
    get_latest_code(install_path)
    if Docker:
        get_latest_docker()
    else:
        get_latest_Apptainer(install_path)
    add_to_Path(install_path, non_interactive,shell_num)
    print("Instalation Complete!!")
    sys.exit()


def get_latest_code(install_path):
    # os.chdir(install_path)
    if Platform == "Windows":
        # use wget with powershell for windows
        subprocess.call(
            f"powershell.exe wget https://gitlab.com/ibsim/virtuallab/-/archive/dev/virtuallab-dev.zip -p {install_path}",
            shell=True,
        )
    else:
        # Mac/Linux hopefully have Curl
        # subprocess.call(f'curl --output {install_path}/VirtualLab.zip -O https://gitlab.com/ibsim/virtuallab/-/archive/dev/virtuallab-dev.zip', shell=True)
        import git

        git.Repo.clone_from(
            "https://gitlab.com/ibsim/virtuallab.git", f"{install_path}"
        )
        my_repo = git.Repo(f"{install_path}")
        my_repo.git.checkout("dev")
        # get binaries from second repo and copy them across
        git.Repo.clone_from(
            "https://gitlab.com/ibsim/virtuallab_bin.git", f"{install_path}/bins"
        )
        my_repo2 = git.Repo(f"{install_path}/bins")
        my_repo2.git.checkout("dev")
        shutil.copytree(
            f"{install_path}/bins", f"{install_path}/bin", dirs_exist_ok=True
        )
        shutil.rmtree(f"{install_path}/bins")


def get_latest_docker():
    print("Pulling latest VLManager container from Dockerhub:\n")
    try:
        subprocess.run("docker pull ibsim/virtuallab_main:Dev", shell=True, check=True)
    except:
        print(
            "docker pull failed. please check docker is installed and working corectly."
        )
        sys.exit()


def get_latest_Apptainer(install_path):
    print(
        "Pulling latest VLManager container from Dockerhub and converting to Apptainer:\n"
    )

    subprocess.run(
        f"apptainer build -F {install_path}/Containers/VL_Manager.sif docker://ibsim/virtuallab:latest",
        shell=True,
        check=True,
    )


def update_vlab(non_interactive=False):
    vlab_dir = os.environ.get("VL_DIR", None)
    if vlab_dir == None:
        vlab_dir = f"{Path.home()}/VirtualLab"

    if not os.path.isdir(vlab_dir):
        print("Installer could not automatically find VirtualLab\n")
        print("Please enter path to VirtualLab directory.\n")
        print("Or press 0 to exit")
        vlab_dir = input(">>")
        if vlab_dir == "0":
            exit()
        elif not os.path.isdir(vlab_dir) or not os.path.isabs(vlab_dir):
            print("that directory does not exist, please try again.")
            print("************************************************")
            update_vlab()
        # this check prevents you from nuking a random directory that happens to exist.
        elif not os.path.exists(f"{vlab_dir}/bin/VirtualLab"):
            print(f"I couldn't find the VirtualLab executable in {vlab_dir}/bin")
            print("are you sure that is the correct path?")
            print("************************************************************")
            update_vlab()

    print(f"Found VirtualLab install in {vlab_dir}")
    if not non_interactive:
        yes_no()
    # reame old dir to avoid deletion
    os.rename(vlab_dir, f"{vlab_dir}-old")
    os.mkdir(vlab_dir)
    get_latest_code(vlab_dir)
    # Docker = check_container_tool()
    Docker = False
    if Docker:
        get_latest_docker()
    else:
        get_latest_Apptainer(vlab_dir)
    print("Update Complete!!")
    print(
        f"Note: your previous install has been saved to {vlab_dir}-old for safekeeping. \n you can delete this if you wish."
    )
    sys.exit()


def check_platform():
    """Simple function to return 'Linux', 'Windows' or 'Darwin' (Mac)
    to allow us to adapt commands for each Os"""
    import platform

    if platform.system() in ["Linux", "Darwin", "Windows"]:
        return platform.system()
    else:
        print("Installer Cannot autmatically identify what OS you are running\n")
        print("Instalation failed\n")
        sys.exit()


def check_container_tool():
    if Platform == "Linux":
        print("Are you using Docker or Apptainer?")
        print("1: Docker")
        print("2: Apptainer")
        print("0: Back")
        choice = input(" >>  ")
        if choice == "0":
            # return to install menu
            menu_actions["1"]()
        elif choice == "1":
            Docker = True
            return Docker
        elif choice == "2":
            Docker = False
            return Docker
        else:
            print("Invalid choice try again")
            check_container_tool()
    else:
        # Windows/Mac only use Docker so no need to ask just return True.
        return True


def add_to_Path(install_dir,non_interactive,shell_num):
    os.system("clear")
    os.chdir(install_dir)
    if Platform == "Linux":
        if non_interactive:
            choice = shell_num
        else:
            print("What Shell are you using?")
            print("If you don't know just stick with the default (i.e. Bash).")
            print("1: Bash (default)")
            print("2: Zsh")
            print("3: Other")
            choice = input(" >>  ")
        if choice == 2:
            output = subprocess.run(
                [
                    f"{install_dir}/Scripts/Install/Set_VLProfile_zsh.sh",
                    f"{install_dir}",
                    f"{Path.home()}",
                ],
                capture_output=True,
            )
            print(output.stdout.decode("utf8"))
        if choice == 3:
            print(
                "****************************************************************************"
            )
            print(
                " Auto setting of path variables is only offically supported with bash and zsh.\n"
            )
            print(
                " For other shells you will need to do this manualy (this is tedious I know).\n"
            )
            print(" Before launching VirtualLab you will need to run:\n")
            print(f" export VL_DIR={install_dir}")
            print(f" then add {install_dir}/bin to your system path.")
            print(
                " Note: you may want to automate this on login with whatever method your\n"
            )
            print(" shell to handles such things.")
            print(
                "****************************************************************************"
            )
        else:
            # default to bash
            output = subprocess.run(
                [
                    f"{install_dir}/Scripts/Install/Set_VLProfile_bash.sh",
                    f"{install_dir}",
                    f"{Path.home()}",
                ],
                capture_output=True,
            )
            print(output.stdout.decode("utf8"))
        subprocess.check_call(["chmod", "+x", f"{install_dir}/bin/VirtualLab"])
    elif Platform == "Darwin":
        # MacOs uses zsh by default
        output = subprocess.run(
            [
                f"{install_dir}/Scripts/Install/Set_VLProfile_zsh.sh",
                f"{install_dir}",
                f"{Path.home()}",
            ],
            capture_output=True,
        )
        print(output.stdout.decode("utf8"))
    else:
        # for windows VirtualLab expects to be instaled in 'C:/Program Files/VirtualLab' this should be easy to automate I just havent got the time right now.
        if install_dir != "C:/Program Files/VirtualLab":
            print(
                "****************************************************************************"
            )
            print(
                " You appear to have set VirtualLab to be installed in a non standard location.\n"
            )
            print(" i.e. NOT in C:/Program Files/VirtualLab'\n\n")
            print(
                " Auto setting of path varaibles for Custom directories is currently only\n"
            )
            print(" supported with Linux and Mac.\n")
            print(
                " For Windows you will need to do this manualy (this is tedious I know).\n"
            )
            print(
                " You will find instructions for doing this in the install guide in the Docs."
            )
            print(" This step will hopefully be automated in a later version.\n")
            print(
                "****************************************************************************"
            )
# add vlab_dir to VLconfig.py
    import re
    with open(f'{install_dir}/VLconfig.py', 'r+') as f:
        file = f.read()
        file = re.sub('VL_HOST_DIR=""',f'VL_HOST_DIR="{install_dir}"',file)
        f.seek(0)
        f.write(file)
        f.truncate()
    return


# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    "main_menu": main_menu,
    "1": Install_menu,
    "2": update_vlab,
    "9": back,
    "0": exit,
}


# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    import argparse
    Apptainer = False
    Platform = check_platform()
    parser = argparse.ArgumentParser()
    shell_types = {"bash":1,"zsh":2, "other":3}
    parser.add_argument(
        "-d",
        "--inst_dir",
        help="Path to custom Directory in which to install VirtuaLab when using -y.",
        default=None,
    )
    parser.add_argument(
        "-y",
        "--yes",
        help="Run installer non-interactivley. By default this installs in /home/$USER/VirtualLab and uses bash as the default shell unless options -d or -S are used. ",
        action="store_true",
    )
    parser.add_argument(
        "-U",
        "--update",
        help="Run update non-interactivley.",
        action="store_true",
    )
    parser.add_argument(
        "-S",
        "--shell",
        help="Specify which shell to use for non-interactive install. Supported shells are: zsh, bash and other",
        default = "bash",
    )

    args = parser.parse_args()
    if args.update and args.yes:
        raise ValueError("You cannot use options -y and -U together as that makes no sense.")
    
    if args.inst_dir != None:
        install_path = args.inst_dir
    else:
        if Platform == "Windows":
            install_path = "C:/Program Files/VirtualLab"
        else:
            install_path = f"{Path.home()}/VirtualLab"

    shell_num = shell_types.get(args.shell,None)
    if shell_num == None:
        raise ValueError(f"Option {args.shell} is not a supported shell must be one of {list(shell_types.keys())}.")

    if args.yes:
        install_Vlab(install_path,non_interactive=True,shell_num=shell_num)
    elif args.update:
        update_vlab(non_interactive=True)
    else:
    # Launch main menu
        main_menu()
