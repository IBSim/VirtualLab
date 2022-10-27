#!/usr/bin/env python
# Script to install/update VirtualLab  
#=======================================================================

# Import the modules needed to run the script.
import sys, os
from pathlib import Path
import subprocess
from zipfile import ZipFile

# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu (execution starts here)
def main_menu():
    os.system('clear')
    
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
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            menu_actions['main_menu']()
    return

# Menu 1
def Install_menu():
    print("Installing VirtualLab !\n")
    print("Please select where to Install VirtualLab to.\n")
    install_path = str(Path.home())
    print(f"1. Default location: {install_path}")
    print("2. Custom location")
    print("9. Back")
    print("0. Quit")
    choice = input(" >>  ")
    if choice == '1':
        install_Vlab(install_path)
    elif choice == '2':
        custom_dir()
    if choice == '':
        # return to install menu
        menu_actions['1']()
    else:
        exec_menu(choice)
    return


# Back to main menu
def back():
    menu_actions['main_menu']()

# Exit program
def exit():
    sys.exit()
    

def custom_dir():
    print("Please enter full path for where Install VirtualLab to.\n")
    print("Or type 0 to go back.\n")
    choice = input(" >>  ")
    if choice == '0':
        # return to install menu
        menu_actions['1']()
    elif not os.path.isabs(choice):
        # choice is not a valid path
        print("Invalid path, please try again.\n")
        custom_dir()
    
    if os.path.isdir(choice):
        print(f'installing VirtualLab to {choice}')
        install_Vlab(choice)
    else:
        # path is valid but does not exist so create it.
        print(f'Installing VirtualLab to {choice}')
        print(f'creating new directory')
        os.makedirs(choice)
        install_Vlab(choice)

def install_Vlab(install_path):
    os.chdir(install_path)
    Docker = check_container_tool()
    print('Downloading VirtuaLab')
    get_latest_code()
    if Docker:
        get_latest_docker()
    else:
        get_latest_Apptainer(install_path)
    print("Instalation Complete!!")
    sys.exit()

def get_latest_code():
    if Platform=='Windows':
        # use wget with powershell for windows
        subprocess.call('powershell.exe wget https://gitlab.com/ibsim/virtuallab/-/archive/master/virtuallab-master.zip', shell=True)
    else:
        # Mac/Linux hopefully have Curl
        subprocess.call('curl -O https://gitlab.com/ibsim/virtuallab/-/archive/master/virtuallab-master.zip', shell=True)
    with ZipFile('virtuallab-master.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
        zipObj.extractall()
    os.rename('virtuallab-master','VirtualLab')

def get_latest_docker():
    print("Pulling latest VLManager container from Dockerhub:\n")
    try:
        subprocess.call('docker pull ibsim/virtuallab_main:Dev',shell=True)
    except:
        print('docker pull failed. please check docker is installed and working corectly.')
        sys.exit()

def get_latest_Apptainer(install_path):
    print("Pulling latest VLManager container from Dockerhub and converting to Apptainer:\n")
    try:
        subprocess.call(f'Apptainer build -F {install_path}/Containers/VirtualLab.sif docker://ibsim/virtuallab_main:Dev',shell=True)
    except:
        print('build failed. please check Apptainer is installed and working corectly.')
        sys.exit()
    print('******************************************************************************\n')
    print(' Note: Unfortunately unlike Docker apptainer does not currently have a built in\n')
    print(' tool to check for container updates. If you would like to ensure that you are\n')
    print(' using the latest version of a particular module. You will need delete the\n')
    print(' already downloaded .sif files for the module you wish to update. These can be\n')
    print(f' found in {install_path}/Containers. You can then re-run VirtualLab which will\n')
    print(' automatically download the newest version.\n')
    print('******************************************************************************\n')

def update_vlab():
    vlab_dir = f'{Path.home()}/VirtualLab'
    if os.path.isdir(vlab_dir):
        print(f'Found VirtualLab install in {vlab_dir}')
    else:
        print("Installer could not automatically find VirtualLab\n")
        print('Please enter path to VirtualLab directory.\n')
        print('Or press 0 to exit')
        vlab_dir =  input('>>')
        if vlab_dir=='0':
            exit()
        elif not os.path.isdir(vlab_dir) or not os.path.isabs(vlab_dir):
            print("that directory does not exist, please try again.")
            print("************************************************")
            update_vlab()
        # this check prevents you from nuking a random directory that happens to exist.
        elif not os.path.exists(f'{vlab_dir}/bin/VirtualLab'):
            print(f"I couldn't find the VirtualLab executable in {vlab_dir}/bin")
            print('are you sure that is the correct path?')
            print("************************************************************")
            update_vlab()
    # reame old dir to avoid acidental deletion
    os.rename(vlab_dir,f'{vlab_dir}-old')
    get_latest_code()
    os.rename('VirtualLab',vlab_dir)
    Docker = check_container_tool()
    if Docker:
        get_latest_docker()
    else:
        get_latest_Apptainer(vlab_dir)
    print("Update Complete!!")
    print(f"Note: your previous install has been saved to {vlab_dir}-old for safekeeping. \n you can delete this if you wish.")
    sys.exit()

def check_platform():
    '''Simple function to return 'Linux', 'Windows' or 'Darwin' (Mac)
    to allow us to adapt commands for each Os'''
    import platform
    if platform.system() in ['Linux','Darwin','Windows']:
        return platform.system()
    else:
        print("Installer Cannot autmatically identify what OS you are running\n")
        print("Instalation failed\n")
        sys.exit()

def check_container_tool():
    if Platform == 'Linux':
        print('Are you using Docker or Apptainer?')
        print('1: Docker')
        print('2: Apptainer')
        print('0: Back')
        choice = input(" >>  ")
        if choice == '0':
            # return to install menu
            menu_actions['1']()
        elif choice == '1':
            Docker = True
            return Docker
        elif choice == '2':
            Docker = False
            return Docker
        else:
            print('Invalid choice try again')
            check_container_tool()
    else:
        #Windows/Mac only use Docker so no need to ask just return True.
        return True
# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': Install_menu,
    '2': update_vlab,
    '9': back,
    '0': exit,
}



# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # Launch main menu
    Apptainer=False
    Platform = check_platform()
    main_menu()