#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:26:10 2020

@author: jiedz

"""
#%%
import os
import sys
import time as tm
import numpy as np
import paramiko
#%%
#General data acquisition parameters
ADC_CLK_PERIOD = 8*1e-9 #RedPitaya ADC clock period [s]
N_SAMPLES_MAX = 2**23
#General remote connection parameters
ssh_username = "root"
ssh_password = "root"
#Data acquisition commands
RP_project_path = "/home/RedPitaya/red-pitaya-notes/ADC-RAM-streaming-Iyad/External-trigger"
command_enter_project_directory = "cd "+RP_project_path
command_load_FPGA_bistream = "cat FPGA-bitstreams/fpga_ADC-RAM-streaming_8MSamples.bit > /dev/xdevcfg"
command_compile_acquisition_script = "make C-code/adc-recorder-trigger"
command_enter_script_directory = "cd C-code"  
command_start_acquisition_script = "./adc-recorder-trigger_save-binary "
terminal_oparamikoutput_text_separator = "\n-----------------------"

class Timer:
    """
    This class defines a simple timer.
    """
    def __init__(self):
        self.start = tm.time()
 
    def start(self):
        self.start = tm.time()
 
    def log(self):
        logger = tm.time() - self.start
        print('Time log -',logger)
    
    def currentTime(self):
        return tm.time() - self.start
        
    def reset(self):
        self.__init__()
        
#%%
class acquisitionADCRAM:
    """
    This class defines a custom RedPitaya acquisition module, based on Pavel Admin's
    ADC-to-RAM streaming FPGA code developed for RedPitayas. The code in question
    streams a number of samples larger than the default buffer size (i.e., 2^14 Sa = 16384 Sa) from the
    fast analog-to-digital converters (ADC) to the processor RAM of the RedPitaya. 
    A custom C script running on the target RedPitaya, can be executed from this class via SSH (Secure SHell) to start a data acquisition. 
    The acquired data are saved on a binary file on the target RedPitaya, and can be transferred via SFTP 
    (Secure File Transfer Protocol) to a target location on the local host (this PC).    
    The maximum number of samples that it is possible to acquire at a time is currently 2^23 = 8388608.
    Multiple RedPitayas can be registered into this acquisition module.
    Each RedPitaya is defined as a sorted dictionary, structured in the following way:
        - ["name"]
        - ["remote_connection_keys"]
            - ["hostname", "ssh_client", "sftp","shell", "connected"]
        - ["acquisition_keys"]
            - ["n_samples", "decimation", Ts", time", "data_channel_1", "data_channel_2", "save_path"]
    where:
        - name: RedPitaya name - string
        - hostname: RedPitaya hostname, of the form 'rp-xxxxxx' - string;
        - ssh_client: SSH client - paramiko.SSHClient() return object
        - sftp: SFTP (Secure File Transfer Protocol) - paramiko.SSHClient.open_sftp() return object
        - shell: virtual shell (command line) - paramiko.SSHClient().invoke_shell() return object
        - connected: True if SSH and SFTP connections are open - boolean
        - n_samples: number of samples of an acquisition - int
        - decimation: decimation factor with respect to the nominal sampling rate (1/ADC_CLOCK_PERIOD) [adimensional] - float
        - Ts: RedPitaya sampling period [s] - float
        - time: time vector [s] for the j-th RedPitaya - array-like of float, of length np.size(n_samples[j-1]), j = 0, 1;
        - data_channel_1: channel 1 data [V] for the j-th RedPitaya - array-like of float, of length np.size(n_samples[j-1]), j = 1, 2;
        - data_channel_2: channel 2 data [V] for the j-th RedPitaya - array-like of float, of length np.size(n_samples[j-1]), j = 1, 2;   
    """
    def __init__(self, names, hostnames, n_samples = None, connect=False):
        names = np.atleast_1d(names)
        hostnames = np.atleast_1d(hostnames)
        
        print("\nCreating RedPitaya acquisition object.")
        self.names = names #list of names of the RedPitayas to be registered
        self.hostnames = hostnames #list of RedPitaya hostnames corresponding to their names
        #General features of a RedPitaya
        self.RedPitaya_keys = ["name", "remote_connection", "acquisition"] 
        #Remote connection features of a RedPitaya
        self.remote_connection_keys = ["hostname", "ssh_client", "sftp","shell", "connected"]
        #Data acquisition features of a RedPitaya
        self.acquisition_keys = ["n_samples", "decimation", "Ts", "time", "data_channel_1", "data_channel_2", "save_path"]
        #Check for errors in the input
        #-------------------------------------------------
        #Empty names list
        if np.size(names)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of RedPitaya names cannot be empty.")
            sys.exit(1)
        #Empty hostnames list
        if np.size(hostnames)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of RedPitaya hostnames cannot be empty.")
            sys.exit(1)
        #Empty number of samples list
        if np.size(n_samples)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of number of acquisition samples cannot be an empty one. If not needed, set to None.")
            sys.exit(1)        
        #Numbers of elements in 'names', 'hostnames' and 'n_samples' do not match 
        if (np.size(names) != np.size(hostnames)) or (np.size(names) != np.size(n_samples)) or (np.size(hostnames) != np.size(n_samples)):
            print("\nERROR acquisitionADCRAM.__init__(): the np.sizegth on non-empty lists 'names', 'hostnames' and 'n_samples' must be the same.")
            sys.exit(1) 
        #----------------------------------------------
        #If not specified, se a default number of samples of each RedPitaya
        if n_samples is None:
                n_samples = 1e5*np.ones((np.size(names), ))
        n_samples = np.atleast_1d(n_samples)
        #-----------------------------------------------
        #Register all the specified RedPitayas
        #----------------------------------------------
        self.RedPitayas = [None for j in range(np.size(names))]

        #Build the database of RedPitayas
        for j in range(np.size(names)):
            name = names[j] #RedPitaya name
            hostname = self.hostnames[j] #RedPitaya hostname
            n_samples_temp = int(n_samples[j]) #number of samples to be acquired by the RedPitaya
            RedPitaya = [] #Current RedPitaya
            ssh_client = paramiko.SSHClient() #RedPitaya SSH client (disconnected)
            sftp = None #no SFTP connection is made while the SSH client is disconnected
            shell = None  #no shell is invoked while the SSH client is disconnected
            connected = False
            #Set the acquisition properties               
            if n_samples_temp > N_SAMPLES_MAX:
                print("\nWARNING: Number of acquisition samples of RedPitaya '"+name+\
                      "' (hostname '"+hostname+"') is set to"+str(n_samples_temp)+\
                      ", but the maximum possible is"+str(N_SAMPLES_MAX)+"."+\
                 "\nSetting the number of samples to "+str(N_SAMPLES_MAX)+".")
                n_samples_temp = N_SAMPLES_MAX
            decimation = 1 #decimation factor with respect to the ADC sampling rate [adimensional]
            Ts = ADC_CLK_PERIOD*decimation #sampling rate [s]
            time = np.linspace(start=0, stop=(n_samples_temp-1)*Ts, num = n_samples_temp) #acquisition time vector [s]
            data_channel_1 = np.zeros((n_samples_temp, )) #channel 1 acquisition data [V]
            data_channel_2 = np.zeros((n_samples_temp, )) #channel 2 acquisition data [V]  
            save_path = os.getcwd() #data save path can be set with a dedicated function
            #Add the properties to the relative datasets
            #Remote connection
            remote_connection = [hostname, ssh_client, sftp, shell, connected]
            remote_connection = dict(zip(self.remote_connection_keys, remote_connection))
            #Acquisition
            acquisition = [n_samples_temp, decimation, Ts, time, data_channel_1, data_channel_2, save_path]
            acquisition = dict(zip(self.acquisition_keys, acquisition))
            #RedPitaya
            RedPitaya = [name, remote_connection, acquisition]
            RedPitaya = dict(zip(self.RedPitaya_keys, RedPitaya))
            #Add the RedPitaya to the list
            self.RedPitayas[j] = RedPitaya
        
        #Pack all RedPitayas in a single dictionary
        self.RedPitayas = dict(zip(self.names, self.RedPitayas))
        #---------------------------------------------
        #Set up remote connections if required
        if connect:
            self.connectRemote(names)
        
        
    
    def connectRemote(self, names):
        """
        This function establishes an SSH (Secure Shell) and an SFTP (Secure File Transfer Protocol)
        with the RedPitayas that are registered with names 'names'
        
        INPUTS:
            - names: names of the RedPitayas to be connected - array-like of string
        OUTPUTS:
            - None
        """
        #Check for errors in the input
        #------------------------------------------------
        if (np.size(names)==0) or (names is None):
            print("\nERROR acquisitionADCRAM.connectRemote(): 'names' must be a non-empty array of string.")
            return
        #-----------------------------------------------
        names = np.atleast_1d(names)
        if names[0] == "all":
            names = self.names
        #Open connection to all specified RedPitayas
        for j in range(len(names)):
            name = names[j]
            #If the name is not in the list, print a warning and skip the iteration
            if not(name in self.names):
                print("\nWARNING acquisitionADCRAM.connectRemote(): RedPitaya with name '"+name+"' is not in the list.")
                continue
            RedPitaya = self.RedPitayas[name]
            remote_connection = RedPitaya["remote_connection"]
            hostname = remote_connection["hostname"]
            connected = remote_connection["connected"]
            #If the RedPitaya is already connected, print a warning and skip the iteration
            if connected:
                print("\nWARNING acquisitionADCRAM.connectRemote(): RedPitaya '"+name+"', with hostname "+hostname+" is already connected.") 
                continue 
            ssh_client = remote_connection["ssh_client"]
            sftp = remote_connection["sftp"]
            #Open the ssh connection with the hostname
            #Load host keys
            ssh_client.load_system_host_keys()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            print("\nSet up an SSH connection with RedPitaya '"+name+"', with hostname "+hostname+".")
            try:
                ssh_client.connect(hostname=hostname+".local", \
                                       username=ssh_username, password=ssh_password)
            except:
                print("\nERROR acquisitionADCRAM.connectRemote(): SSH connection failed")
                print("Error:", sys.exc_info()[0])
                raise
                return
            #SFTP
            print("\nSet up an SFTP connection with RedPitaya '"+name+"', with hostname "+hostname+".")
            try:
                sftp = ssh_client.open_sftp()
            except:
                print("\nERROR acquisitionADCRAM.connectRemote(): SFTP connection failed")
                ssh_client.close()
                return   
            connected = True
            #Add the updated parameters to the datasets
            remote_connection["ssh_client"] = ssh_client
            remote_connection["sftp"] = sftp          
            remote_connection["connected"] = connected
            RedPitaya["remote_connection"] = remote_connection
            self.RedPitayas["name"] = RedPitaya
            
    def disconnectRemote(self, names):
        """
        This function closes an existing SSH (Secure Shell) and SFTP (Secure File Transfer Protocol)
        with the RedPitayas that are registered with names 'names'
        
        INPUTS:
            - names: names of the RedPitayas to be disconnected - array-like of string
        OUTPUTS:
            - None
        """
        #Check for errors in the input
        #------------------------------------------------
        if (np.size(names)==0) or (names is None):
            print("\nERROR acquisitionADCRAM.connectRemote(): 'names' must be a non-empty array of string.")
            return
        #-----------------------------------------------
        names = np.atleast_1d(names)
        if names[0] == "all":
            names = self.names
        #names = np.atleast_1d(names)
        #Open connection to all specified RedPitayas
        for name in names:
            #If the name is not in the list, print a warning and skip the iteration
            if not(name in self.names):
                print("\nWARNING acquisitionADCRAM.connectRemote(): RedPitaya with name '"+name+"' is not in the list.")
                continue
            RedPitaya = self.RedPitayas[name]
            remote_connection = RedPitaya["remote_connection"]
            hostname = remote_connection["hostname"]
            connected = remote_connection["connected"]
            #If the RedPitaya is already connected, print a warning and skip the iteration
            if not(connected):
                print("\nWARNING acquisitionADCRAM.connectRemote(): RedPitaya '"+name+"', with hostname "+hostname+" is not connected.") 
                continue 
            ssh_client = remote_connection["ssh_client"]
            sftp = remote_connection["sftp"]
            #Close SSH and SFTP connection
            print("\nClosing SSH and SFTP connections with '"+name+"', with hostname "+hostname+".")
            sftp.close()
            ssh_client.close()
            connected = False
            #Add the updated parameters to the datasets
            remote_connection["sftp"] = sftp
            remote_connection["ssh_client"] = ssh_client
            remote_connection["connected"] = connected
            RedPitaya["remote_connection"] = remote_connection
            self.RedPitayas["name"] = RedPitaya

    def isConnected(self, name):
        """
        This function checks whether the RedPitaya with specified name 'name' is connected with the local host through SSH and SFTP.
        
        INPUTS:
            - name: RedPitaya name - string
        OUTPUTS:
            - is_connected: True if RedPitaya 'name' is connected to local host - boolean
        
        """
        #Check for errors in the input
        #------------------------------------------------
        if name is None:
            print("\nERROR acquisitionADCRAM.isConnected(): 'name' cannot be None.")
            return
        #RedPitaya with name 'name' is not in the list
        if name not in self.names:
            print("\nERROR acquisitionADCRAM.isConnected(): RedPitaya with name '"+name+"' is not in the list.")
            return 
        #-----------------------------------------------
        return self.RedPitayas[name]["remote_connection"]["connected"]
            
     
    def acquire(self, names, acquisition_mode="asynchronous", default_sleep_time = 0.5, timeout_communication=1, keep_connected=True):
        """
        This function performs single data acquisitions form multiple RedPitayas.
        It performs the following steps:
            1) Establish a SSH connection (Secure SHell) with RedPitayas;
            2) Establish a SFTP (Secure File Transfer Protocol) connection with RedPitayas;
            3) Send the needed commands to start the acquisitions;
            4) Wait until the acquisition is finished;
            5) Transfer the acquired data to the local computer, in the preferred directory;
            6) Close the connections;
            7) Convert the acquired 14 bit integers into real voltage values.
        INPUTS:
            - names: names of the two RedPitayas to acquire from - array-like of string
            - acquisition mode: string. Can be one of the following:
                - 'synchronous': synchronous data acquisition using hardware trigger (requires hardware connection between RedPitayas);
                - 'asynchronous': independent data acquisition for each RedPitaya (no hardware trigger, and no hardware connection required).                
            - default_sleep_time: time breaks between critical steps of the acquisition - float
            - timeout_communication : float (>0)
                The time maximum time that the function waits for a reply from the remote acquisition device [s]
            - keep_connected: True if the RedPityas should keep SSH and FTP connection to the local host, after the acquisition - boolean
        
        OUTPUTS: 
            - RP_data: dictionary containing the acquired data from all RedPitayas. It is structured as follows:
                - ["names"]: names of the RedPitayas to acquire from - array-like of string
                - ["data"]: a dictionary, for each RedPitaya, structured in the following way:
                    - ["time"]: time vector [s] - array-like of float, of size n_samples;
                    - ["data_channel_1"]: channel 1 data [V] - array-like of float, of length n_samples;
                    - ["data_channel_2"]: channel 2 data [V] - array-like of float, of length n_samples;   
         """
        names_original = names
        acquisition_mode_string = acquisition_mode
        if acquisition_mode == "asynchronous":
            acquisition_mode = 0
        elif acquisition_mode == "synchronous":
            acquisition_mode = 1
        else:
            print("\nERROR acquisitionADCRAM.acquire(): 'acquisition_mode' must be one of the folliwng."+\
                  "\n - 'synchronous';"+\
                  "\n - 'asynchronous.")
        
        #Check for errors in the input
        #----------------------------------------------------------------------
        #No names specified
        if (np.size(names)==0) or (names is None):
            print("\nERROR acquisitionADCRAM.acquire(): 'names' must be a non-empty array of string.")
            return
        #---------------------------------------------------------------------- 
        if names == "all":
            names = self.names
        names = np.atleast_1d(names)
        #Open SSH and SFTP connection, if not done yet
        for name in names:
            if not(self.isConnected(name)):
                self.connectRemote(name)
    
        #----------------------------------------------------------------
        #Define individual properties
        RP, hostname, ssh_client, sftp, shell, n_samples, time, data_channel_1, data_channel_2 = [None for j in range(9)]
        string_introduction = "\nStart "+acquisition_mode_string+" remote data acquisition from RedPitayas:"
        #Perform the data acquisition from remote
        for name in names:
            RP = self.RedPitayas[name]
            hostname = RP["remote_connection"]["hostname"]
            string_introduction += "\nName: "+name+"; hostname: "+hostname
        print(string_introduction)
        #Initialize all the command line shells to be used for remote communication
        for name in names:
            RP = self.RedPitayas[name]
            ssh_client = RP["remote_connection"]["ssh_client"]
            shell = ssh_client.invoke_shell()
            RP["remote_connection"]["shell"] = shell
            self.RedPitayas[name] = RP
        #Perform the first steps in series
        for name in names:   
            print(terminal_output_text_separator)
            RP = self.RedPitayas[name]
            shell = RP["remote_connection"]["shell"]
            n_samples = RP["acquisition"]["n_samples"]         
            #1) Enter project directory-----------------------------------------------------
            print("\n"+name+": 1) Enter the acquisition project directory.")
            shell.send(command_enter_project_directory+"\n")
            #2) Load FPGA bitstream--------------------------------------------------------
            print("\n"+name+": 2) Load the right FPGA bitstream into the RedPitayas.")
            shell.send(command_load_FPGA_bistream+"\n")
            #3) Compile the acquisition script----------------------------------------------
            print("\n"+name+": 3) Compile the data acquisition script.")
            shell.send(command_compile_acquisition_script+"\n")
            while not(shell.recv_ready()):
                pass
            tm.sleep(default_sleep_time)
            #4) Enter the acquisition script directory-------------------------------------
            print("\n"+name+": 4) Enter the acquisition script directory.")
            shell.send(command_enter_script_directory+"\n")
            
            #5) Start the data acquisition script-------------------------------------------
            print("\n"+name+": 5) Start the data acquisition script.")
            shell.send(command_start_acquisition_script+str(int(n_samples))+" "+str(acquisition_mode)+"\n")
            while not(shell.recv_ready()):
                pass
            RP["remote_connection"]["shell"] = shell
            self.RedPitayas[name] = RP
            print(terminal_output_text_separator)
            
        tm.sleep(default_sleep_time)
        #6) Start acquisition on all RedPitayas
        print(terminal_output_text_separator)
        print("\n6) Start the data acquisition on all RedPitayas.")
        for name in names:          
            RP = self.RedPitayas[name]
            shell = RP["remote_connection"]["shell"]
            shell.send("\n")
            RP["remote_connection"]["shell"] = shell
            self.RedPitayas[name] = RP
        #Wait until the acquisition is finished:
        # -Read the command line output until the output string "End" is encountered
        #-----------------------------------------------------------------------------
        #Create a dictionary that keeps track of which RedPitaya has finished the acquisition
        acquisition_finished = []
        last_outputs = []
        for name in names:
            acquisition_finished.append(False)
            last_outputs.append("")
        acquisition_finished = dict(zip(names, acquisition_finished))
        last_outputs = dict(zip(names, last_outputs))
        finished_count = 0 #counter for the number of finished acquisitions
        timer_acquisition_stuck = Timer()
        acquisition_stuck = False
        #Until all RedPitayas finish their acquisition
        while finished_count < np.size(names):
            #Check all the progress of all RedPitayas 
            for name in names:
                #If the acquisition is still running on the current RedPitaya
                if not(acquisition_finished[name]):
                    RP = self.RedPitayas[name]
                    shell = RP["remote_connection"]["shell"]
                    last_output = last_outputs[name]
                    last_output_split = ""
                    if shell.recv_ready(): 
                        last_output += str(shell.recv(9999))
                    #Check whether the acquisition has ended
                    if ("Start time" in last_output) \
                        or ("Stop time" in last_output) or ("End" in last_output) or ("Duration" in last_output):  
                        acquisition_finished[name] = True
                        finished_count += 1
                    #Read the current process progress from the command line output
                    last_output_split = last_output.split("\\r")
                    #Extract only the lines where the progress is printed
                    last_output_split = [s for s in last_output_split if "%" in s]
                    #Print the progress, if those lines are not empty
                    if last_output_split:
                        print("\n  Progress in acquiring for RedPitaya '"+name+"': "+last_output_split[-1][-3:])
# =============================================================================
#                     else:
#                         if not acquisition_stuck:
#                             acquisition_stuck = True
#                             timer_acquisition_stuck.reset()
#                         else:
#                             if timer_acquisition_stuck.currentTime() > timeout_communication:
#                                 #If the acquisition has been stuck for too long, retry
#                                 print('\n Acquisition stuck: restarting.')
#                                 self.disconnectRemote('all')
#                                 return self.acquire(names_original, acquisition_mode_string, default_sleep_time, timeout_communication, keep_connected)                             
# =============================================================================
                    tm.sleep(default_sleep_time)
                    last_outputs[name] = last_output
                    RP["remote_connection"]["shell"] = shell               
                    self.RedPitayas[name] = RP
            tm.sleep(default_sleep_time)
        print(terminal_output_text_separator)
        #---------------------------------------------------------------------------
        #Transfer the acquired data to the 'save_path' directory
        #---------------------------------------------------------
        print("\n\nAcquisition completed.")
        for name in names:
            print(terminal_output_text_separator)
            print("\nRetrieve the acquired data from RedPitaya "+name+".")
            print(terminal_output_text_separator)
            RP = self.RedPitayas[name]
            hostname = RP["remote_connection"]["hostname"]
            sftp = RP["remote_connection"]["sftp"]   
            time = RP["acquisition"]["time"]
            data_channel_1 = RP["acquisition"]["data_channel_1"]
            data_channel_2 = RP["acquisition"]["data_channel_2"]
            save_path = RP["acquisition"]["save_path"]
            filename = "acquisition_"+hostname+".bin"
            print("\n"+save_path+"/"+filename)
            remote_file_path = RP_project_path+"/C-code"+"/"+filename 
            sftp.get(remote_file_path, save_path+"/"+filename)
            print("\n"+name+": File transfer completed.")
        #--------------------------------------------------------
        #Read the the .txt files, where the data are contained, and fill up the numpy vectors
            #--------------------------------------------------------
            print("\n"+name+": Load the acquired data from the transferred files.\nTransform integer values into voltage values.")
            #Read file in .txt format
# =============================================================================
#             acquisition_file = open(save_path + "/" + filename, mode="r")
#             for j in range(n_samples):
#                 line = acquisition_file.readline()
#                 line = line.strip().split(",")
#                 data_channel_1[j] = int(line[0])/2**(14-1)  #real voltage in the interval [-1, 1] from 14-bit integer
#                 data_channel_2[j] = int(line[1])/2**(14-1)  #real voltage in the interval [-1, 1] from 14-bit integer
# =============================================================================
            #Read file in binary format
            acquisition_file = open(save_path + "/" + filename, mode="rb")
            data_file = np.fromfile(acquisition_file,  dtype=np.int16)
            data_channel_1 = data_file[0::2] #even indices
            data_channel_1 = data_channel_1.astype(float)/2**(14-1) #real voltage in the interval [-1, 1] from 14-bit integer
            data_channel_2 = data_file[1::2] #odd indices
            data_channel_2 = data_channel_2.astype(float)/2**(14-1) #real voltage in the interval [-1, 1] from 14-bit integer

            acquisition_file.close()  
            #--------------------------           
            #Save data acquisition properties
            #--------------------------  
            RP["acquisition"]["time"] = time
            RP["acquisition"]["data_channel_1"] = data_channel_1
            RP["acquisition"]["data_channel_2"] = data_channel_2
            #Put the RedPityas back into the dictionary
            self.RedPitayas[name]["acquisition"]["time"] = time
            self.RedPitayas[name]["acquisition"]["data_channel_1"] = data_channel_1
            self.RedPitayas[name]["acquisition"]["data_channel_2"] = data_channel_2
            print(terminal_output_text_separator)
        #-------------------------------------------------------------------------------------
        #Close remote connection with RedPitayas if required
        if not(keep_connected):
            self.disconnectRemote(names)
        #Prepare the data to be returned
        #Put all RedPitaya data in a dictionary
        RP_data = []
        keys = ["time", "data_channel_1", "data_channel_2"]
        for name in names:
            RP = self.RedPitayas[name]
            time  = RP["acquisition"]["time"]
            data_channel_1 = RP["acquisition"]["data_channel_1"]
            data_channel_2 = RP["acquisition"]["data_channel_2"]
            RP_data.append(dict(zip(keys, [time, data_channel_1, data_channel_2])))
            
        RP_data = dict(zip(names, RP_data))        
        return  RP_data       
    
     
    def acquire2(self, names, default_sleep_time=0.5, keep_connected=True):
        """
        This function performs a simulataneous data acquisition from the analog input channels
        of two RedPitayas, at maximum sampling rate (decimation = 1).
        It performs the following steps:
            1) Establish a SSH connection (Secure SHell) with both RedPitayas;
            2) Establish a SFTP (Secure File Transfer Protocol) connection with both RedPitayas;
            3) Send the needed commands to start the simultaneous acquisition;
            4) Wait until the acquisition is finished;
            5) Transfer the acquired data to the local computer, in the preferred directory;
            6) Close the connections;
            7) Convert the acquired 14 bit integers into real voltage values.
        INPUTS:
            - names: names of the two RedPitayas to acquire from - array-like of string
            - default_sleep_time: time breaks between critical steps of the acquisition - float
            - keep_connected: True if the RedPityas should keep SSH and FTP connection to the local host, after the acquisition - boolean
        
        OUTPUTS: 
            - RP1_data, RP2_data: dictionaries containing the acquired data. Each of them is structured as follows:
                - ["time"]: time vector [s] for the j-th RedPitaya - array-like of float, of size n_samples[j], j = 0, 1;
                - ["channel_1"]: channel 1 data [V] for the j-th RedPitaya - array-like of float, of np.sizegth n_samples[j-1], j = 1, 2;
                - ["channel_1"]: channel 2 data [V] for the j-th RedPitaya - array-like of float, of np.sizegth n_samples[j-1], j = 1, 2;   
         """
        acquisition_mode = 1 #Two-RedPitayas synchronized acquisition mode
        #Check for errors in the input
        #----------------------------------------------------------------------
        #The number of specified RedPitayas is not equal to 2
        if np.size(names) != 2:
            print("\ERROR acquisitionADCRAM.acquire2(): 'names' must contain exactly two elements.")
        #The specified RedPitaya names is not in the list
        for name in names:
            if not(name in self.names):
                print("\nERROR acquisitionADCRAM.connectRemote(): RedPitaya with name '"+name+"' is not in the list.")
                return       
        #----------------------------------------------------------------------  
        #Open SSH and SFTP connection, if not done yet
        if not(self.isConnected(names[0])):
            self.connectRemote(names[0])
        if not(self.isConnected(names[1])):
            self.connectRemote(names[1])
        #Get the RedPitayas
        RP1 = self.RedPitayas[names[0]]
        RP2 = self.RedPitayas[names[1]]
        #----------------------------------------------------------------
        #Get the relevant parameters for both RedPitayas
        #Remote connection parameters
        #---------------------------
        #RedPitaya 1
        RP1_hostname = RP1["remote_connection"]["hostname"]
        RP1_ssh_client = RP1["remote_connection"]["ssh_client"]
        RP1_sftp = RP1["remote_connection"]["sftp"]
        RP1_shell = RP1["remote_connection"]["shell"]
        #RedPitaya 2
        RP2_hostname = RP2["remote_connection"]["hostname"]
        RP2_ssh_client = RP2["remote_connection"]["ssh_client"]
        RP2_sftp = RP2["remote_connection"]["sftp"]
        RP2_shell = RP2["remote_connection"]["shell"]   
        #--------------------------           
        #Data acquisition parameters
        #--------------------------  
        #RedPitaya 1
        RP1_n_samples = RP1["acquisition"]["n_samples"]
        RP1_time = RP1["acquisition"]["time"]
        RP1_data_channel_1 = RP1["acquisition"]["data_channel_1"]
        RP1_data_channel_2 = RP1["acquisition"]["data_channel_2"]
        #RedPitaya 2
        RP2_n_samples = RP2["acquisition"]["n_samples"]
        RP2_time = RP2["acquisition"]["time"]
        RP2_data_channel_1 = RP2["acquisition"]["data_channel_1"]
        RP2_data_channel_2 = RP2["acquisition"]["data_channel_2"]
        #--------------------------  
        #Perform the data acquisition from remote
        print("\nStart synchronous remote data acquisition from RedPitayas"+\
              "\n - name: "+names[0]+"; hostname: "+RP1_hostname+\
              "\n - name: "+names[1]+"; hostname: "+RP2_hostname)
        #Get shell objects for executing commands
        RP1_shell = RP1_ssh_client.invoke_shell()
        RP2_shell = RP2_ssh_client.invoke_shell()
        #1) Enter project directory-----------------------------------------------------
        print("\n1) Enter the acquisition project directory.")
        RP1_shell.send(command_enter_project_directory+"\n")
        RP2_shell.send(command_enter_project_directory+"\n")
        #2) Load FPGA bitstream--------------------------------------------------------
        print("\n2) Load the right FPGA bitstream into the RedPitayas.")
        RP1_shell.send(command_load_FPGA_bistream+"\n")
        RP2_shell.send(command_load_FPGA_bistream+"\n")
        #3) Compile the acquisition script----------------------------------------------
        print("\n3) Compile the data acquisition script.")
        RP1_shell.send(command_compile_acquisition_script+"\n")
        RP2_shell.send(command_compile_acquisition_script+"\n")
        while not(RP1_shell.recv_ready() & RP2_shell.recv_ready()):
            pass
        tm.sleep(default_sleep_time)
        #4) Enter the acquisition script directory-------------------------------------
        print("\n4) Enter the acquisition script directory.")
        RP1_shell.send(command_enter_script_directory+"\n")
        RP2_shell.send(command_enter_script_directory+"\n")
        #5) Start the data acquisition script-------------------------------------------
        print("\n5) Start the data acquisition script.")
        RP1_shell.send(command_start_acquisition_script+str(int(RP1_n_samples))+" "+str(acquisition_mode)+"\n")
        RP2_shell.send(command_start_acquisition_script+str(int(RP2_n_samples))+" "+str(acquisition_mode)+"\n")
        while not(RP1_shell.recv_ready() & RP2_shell.recv_ready()):
            pass
        tm.sleep(default_sleep_time)
        #6) Start acquisition on both RedPitaya and wait until it finishes
        print("\n6) Start the data acquisition on both RedPitayas.")
        RP1_shell.send("\n")
        tm.sleep(default_sleep_time)
        RP2_shell.send("\n")
        
        #Read the command line output until the output string "End" is encountered
        RP1_last_output = ""
        RP1_last_output_split = ""
        RP1_finished = False
        RP2_last_output = ""
        RP2_last_output_split = ""
        RP2_finished = False
        while not(RP1_finished and RP2_finished):
            if RP1_shell.recv_ready(): 
                RP1_last_output += str(RP1_shell.recv(9999))
            if RP2_shell.recv_ready():
                RP2_last_output += str(RP2_shell.recv(9999))
            #Check whether the acquisition has ended for any of the RedPitayas
            if ("End" in RP1_last_output) or ("#" in RP1_last_output):
                RP1_finished = True
            if ("End" in RP2_last_output) or ("#" in RP2_last_output):
                RP2_finished = True
            #Read the current process progress from the command line output
            RP1_last_output_split = RP1_last_output.split("\\r")
            RP2_last_output_split = RP2_last_output.split("\\r")
            #Extract only the lines where the progress is printed
            RP1_last_output_split = [s for s in RP1_last_output_split if "%" in s]
            RP2_last_output_split = [s for s in RP2_last_output_split if "%" in s]
            #Print the progress, if those lines are not empty
            if RP1_last_output_split:
                print("\n Progress in acquiring for RedPitaya '"+names[0]+"': "+RP1_last_output_split[-1][-3:])
            if RP2_last_output_split:
                print("\n Progress in acquiring for RedPitaya '"+names[1]+"': "+RP2_last_output_split[-1][-3:])
            tm.sleep(default_sleep_time)
        tm.sleep(default_sleep_time)
        #Transfer the acquired data to the 'data_path' directory------------------------------------
        print("\n\nAcquisition completed.\nRetrieve the acquired data from RedPitayas.")
        #RedPitaya1
        #filename = "acquisition_"+RP1_hostname+".txt"
        filename = "acquisition_"+RP1_hostname+".txt"
        remote_file_path = "/root/"+RP_project_path+"/C-code"+"/"+filename 
        RP1_sftp.get(remote_file_path, os.getcwd()+"/"+filename)
        #RP1_sftp.close()
        #RedPitaya2
        filename = "acquisition_"+RP2_hostname+".txt"
        remote_file_path = "/root/"+RP_project_path+"/C-code"+"/"+filename
        RP2_sftp.get(remote_file_path, os.getcwd()+"/"+filename)
        #RP2_sftp.close()
        print("\nFile transfer completed.")

        #Read the the .txt files, where the data are contained, and fill up the numpy vectors
        print("\nLoad the acquired data from the transferred .txt files.\nTransform integer values into voltage values.")
        #RedPitaya1
        filename = "acquisition_"+RP1_hostname+".txt"
        acquisition_file = open(os.getcwd() + "/" + filename, "r")
        for j in range(RP1_n_samples):
            line = acquisition_file.readline()
            line = line.strip().split(",")
            RP1_data_channel_1[j] = int(line[0])/2**(14-1)  #real voltage in the interval [-1, 1] from 14-bit integer
            RP1_data_channel_2[j] = int(line[1])/2**(14-1)  #real voltage in the interval [-1, 1] from 14-bit integer
        
        acquisition_file.close()
    
        #RedPitaya2 
        filename = "acquisition_"+RP2_hostname+".txt"
        acquisition_file = open(os.getcwd() + "/" + filename, "r")
        for j in range(RP2_n_samples):
            line = acquisition_file.readline()
            line = line.strip().split(",")
            RP2_data_channel_1[j] = float(line[0])/2**(14-1) #real voltage in the interval [-1, 1] from 14-bit integer
            RP2_data_channel_2[j] = float(line[1])/2**(14-1) #real voltage in the interval [-1, 1] from 14-bit integer
            
        acquisition_file.close()
        
        #Add the modified properties back to the datasets
        #RedPitaya 1
        RP1["remote_connection"]["ssh_client"] = RP1_ssh_client
        RP1["remote_connection"]["ssh_sftp"] = RP1_sftp
        RP1["remote_connection"]["shell"] = RP1_shell
        #RedPitaya 2   
        RP2["remote_connection"]["ssh_client"] = RP2_ssh_client
        RP2["remote_connection"]["ssh_sftp"] = RP2_sftp
        RP2["remote_connection"]["shell"] = RP2_shell        
        #--------------------------           
        #Data acquisition properties
        #--------------------------  
        #RedPitaya 1
        RP1["acquisition"]["time"] = RP1_time
        RP1["acquisition"]["data_channel_1"] = RP1_data_channel_1
        RP1["acquisition"]["data_channel_2"] = RP1_data_channel_2
        #RedPitaya 2
        RP2["acquisition"]["time"] = RP2_time
        RP2["acquisition"]["data_channel_1"] = RP2_data_channel_1
        RP2["acquisition"]["data_channel_2"] = RP2_data_channel_2  
        #Put the RedPityas back into the dictionary
        self.RedPitayas[names[0]] = RP1
        self.RedPitayas[names[1]] = RP2
        #Close remote connection with RedPitayas if required
        if not(keep_connected):
            self.disconnectRemote(names)
        #Prepare the data to be returned
        keys = ["time", "data_channel_1", "data_channel_2"]
        RP1_data = [RP1_time, RP1_data_channel_1, RP1_data_channel_2]
        RP1_data = dict(zip(keys, RP1_data))
        
        RP2_data = [RP2_time, RP2_data_channel_1, RP2_data_channel_2]
        RP2_data = dict(zip(keys, RP2_data))
        
        return RP1_data, RP2_data
    
    def set_n_samples(self, names, n_samples):
        """
        This function sets a number of samples to be acquired to the specified RedPitayas.
        If the number of samples exceedes N_SAMPLES_MAX (the maximum allowed), it is set to N_SAMPLES_MAX
        
        INPUTS:
            - names: names of the RedPitayas to be connected - array-like of string
            - n_samples: number of samples of an acquisition - array-like of float or int, same np.sizegth as 'names'
        OUTPUTS:
            - None
        """
        names = np.atleast_1d(names)
        n_samples = np.atleast_1d(n_samples)
        #Check for errors in the input
        #-------------------------------------------------
        #Empty names list
        if np.size(names)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of RedPitaya names cannot be empty.")
            return
        #Empty number of samples list
        if np.size(n_samples)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of number of acquisition samples cannot be an empty one. If not needed, set to None.")
            return     
        #Numbers of elements in 'names' and 'n_samples' do not match 
        if (np.size(names) != np.size(n_samples)):
            print("\nERROR acquisitionADCRAM.__init__(): the np.sizegth on non-empty lists 'names' and 'n_samples' must be the same.")
            return
        #----------------------------------------------  
        names = np.atleast_1d(names)
        for j in range(np.size(names)):
            self.RedPitayas[names[j]]["acquisition"]["n_samples"] = n_samples[j]
            Ts = ADC_CLK_PERIOD*self.RedPitayas[names[j]]["acquisition"]["decimation"]
            self.RedPitayas[names[j]]["acquisition"]["time"] = np.linspace(start=0, stop=(n_samples[j]-1)*Ts, num = n_samples[j]) #acquisition time vector [s]
            
    def set_save_path(self, names, save_paths):
        """
        This function sets the directory where the next acquisition data will be saved, to the specified RedPitayas.
        
        INPUTS:
            - names: names of the RedPitayas to be connected - array-like of string
            - save_paths: saving directories - array-like of string, same shape as 'names'
        OUTPUTS:
            - None
        """
        #Check for errors in the input
        #-------------------------------------------------
        #Empty names list
        if np.size(names)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of RedPitaya names cannot be empty.")
            return
        #Empty number of samples list
        if np.size(save_paths)==0:
            print("\nERROR acquisitionADCRAM.__init__(): the list of number of acquisition samples cannot be an empty one. If not needed, set to None.")
            return     
        #Numbers of elements in 'names' and 'n_samples' do not match 
        if (np.size(names) != np.size(save_paths)):
            print("\nERROR acquisitionADCRAM.__init__(): the np.sizegth on non-empty lists 'names' and 'n_samples' must be the same.")
            return
        #----------------------------------------------  
        names = np.atleast_1d(names)
        for j in range(np.size(names)):
            self.RedPitayas[names[j]]["acquisition"]["save_path"] = save_paths[j]
            

            
