#%%
#Imports
import numpy as np
#%%
class PID:
    """
    This class defines a simple discrete time PID controller. Given a discrete time
    error signal, e_k, the output of the PID controller is
        c_k = K_P*e_k + I_value + K_D*(e_k-e_{k-1})/T
        
        where K_P is the proportional gain, K_D is the differential gain, T is 
        the acquisition sampling period and
        
                I_value = K_I * T * (\sum_{n=0}^{k} e_n)
        where K_I is the integral gain.
    """
    def __init__(self, K_P=0, K_I=0, K_D=0, I_value=0, setpoint=0, sampling_period=1, output_limits=(-np.inf, np.inf)):
        self.K_P = K_P #proportional gain
        self.K_I = K_I #integral gain
        self.K_D = K_D #differential gain
        self.I_value = I_value #accumulated integral value
        self.setpoint = setpoint #desired error signal value
        self.error_previous = 0 #last value of the error signal
        self.sampling_period = sampling_period #the error signal acquisition sampling period [s]
        self.output_limits = output_limits
        
    def resetIntegral(self):
        self.I_value=0
        
    def reset(self):
        self.__init__(setpoint=self.setpoint)
        
    def control(self, error):
        """
        This function 
        
        INPUTS
        ----------
        error : float
            The last acquired value of the error signal

        OUTPUTS
        -------
        output : float
            The control output

        """
        error = self.setpoint - error
        #Update the accumulated integral
        self.I_value += self.K_I * error * self.sampling_period
        #Calculate the control output
        output =  self.K_P*error + self.I_value + self.K_D*(error-self.error_previous)/self.sampling_period
        #Limits the control output
        output = max([output, self.output_limits[0]])
        output = min([output, self.output_limits[1]])
        #Update the previous value of the error signal
        self.error_previous = error
        return output