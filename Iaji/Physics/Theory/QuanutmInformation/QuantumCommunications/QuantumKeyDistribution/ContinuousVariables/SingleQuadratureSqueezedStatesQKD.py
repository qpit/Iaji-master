"""
This file defines classes that describe a fully-Gaussian CVQKD system, where
the p quadrature is squeezed and modulated.
This protocol is asymmetric, therefore, especially in the presence of phase noise, 
the entangling cloner attack (even asymmetric) is not a general collective attack.

This protocol has two peculiarities, that make it a special case compared to
GaussianStatesCVQKD:
    
    1) Since only one quadrature is modulated, the correlations in the unmodulated
       quadrature must be estimated using some other accessible mode. In this case, 
       two additional homodyne detectors measuring the orthogonal quadrature are placed at
       the transmitter and at the receiver in the prepare & measure version
       
    2) Since the protocol is asymmetric, there is no well-known general collective attack.
       Thus, the secret key rate is estimated from the covariance matrix that excludes 
       Eve's system and includes all the modes that compose the shared quantum state between
       Alice and Bob
"""

#%%
#Imports
import numpy as np
import sympy
sympy.init_printing()
from Iaji.Physics.Theory.QuanutmInformation import QuantumInformationUtilities as QIUtils
from quik.qip.nmodes import vacuum
from quik.qip.nmodes_symbolic import Vacuum as vacuum_symbolic
from quik.qip.nmodes_symbolic import CovarianceMatrix as CovarianceMatrix_symbolic
from quik.qip.nmodes import covariancematrix as CovarianceMatrix
#%%
print_separator = "---------------------------------------------"
#%%    
def fromNamesToVariables(string_symbols):
    """
    This function converts a string containing a symbolic expression
    into a string corresponding to the same Python expression with names of variables. 
    """
    return string_symbols.replace('(nomod)', '_nomod').replace('{', '').replace('}', '').replace('\\eta', 'eta').replace('\\sigma', 'sigma').replace('\\phi', 'phi')
#%%
class Parameter:
    """
    This class describes a generic parameter. A parameter is described by the following properties:
        -name
        
        -symbol
        
        -value: the numerical value of the parameter
        
        -expression_symbolic: a symbolic mathematical expression of the parameter, in terms of other symbols
       
        -expression_numeric: a function that allows numerical evaluation of the parameter's value form numerical input parameters
    """
    
    def __init__(self, name='x', value=None, real=True, nonnegative=True):
        self.name = name
        self.value = value
        self.symbol = sympy.symbols(names=name, real=real, nonnegative=nonnegative)
        self.expression_symbolic = None
        self.expression_numeric = None
#%%
class QKDParameters:
    """
    This class describes the parameters of the QKD system:
        
    - n_p: real, > 0.

         mean photon number of the thermal state generated by Gaussian modulation of a displaced vacuum state along quadrature p
         
    - V_s$ real, > 0.

        variance of the squeezed quadrature of the unmodulated signal,

    - n_c_q, n_c_p: real, > 0

        mean photon number of the channel thermal noise for quadratures q and p, at the input of a thermal-lossy channel 
     
    - eta_q, eta_p : real (in [0, 1])
        
        asymmetric thermal channel transmission efficiencies, for q and p quadratures
     
    - R_A_tap, R_B_tap : real (in [0, 1])
        
        power reflectivity of the beam splitter that taps the signal at Alice (Bob)
    
    - T_A_tap, T_B_tap : real (in [0, 1])
        
        power transmissivity from the tap beam splitters to the corresponding homodyne detectors (including detector loss)
    
    - T_A : real (in [0, 1])
        
        power transmissivity from Alice's tap to Bob's tap. 
        
    - \sigma_{\phi_{A}}: real, > 0
        standard deviation of the Gaussian phase noise from the transmitter's local oscillaltor
        
    - \sigma_{\phi_{B}}: real, > 0
        standard deviation of the Gaussian phase noise from the receiver's local oscillaltor
    """      
    
    def __init__(self, V_s=1, n_p=None, R_A_tap=0.05, T_A_tap=1, \
                 T_A=1, R_B_tap=0.05, T_B_tap=1, \
                 eta_q=1, eta_p=1, n_c_q=0, n_c_p=0, sigma_phi_A=0, sigma_phi_B=0):
        
        values = [V_s, n_p, R_A_tap, T_A_tap, T_A, R_B_tap, T_B_tap, eta_q, eta_p, n_c_q, n_c_p,\
                  sigma_phi_A, sigma_phi_B]
        symbol_names = ["V_s", "n_p", "R_{A_{tap}}", "T_{A_{tap}}", "T_A", "R_{B_{tap}}", "T_{B_{tap}}", "\\eta_q", \
                                  "\\eta_p", "n_{c_q}", "n_{c_p}", "\\sigma_{\\phi_A}", "\\sigma_{\\phi_B}"]
        for j in range(len(symbol_names)):
            setattr(self, fromNamesToVariables(symbol_names[j]), Parameter(name=symbol_names[j], value=values[j]))   
        [Parameter(name=symbol_names[j], value=values[j]) for j in range(len(values))]
    def toList(self):
        """
        This function returns a list of all the QKD parameters, ordered by their name
        """
        dictionary = vars(self)
        dictionary_sorted = dict(sorted(dictionary.items(), key=lambda item: item[0]))
        return list(dictionary_sorted.values())  
    
    def toDict(self, attribute='name'):
        """
        This function returns a dictionary of all the QKD parameters, with keys given
        by the parameter attribute 'attribute'.
        
        INPUTS
        ------------------
        attribute : string
            The attribute used as key to the dictionary. Refer to Parameter class for the attributes.
        """
        
        parameters = self.toList() 
        attributes = [getattr(p, attribute) for p in parameters]
        return dict(zip(attributes, parameters))
        

#%%
class QKDPrepareAndMeasureCovariances:
    """
    This class defines the entries of the prepare & measure covariance
    matrix of the QKD system, relevant to experimental parameter estimation:
        
        - V_q_A_tap, V_q_B_tap : real > 0
            variance of the q quadrature measured at Alice's and Bob's tap detectors
        
        - V_p_B, V_p_B_nomod : real > 0
            variance of the p quadrature at Bob's detector, with and without modulation
            
        - C_q : real
            covariance between the quadratures measured at A_tap and B_tap
            
        - C_p : real
            covariance between the symbols generated by Alice and the modulated p quadrature, measured 
            at Bob's detector
    """

    def __init__(self):
        symbol_names = ["V_{q_{A_{tap}}}", "V_{q_{B_{tap}}}", "V_{p_B}", "V_{p_B(nomod)}", "C_q", "C_p"]
        for j in range(len(symbol_names)):
            setattr(self, fromNamesToVariables(symbol_names[j]), Parameter(name=symbol_names[j], value=None))   
    def toList(self):
        """
        This function returns a list of all the QKD parameters, ordered by their name
        """
        dictionary = vars(self)
        dictionary_sorted = dict(sorted(dictionary.items(), key=lambda item: item[0]))
        return list(dictionary_sorted.values())     
    
#%%+
class QKDSystem:
    """
    This class describes a QKD system, which has the following properties:
        - A set of QKD parameters
        - A prepare & measure covariance matrix (quantum mechanical and classical)
        - An entanglement-based covariance matrix
        - An error correction efficiency
        - A set of experimentally measured covariances, i.e., entries of the prepare & measure classical covariance matrix
        - A secret key rate
    
    The class allows to perform the following tasks:
        - Compute the covariance matrices associated to the system, given QKD parameters (symbolically and numerically)
        - Perform parameter estimation (symbolically and numerically)
        - Estimate the secret key rate (numerically)
        
    The vacuum quadrature variance is assumed to be equal to 1. All quadrature covariances are 
    normalized to the vacuum quadrature variance.
    """    
    
    def __init__(self):
        self.parameters = QKDParameters() #QKD parameters
        self.covariances_PM = QKDPrepareAndMeasureCovariances()#prepare and measure covariances
        self.channel = None
        
        self.covariance_matrix_PM = Parameter(name='\Sigma_{PM}', value=None, nonnegative=False) #entanglement-based covariance matrix
       # self.computeCovarianceMatrixPM()
        
        self.covariance_matrix_EB = Parameter(name='\Sigma_{EB}', value=None, nonnegative=False) #entanglement-based covariance matrix
        #self.computeCovarianceMatrixEB()
        
        self.beta = Parameter(name='\\beta') #error correction efficiency [adimensional, in [0, 1]]
        self.secret_key_rate = Parameter(name='R', real=True, nonnegative=False) #secret key rate [bit/symbol]
        
        self.I_AB = Parameter(name='I_{AB}')
        self.holevo_information = Parameter(name='I_{AB}')
        
        #Assign expressions to the prepare & measure covariances
        parameter_names = list(vars(self.parameters).keys())
        parameter_names.sort()
        
    def computeCovarianceMatrixPM(self, form='symbolic'): #TODO
        """
        This function constructs the prepare & measure covariance matrix.
        
        INPUTS
        ----------
        form : string
            'symbolic' or 'numeric'      
        OUTPUTS:
        -------
        CM : 2D array-like of float
            The prepare & measure (classical) covariance matrix. The modes of the field are ordered as follows:
                1 The classical mode describing the Gaussian-distributed QKD symbols generated by Alice
                2 (A_tap): the tap mode at Alice
                3 (B_tap): the tap mode at Bob
                4 (B): Bob's mode
        """
        
        if form=='symbolic':
            CM = vacuum_symbolic(4)
            parameter_names = ["V_s", "n_p", "R_A_tap", "T_A_tap", "T_A", "R_B_tap", "T_B_tap", \
                               "eta_q", "eta_p", "n_c_q", "n_c_p", "sigma_phi_A", "sigma_phi_B"]
            covariance_names = ["V_q_A_tap", "V_q_B_tap", "V_p_B", "V_p_B_nomod", "C_q", "C_p"]
            #Load all the QKD parameter symbols
            V_s, n_p, R_A_tap, T_A_tap, T_A, R_B_tap, T_B_tap, \
                               eta_q, eta_p, n_c_q, n_c_p, sigma_phi_A, sigma_phi_B = \
            [getattr(self.parameters, name).symbol for name in parameter_names]
            #Load PM covariance parameters (not symbols)
            V_q_A_tap, V_q_B_tap, V_p_B, V_p_B_nomod, C_q, C_p =\
                    [getattr(self.covariances_PM, name) for name in covariance_names] 
            if self.channel == "uknown":
                raise InvalidOptionError("The channel should be known for estimating the prepare & measure covariance matrix")     
            else:
                #Construct the covariance matrix
                CM[0, 0] = 0
                CM[1, 1] = 2*n_p
                CM[0, 1] = CM[1, 0] = 0
                CM[2, 2] = 1/V_s
                CM[3, 3] = V_s + 2*n_p
                CM[0, 2] = CM[2, 0] = 0
                CM[1, 3] = CM[3, 1] = 2*n_p
                CM = CM.bs(2, 3, R=R_A_tap)
                CM = CM.opticalefficiency(1, T_A_tap, T_A, 1)
                if self.channel=="thermal-lossy (asymmetric)": #TODO
                    T_c = vacuum_symbolic(4)
                    T_c[4, 4] = sympy.sqrt(eta_q) * vacuum_symbolic(1)
                    T_c[5, 5] = sympy.sqrt(eta_p) * vacuum_symbolic(1)
                    N_c = vacuum_symbolic(4)
                    N_c[4, 4] = (1-eta_q)*(2*n_c_q+1)
                    N_c[5, 5] = (1-eta_p)*(2*n_c_p+1)
                    N_c[0, 0] = N_c[1, 1] = N_c[2, 2] = N_c[3, 3] = N_c[6, 6] = N_c[7, 7]
                    CM = T_c @ CM @ T_c.T+ N_c  
                CM = CM.bs(3, 4, R=R_B_tap)
                CM = CM.opticalefficiency(1, 1, T_B_tap, 1)
                self.covariance_matrix_PM.expression_symbolic = CM
                parameter_names = list(vars(self.parameters).keys())
                parameter_names.sort()
                #Assigna analytical expression to the prepare & measure covariances
                V_q_A_tap.expression_symbolic = CM[2, 2]
                V_q_B_tap.expression_symbolic = CM[4, 4]
                V_p_B.expression_symbolic = CM[7, 7]
                V_p_B_nomod.expression_symbolic = V_p_B.expression_symbolic.subs([(n_p, 0)]) 
                C_p.expression_symbolic = CM[1, 7]
                C_q.expression_symbolic = CM[2, 4]
            #Construct a matrix of python functions associated with the expression of the 
            #entries of the covariance matrix
            shape = np.shape(self.covariance_matrix_PM.expression_symbolic)
            CM_numeric = np.empty(shape, dtype=object)
            for j in range(shape[0]):
                for k in range(shape[1]):
                        CM_numeric[j, k] = sympy.lambdify(parameter_names, fromNamesToVariables(str(CM[j, k])))
            self.covariance_matrix_PM.expression_numeric = CM_numeric
        else:
            if self.covariance_matrix_PM.expression_symbolic is None:
                self.computeCovarianceMatrixPM(form='symbolic')
            parameter_values = [p.value for p in self.parameters.toList()]
            shape = np.shape(self.covariance_matrix_PM.expression_symbolic)
            CM = np.zeros(shape)
            for j in range(shape[0]):
                for k in range(shape[1]):
                    CM[j, k] = self.covariance_matrix_PM.expression_numeric[j, k](*parameter_values) 
            self.covariance_matrix_PM.value = CM
            self.covariances_PM.V_q_B.value = CM[2, 2]
            self.covariances_PM.V_p_B.value = CM[5, 5]
            self.covariances_PM.C_q.value = CM[0, 2]
            self.covariances_PM.C_p.value = CM[1, 5]
            #Compute the value of the covariance without modulation
            parameter_values_nomod = parameter_values
            parameter_names = list(vars(self.parameters).keys())
            parameter_names.sort()
            for j in range(len(parameter_names)):
                parameter_name = parameter_names[j]
                if parameter_name == "n_p":
                    parameter_values_nomod[j] = 0
            self.covariances_PM.V_q_B_nomod.value = self.covariance_matrix_PM.expression_numeric[2, 2](*parameter_values_nomod)
            self.covariances_PM.V_p_B_nomod.value = self.covariance_matrix_PM.expression_numeric[5, 5](*parameter_values_nomod)

        return CM
    
   
    def computeCovarianceMatrixEB(self, form='symbolic'): #TODO
        """
        This function constructs the entanglement-based covariance matrix
        
        INPUTS
        ----------
        form : string
            'symbolic' or 'numeric'
        print_warnings : boolean
            If True, sanity check warnings on the covariance matrix are printed 
            after the calculation.
             
        OUTPUTS:
        -------
        CM : 2D array-like of float
            The entanglement-based covariance matrix. The modes of the field are ordered as follows:
                1. A': mode kept by Alice to perform asymmetric homodyne detections on
                2. B: mode input to Bob's asymmetric homodyne detection
        """
              #------------------------------------------
        parameter_names = ["V_s", "n_q", "n_p", "eta", "n_c", "R_B"]
        covariance_names = ['V_q_B', 'V_p_B','C_q', 'C_p']
        if form=='symbolic':
            CM = vacuum_symbolic(2)
            #Load all the QKD parameter symbols
            V_s, n_q, n_p, eta, n_c, R_B = \
            [getattr(self.parameters, name).symbol for name in parameter_names]
            #Load PM covariance parameters (not symbols)
            V_q_B, V_p_B, C_q, C_p =\
                    [getattr(self.covariances_PM, name) for name in covariance_names] 
            #Construct the covariance matrix
            nu = sympy.sqrt((1+2*n_q*V_s)*(V_s+2*n_p)/V_s)
            mu = sympy.sqrt((1+2*n_q*V_s)/(V_s*(V_s+2*n_p)))
            CM = CM.epr_state(1, 2, mu=nu)
            CM = CM.squeeze(0, -sympy.log(mu)/2)
            if self.channel == "thermal-lossy (symmetric)":
                T_c = vacuum_symbolic(2)
                T_c[2:4, 2:4] = sympy.sqrt(eta) * vacuum_symbolic(1)
                N_c = vacuum_symbolic(2)
                N_c[2:4, 2:4] = (1-eta)*(2*n_c+1)*vacuum_symbolic(1)
                N_c[0, 0] = N_c[1, 1] = 0
                CM = T_c @ CM @ T_c.T+ N_c            
            self.covariance_matrix_EB.expression_symbolic = CovarianceMatrix_symbolic(sympy.simplify(CM))
            #Construct a matrix of python functions associated with the expression of the 
            #entries of the covariance matrix
            shape = np.shape(self.covariance_matrix_EB.expression_symbolic)
            CM_numeric = np.empty(shape, dtype=object)
            for j in range(shape[0]):
                for k in range(shape[1]):
                    CM_numeric[j, k] = sympy.lambdify(parameter_names+covariance_names, fromNamesToVariables(str(CM[j, k])))
            self.covariance_matrix_EB.expression_numeric = CM_numeric
        else:
            if self.covariance_matrix_EB.expression_symbolic is None:
                self.computeCovarianceMatrixEB(form='symbolic')
            input_values = []
            for name in parameter_names:
                input_values.append(getattr(self.parameters, name).value)
            for name in covariance_names:
                input_values.append(getattr(self.covariances_PM, name).value)
            shape = np.shape(self.covariance_matrix_EB.expression_symbolic)
            CM = np.zeros(shape)
            for j in range(shape[0]):
                for k in range(shape[0]):
                    CM[j, k] = self.covariance_matrix_EB.expression_numeric[j, k](*input_values)
            self.covariance_matrix_EB.value = CM
        return CM
    

    def computeKeyRate(self, form='symbolic', print_warnings=False): #TODO
        """
        This function calculates the secret key rate from the entanglement-based covariance matrix.
        
        INPUTS
        ----------
        print_warnings: boolean
            If 'True', the function will print eventual warnings regarding the calculations. 
        
        OUTPUTS:
        -------
        The value of the secret key rate of the system : float 
        """
        if form=='symbolic':
            attribute = 'symbol'
        else:
            attribute = 'value'
            #Load all the QKD parameters into shorter-named variables
            #Load the needed variances and covariances
        R_B = getattr(self.parameters.R_B, attribute)
        [n_q, n_p] = [getattr(self.parameters.n_q, attribute), getattr(self.parameters.n_p, attribute)]
        V_p_B, C_p = [getattr(self.covariances_PM.V_p_B, attribute), getattr(self.covariances_PM.C_p, attribute)]
        V_q_B, C_q = [getattr(self.covariances_PM.V_q_B, attribute), getattr(self.covariances_PM.C_q, attribute)]
        beta = getattr(self.beta, attribute)
        #Compute the Shannon's mutual information of the modulated (squeezed) signal quadrature measured at the receiver's homodyne detector 'p' 
        #and the modulation signal (QKD symbols) prepared at the transmitter 
       # if form=="numeric":
            #print("I_AB: %0.3f"%I_AB)
        #Compute the entanglement-based covariance matrix of the system
        if form == 'symbolic':     
            CM = self.covariance_matrix_EB.expression_symbolic
            CM_B = vacuum_symbolic(3)
        else:
            CM = self.computeCovarianceMatrixEB(form='numeric')
            CM_B = vacuum(3)
        I_AB = 0
        if form=="numeric":
            if n_p != 0:
                I_AB = QIUtils.mutualInformation(variance_1=CM[1, 1], variance_2=CM[3, 3], covariance=CM[1, 3]) #[bit]
            if n_q != 0:
                I_AB += QIUtils.mutualInformation(variance_1=CM[0, 0], variance_2=CM[2, 2], covariance=CM[0, 2])
        #Compute the entanglement-based covariance matrix after asymmetric homodyne detection of the receiver's mode, along the 'p' quadrature     
        CM_B[0:4, 0:4] = CM
        CM_B = CM_B.pick_modes(1, 3, 2)
        CM_B = CM_B.bs(2, 3, R=R_B)
        if R_B == 0:
            CM_B = CM_B.pick_modes(1, 2)
            CM_B = CM_B.homodyne_detection(2, "p")
        elif R_B == 1:
            CM_B = CM_B.pick_modes(1, 2)
            CM_B = CM_B.homodyne_detection(2, "x")
        else:
            CM_B = CM_B.homodyne_detection(2, "x").homodyne_detection(2, "p")
        
        if print_warnings and CM_B.physicality<0:
            print("\nWarning in keyRate(): the entanglement-based covariance matrix after homodyne detection of the receiver's mode is unphysical.")
        #Compute the von Neumann entropy of the quantum states described by the covariance matrix with and without homodyne detection at the receiver
        S = QIUtils.VonNeumannEntropy(CM, print_warnings=print_warnings) #without homodyne detection
        #print("S: %0.3f"%S)
        S_B = QIUtils.VonNeumannEntropy(CM_B, print_warnings=print_warnings) #with homodyne detection
        #print("S_B: %0.3f"%S_B)
        #Compute the Holevo information
        holevo_information = S - S_B
        #Compute the secret key rate
        R = beta*I_AB - holevo_information
        if form == 'symbolic':
            attribute = 'expression_symbolic'
            R = sympy.simplify(R)
        setattr(self.secret_key_rate, attribute, R)    
        setattr(self.I_AB, attribute, I_AB)
        setattr(self.holevo_information, attribute, holevo_information)
        return R 
    
        
    def estimateParameters(self, parameter_names=None, covariance_names = None, \
                           form='symbolic'):
        """
        This function performs parameter estimation of the parameters specified by
        
        Parameters
        ----------
        parameter_names : array-like of string
            The names of the parameters to be estimated, in variables format (e.g., ['eta'])
        covariance_names : array-like of string
            The names of the quadrature covariances from which the parameters are to be estimated, in variable format (e.g., ['V_q_Tx'])
        mode : string
            'symbolc' 'numeric'.

        Returns
        -------
        None.

        """

        #Load the target parameters
        if parameter_names is None:
            raise ParameterEstimationError('No parameters were specified.')  
        equations_specified = covariance_names is not None
        #Load the prepare & measure covariances as parameters
        parameter_names.sort()
        covariance_names.sort()
        parameters = [getattr(self.parameters, name) for name in parameter_names]
        #Load the known parameteres
        known_parameter_names = [name for name in list(vars(self.parameters).keys()) if name not in parameter_names+['eta', 'w_q', 'w_p']]
        known_parameter_names.sort()
        known_parameters = [getattr(self.parameters, name) for name in known_parameter_names]
        analytical_expression_missing = sum([p.expression_symbolic is None for p in parameters]) > 0
        #Check for possible errors in the input
        if not(equations_specified) and analytical_expression_missing:
            raise ParameterEstimationError("Analytical expressions of the parameters to be estimated are not available, but prepare & measure covariances were not specified."\
                                           +"\n You need to specify the covariances which parameters are estimated from.")
        covariances= [getattr(self.covariances_PM, name) for name in covariance_names]
        #Symbolic parameter estimation 
        if form=='symbolic':
            #Set up the system of equations for estimating parameters
            system = []
            for covariance in covariances:
                print(covariance.name)
                system.append(sympy.Eq(covariance.expression_symbolic, covariance.symbol))
            system = tuple(system)
            parameter_symbols = [p.symbol for p in parameters]
            #Solve the system of equations
            try:
                solutions = sympy.solve(system, tuple(parameter_symbols), dict=True)[0]
            except:
                raise ParameterEstimationError('No solution was found')
            #Set the symbolic expressions into the QKD system
            for solution_symbol in list(solutions.keys()):
                parameter = [p for p in parameters if p.name==solution_symbol.name][0]
                parameter.expression_symbolic = solutions[solution_symbol]
                parameter.expression_numeric = sympy.lambdify(known_parameter_names+covariance_names, fromNamesToVariables(str(solutions[solution_symbol])))  
        else: 
            if analytical_expression_missing:
                self.estimateParameters(parameter_names=parameter_names, covariance_names=covariance_names, form='symbolic')
            #Load the known parameters as symbols or values
            known_parameters = [p.value for p in known_parameters]
            #Load the prepare & measure covariances as symbols or values
            covariances = [c.value for c in covariances]
            for parameter in parameters:
                parameter.value = parameter.expression_numeric(*(known_parameters+covariances))
                        

#%%
#Define exceptions
class ParameterEstimationError(Exception):
    def __init__(self, error_message):
        print(error_message)
        
class InvalidOptionError(Exception):
    def __init__(self, error_message='The selected option is invalid'):
        print(error_message)
        