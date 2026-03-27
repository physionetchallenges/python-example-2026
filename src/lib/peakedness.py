import pandas as pd
import numpy as np 
from numpy.fft import fftshift
from numpy.fft import fft
from scipy.signal import detrend, find_peaks
from time import time
import os

def _safe_ratio(numerator, denominator, default=0.0):
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return default
    value = numerator / denominator
    if np.isfinite(value):
        return value
    return default

def setParamFr(Setup):
    if 'DT' not in Setup.keys():
        Setup["DT"] = 5
        DT = 5
    else:
        DT = Setup["DT"]

    if 'Ts' not in Setup.keys():
        Setup["Ts"] = 42
        Ts = 42
    else:
        Ts = Setup["Ts"]

    if 'Tm' not in Setup.keys():
        Setup["Tm"] = 12
        Tm = 12
    else:
        Tm = Setup["Tm"]

    if 'Nfft' not in Setup.keys():
        Setup["Nfft"] = np.power(2,12)
        Nfft = np.power(2,12)
    else:
        Nfft = Setup["Nfft"]
     
    if 'K' not in Setup.keys():
        Setup["K"] = 5
        K = 5
    else:
        K = Setup["K"]
     
    if 'Omega_r' not in Setup.keys():
        Setup["Omega_r"] = np.array([0.04, 1])
        Omega_r = np.array([0.04, 1])
    else:
        Omega_r = Setup["Omega_r"]
     
    if 'ksi_p' not in Setup.keys():
        Setup["ksi_p"] = 45
        ksi_p = 45
    else:
        ksi_p = Setup["ksi_p"]

    if 'N_k' not in Setup.keys():
        Setup["N_k"] = 4
        N_k = 4
    else:
        N_k = Setup["N_k"]
      
    if 'ksi_a' not in Setup.keys():
        Setup["ksi_a"] = 85
        ksi_a = 85
    else:
        ksi_a = Setup["ksi_a"]
     
    if 'd' not in Setup.keys():
        Setup["d"] = 0.125
        d = 0.125
    else:
        d = Setup["d"]
     
    if 'b' not in Setup.keys():
        Setup["b"] = 0.8
        b = 0.8
    else:
        b = Setup["b"]
     
    if 'a' not in Setup.keys():
        Setup["a"] = 0.5
        a = 0.5
    else:
        a = Setup["a"]
     
    if 'plotflag' not in Setup.keys():
        Setup["plotflag"] = False
        plotflag =False
    else:
        plotflag = Setup["plotflag"]
     
    
    return [ DT, Ts, Tm, Nfft, K, Omega_r, ksi_p, ksi_a, d, b, a, N_k, plotflag, Setup] 

def extract_interval( x, t, int_ini, int_end ):
    # EXTRACT_INTERVAL     Very simple function to extract an interval from a signal
    # 
    #  Created by Jesús Lázaro <jlazarop@unizar.es> in 2011
    # --------
    #    Sintax: [ x_int, t_int, indexes ] = extract_interval( x, t, int_ini, int_end )
    #    In:   x = signal
    #          t = time vector
    #          int_ini = interval begin time (same units as 't')
    #          int_end = interval end time (same units as 't')
    # 
    #    Out:  x_int = interval [int_ini, int_end] of 'x'
    #          t_int = interval [int_ini, int_end] of 't'
    #          indexes = indexes corresponding to returned time interval

    x_int = x[(t>=int_ini) & (t <=int_end)]
    t_int = t[(t>=int_ini) & (t <=int_end)]

    return [ x_int, t_int ]

def normalizar_PSD( PSD, f = 'default', rango = 'default'):
    # NORMALIZAR_PSD   Normaliza una densidad espectral de potencia en el rango
    #                  de frecuencias requerido.
    # 
    #  Created by Jesús Lázaro <jlazarop@unizar.es> in 2011
    # -------
    #    Sintax: [ PSD_norm, f_PSD_norm, factor_norm ] = normalizar_PSD( PSD, f, rango )
    #    In:   PSD = Densidad espectral de potencia
    #          f = Vector de frecuencias para PSD [Por defecto: frecuencias digitales]
    #          rango = Rango [f1, f2] en el que se aplicarï¿½ la normalizaciï¿½n [Por defecto: Todo f]
    # 
    #    Out:  PSD_norm = Densidad espectral de potencia notmalizada
    #          f_PSD_norm = Vector de frecuencias para PSD_norm
    #          factor_norm = Factor de normalizaciï¿½n utilizado

    if f == 'default':
        f = np.arange(0,PSD.shape[0]) / PSD.shape[0] - 1/2
    
    if rango == 'default':
        rango = [f[0], f[-1]]
    

    # Seleccionar rango de interï¿½s:
    f_PSD_norm = f[(f>=rango[0]) & (f<=rango[1])]
    PSD = PSD[(f>=rango[0]) & (f<=rango[1])]
    if ~f_PSD_norm.any(): # El vector de frecuencias no estaba ordenado
        print('El vector de frecuencias debe estar ordenado de forma ascendente');
    

    # Calcular factor de normalizaciï¿½n y normalizar:
    ##ADD NAN removal from PSD
    factor_norm = sum(PSD)
    # print("factor_norm "+ str(factor_norm))
    if factor_norm == 0:
        # print("stop") // IMPORTANT MODIFICATION TODO
        PSD_norm = PSD
    else:
        PSD_norm = PSD/factor_norm

    return [ PSD_norm, f_PSD_norm, factor_norm ]

def init_module(kk,vars,param, plotflag):
    # function vars = init_module(kk,vars,param, plotflag)
    # This function is used for initialization and reinitialization of bar_fr
    Skl = vars["Skl"]
    t_orig = vars["t_orig"]
    t_aver = vars["t_aver"]
    f = vars["f"]
    L = vars["L"]

    DT = param["DT"]
    K = param["K"]
    ksi_p = param["ksi_p"]
    d = param["d"]

    # Increment of number of spectra for averaging
    if kk == 0: # INITIALIZATION
        N = 4*np.floor(K/2)
    else: # RE-INITIALIZATION
        N = 2*np.floor(K/2)
    
    ###### Peakedness Analysis :
    # Indexes of original spectra that take part in the average
    O = np.bitwise_and(t_orig>=t_aver[kk]-N*DT, t_orig<=t_aver[kk]+N*DT)
    W = np.arange(O.shape[0])
    O = W[O]
    # W = np.ones([O.shape[0]])
    # O1 = W[O]
    # Pre-allocate
    Xkl = np.empty((O.shape[0], L))
    Xkl[:] = np.nan
    for k in range(O.shape[0]):
        for l in range(L):
            S = Skl[:, O[k], l]
            # print(S.shape)
            # Use as reference for Pkl calculation the absolute maximum
            i_m = S.argmax()
            fr_max = f[i_m]
            
            # Define the Omega, Omega_p bands
            Omega = np.bitwise_and(f>=fr_max-d, f<=fr_max+d)
            
            # Modified limits for initialization (reduces the risk for 0.1 Hz)
            Omega_p = np.bitwise_and(f>=max(fr_max-0.4*d,0.15), f<=min(fr_max+0.4*d,0.8))
            
            # Peakedness
            # print(S[Omega])
            band_power = np.sum(S[Omega])
            peaky_power = np.sum(S[Omega_p])
            Pkl = 100*_safe_ratio(peaky_power, band_power)
            
            if Pkl >= ksi_p:
                Xkl[k,l] = 1
            else:
                Xkl[k,l] = 0

    # Initialization for averaged spectrum (if cannot be defined)
    if L>1:
        averS = np.mean(np.squeeze(np.mean(Skl[:, O, :],1)),1)
    else:
        averS = np.mean(np.mean(Skl[:, O, :],1),1)


    if kk == 0: #INITIALIZATION
        if  np.sum(Xkl[:]) > 0: # One or more spectra were peaked enough
            
            # Sum all peaky spectra
            averS = np.zeros((f.shape[0], 1))
            for k in range(O.shape[0]):
                for l in range(L):
                    if Xkl[k, l] == 1:
                        averS = averS + Skl[:, O[k], l]

        # Select the maximum in the spectrum
        i_m = averS.argmax()

        # Save in vars
        vars["bar_fr"][0] = f[i_m]
    
    else: # RE-INITIALIZATION
        # One or more spectra were peaked enough
        if  np.sum(Xkl[:]) > 0:
            # Sum all peaky spectra
            averS = np.zeros(f.shape[0])
            for k in range(O.shape[0]):
                for l in range(L):
                    if Xkl[k,l] == 1:
                        averS = averS + Skl[:, O[k], l]

            # Local maxima in the averaged spectrum
            j_pk = find_peaks(averS)
            j_pk = j_pk[0]
            pk = averS[j_pk]
            # Extra restriction : consider peaks with important power
            # j_del = pk<0.5*np.max(averS) # IMPORTANTE TODO
            j_del = pk<0.2*np.max(averS)
            pk = pk[~j_del]
            j_pk = j_pk[~j_del]
            
            # Cost function for deviation from previous fr and maximum power
            max_s = np.max(S)
            C_a = 1-_safe_ratio(np.transpose(pk), max_s, default=np.zeros_like(pk, dtype=float))
            fr_prev = vars["bar_fr"][np.max(kk,0)]
            C_f = abs(f[j_pk[:]]-fr_prev)/(2*d)
            # C_f = abs(f(i_pk(:))-fr_prev)/(Omega_r(2)-Omega(1));
            
            C = C_a +C_f
            try:
                j_min = C.argmin()
                fj = j_pk[j_min]
                vars["bar_fr"][kk] = f[fj]
            except:
                vars["bar_fr"][kk] = 0
            # Save in vars
            # vars["bar_fr"][kk] = f[fj]
            
            # if plotflag:
            #     if plt is None:
            #         raise ModuleNotFoundError("matplotlib is required when plotflag=True")
            #     plt.plot(f, averS)
            #     plt.plot(f[fj], averS[fj], '-')
            #     plt.title('Initialization - Averaged Spectrum')
            #     plt.show()

    return vars
    # # No spectra fulfill the initialization
    # if plotflag:
    #     keyboard

def compute_Xkl( Skl, f, bar_fr, O, ksi_p, ksi_a, d):
    # function [ Xkl ] = compute_Xkl( Skl, f, bar_fr, O, ksi_p, ksi_a, d)
    # Created by Spyros Kontaxis <sikontax@gmail.com> in 2019
    # Computation of peakedness for a power spectrum
    # Sintax: [ Xkl ] = compute_Xkl( Skl, f, bar_fr, O, ksi_p, ksi_a, d)
    # Inputs:
    #  Skl : Welch TF maps in a 3D matrix (f x t x DR signals)
    #  f : frequency vector (Hz)
    #  bar_fr : smoothed estimate of the respiratory rate (Hz)
    #  O : Indexes of original spectra that take part in the average
    #  ksi_p : peakedness threshold based on power concentration (%)
    #  ksi_a : peakedness threshold based on absolute maximum (%)
    #  d : half bandwith of Omega centered around bar_fr (Hz)
    # Outputs:
    # Xkl : 1-> the o:th spectrum will be used in the average
    #       0-> the o:th spectrum will not be used in the average
    #

    # % Define two search window arround the estimated respiratory rate
    Omega = np.bitwise_and(f>=bar_fr-d, f<=bar_fr+d)
    Omega_p = np.bitwise_and(f>=bar_fr-0.4*d, f<=bar_fr+0.4*d)
    
    # % Get the ammount of signals
    L = Skl.shape[2]

    # % Pre-allocate
    Xkl = np.zeros((O.shape[0],L))

    # % Loop over all segments
    for k in range(O.shape[0]):
        
        # % Loop over all signals
        for l in range(L):
            # % Select the power spectrum of one segment
            S = Skl[:, O[k], l]
            
            # % Define peakedness based on the power concentration
            band_power = np.sum(S[Omega])
            peaky_power = np.sum(S[Omega_p])
            Pkl = 100*_safe_ratio(peaky_power, band_power)

            # % Define peakedness based on the absolute maximum
            # print(max(S))
            max_s = np.max(S)
            max_band = np.max(S[Omega]) if np.any(Omega) else 0.0
            Akl = 100*_safe_ratio(max_band, max_s)
            # % If the spectrum is concidered peaky by both conditions, mark as
            # % peaky
            if np.bitwise_and(Pkl >= ksi_p, Akl >= ksi_a):
                Xkl[k,l] = 1
            else:
                Xkl[k,l] = 0

    return Xkl

def compute_fJmin( S, f, bar_fr, d):
    # function [ fJmin ] = compute_fJmin( S, f, bar_fr, d)
    # Created by Spyros Kontaxis <sikontax@gmail.com> in 2019
    # Spectral peak selection based on cost function
    # Sintax: [ fJmin ] = compute_fJmin( S, f, bar_fr, d)
    # Inputs:
    #  S : Averaged Spectrum
    #  f : frequency vector (Hz)
    #  bar_fr : smoothed estimate of the respiratory rate (Hz)
    #  d : half bandwith of Omega centered around bar_fr (Hz)
    #  Outputs:
    # fJmin : respiratory rate estimate
    # 

    # Define the search window
    Omega = np.bitwise_and(f >= bar_fr-d, f <= bar_fr+d)

    # Pre-allocate
    fJmin = np.nan

    # Locate peaks in the search window
    [peaks, properties] = find_peaks(S[Omega]) #,'SortStr','descend'

    # Put the location in the correct perspective
    lm = peaks + (Omega[:] ==1).argmax()

    # Select the frequency that corresponds to the location
    fJ = f[lm]

    # print(len(lm))
    if len(lm) > 0:
        # Compute the cost function for deviation from previous fr and maximum power
        C_f = abs(fJ-bar_fr)/(2*d)
        C_a = 1-S[lm]/max(S[Omega])
        
        # Select the minimum cost
        C = C_f+C_a
        Jmin = C.argmin()
        
        # Store the frequency with the minimum cost
        fJmin = fJ[Jmin]

    return fJmin

def peakednessCost(signals, ts, fs, Setup = {}, title = "", storeGraph = False, subjet =1):

    vars = {}
    # Set parameters / Arrange inputs
    [ DT, Ts, Tm, Nfft, K, Omega_r, ksi_p, ksi_a, d, b, a,N_k,  plotflag , Setup]  = setParamFr(Setup)

    # Start the time stamps at zero
    ts1 = ts[0]
    ts = ts-ts1
    if type(signals) == type(pd.DataFrame()):
        signals = signals.to_numpy()

    # Get the number of signals
    if len(signals.shape) == 1:
        signals = np.reshape(signals, (signals.shape[0],1))
    if signals.shape[0]<signals.shape[1]:
        signals=np.transpose(signals)
    vars["L"] = signals.shape[1]
    
    # Create a frequency vector with frequencies within the selected band
    f = fs * np.arange(0,Nfft)/Nfft - fs/2
    f_ind = np.bitwise_and(f >= Omega_r[0], f < Omega_r[1])
    vars["f"] = f[f_ind]

    # Time vector for original Welch periodograms
    # vars["t_orig"] = np.arange(Ts/2, ts[-1]+DT,DT) - Ts/2+DT #Es posible que sea esto lo que quieren pero no es lo que sale de MATLAB
    vars["t_orig"] = np.arange(Ts/2, ts[-1]- Ts/2+DT,DT) #Esto es lo que sale de MATLAB

    # Pre-allocate
    vars["Skl"] = np.empty((vars["f"].shape[0], vars["t_orig"].shape[0], vars["L"]))
    vars["Skl"][:] = np.nan

    t_for1 = time()
    for ii in range(vars["L"]):
        t_for_L = time()
            
        # Select signal
        signal = signals[:, ii]
        # signal = np.reshape(signal, (signal.shape[0],1))
        # Compute the Welch Periodgrams
        for k, ki in zip(vars["t_orig"], range(vars["t_orig"].shape[0])):
            # Begin of Ts seconds interval
            # Ws_begin = vars["t_orig"][k] - Ts/2
            Ws_begin = k - Ts/2
            
            # End of Ts seconds interval
            Ws_end = Ws_begin + Ts
            [int_Ts_sig, int_Ts_t] = extract_interval(signal, ts, Ws_begin, Ws_end); # Ts seconds interval
            S = np.zeros((vars["f"].shape[0]))
            if int_Ts_sig.shape[0] < (Tm*100)/2:
                vars["Skl"][:, ki, ii] = np.zeros((vars["f"].shape[0]))
                continue
            # Number of Tm length subintervals
            NWm = int(np.floor(2*Ts/Tm))
            I=0
            
            for i_Tm in range(NWm):
                S_i = []
                
                # Begin of Tm seconds interval
                Wm_begin = Ws_begin + (i_Tm)*Tm/2
                
                # End of Tm seconds interval
                Wm_end = min(Wm_begin + Tm, Ws_end)
                
                # Tm seconds interval
                [int_Tm_sig, int_Tm_t] = extract_interval(int_Ts_sig, int_Ts_t, Wm_begin, Wm_end)
                
                # Estimate the spectrum only for intervals without NaNs
                if ~np.isnan((int_Ts_sig.astype(float))).any():
                    S_i = abs(fftshift(fft(detrend(int_Tm_sig[:-1]), Nfft)))**2
                    # S_i = abs(fftshift(fft(int_Tm_sig[:-1], Nfft)))**2
                    S_i = S_i[f_ind]
                    [ S_i, f_PSD_norm, factor_norm ] = normalizar_PSD(S_i)
                    if ~np.isnan(S_i).any():
                        S = S + (1/NWm)*S_i #TODO  hacer una median real, que si uno falla la media se coja con los otros 3 dividido entre 3
                        I=I+1

            if I < 0.5*NWm :
                vars["Skl"][:, ki, ii] = np.zeros((vars["f"].shape[0]))
            else:
                # Define the spectrum when enough subintervals were used
                vars["Skl"][:, ki, ii] = S
            

    

    ##### Peak-conditioned spectral average:  ######
    # Pre-allocate
    N = int(np.floor(K/2))
    vars["t_aver"] = vars["t_orig"][N:-N]
    if vars["t_aver"].shape[0] == 0:
        print("No hay tiempo para promediar")
        empty_spectra = np.empty((vars["f"].shape[0], 0))
        empty_used = np.empty((0, vars["L"]))
        return np.array([]), empty_spectra, np.array([]), empty_used
    vars["Sk"] = np.empty((vars["f"].shape[0], vars["t_aver"].shape[0]))
    vars["Sk"][:] = np.nan
    vars["bar_fr"] = np.empty(( vars["t_aver"].shape[0]))
    vars["bar_fr"][:] = np.nan
    vars["hat_fr"] = np.empty((vars["t_aver"].shape[0]))
    vars["hat_fr"][:] = np.nan
    vars["Naveraged"] = np.zeros((vars["t_aver"].shape[0]))
    vars["used"] = np.zeros((vars["t_aver"].shape[0],vars["L"]))
    vars["times_used"] = np.zeros((vars["t_orig"].shape[0],vars["L"]))

    # Call the initialization module
    k_ini = 0
    plotFlag = False
    # print(vars["t_aver"])
    vars = init_module(k_ini,vars,Setup,plotFlag); #bar_fr has been initialized

    for k in np.arange(k_ini, vars["t_aver"].shape[0]):
        if k >= 1:
            k_prev = k-1
        else:
            k_prev = 0

        # Re-initialization when hat_fr has not been defined for N_k time instants
        N_k = 2#3+1#vars["N_k"]
        N_prev = np.arange(k,max(k-N_k,-1),-1)
        if np.isnan(vars["hat_fr"][N_prev]).all() and k > 2:
            vars = init_module(k_prev,vars,Setup,plotFlag) # bar_fr has been re-initialized
 
        #  Peakedness Analysis:
        # Indexes of original spectra that take part in the average
        O = np.bitwise_and(vars["t_orig"]>=vars["t_aver"][k]-N*DT, vars["t_orig"]<=vars["t_aver"][k]+N*DT)
        W = np.arange(O.shape[0])
        O = W[O]

        # Compute the peakedness of the power spectrum (1 or 0)
        Xkl = compute_Xkl(vars["Skl"], vars["f"], vars["bar_fr"][k_prev], O, ksi_p, ksi_a, d)
        

        if np.sum(Xkl) == 0:  # No spectrum was peaked
            # Store the previous respiratory frequency
            vars["bar_fr"][k] = vars["bar_fr"][k_prev]
            
            # Compute averaged spectrum just for visualization
            if vars["L"]>1:
                vars["Sk"][:, k] = np.mean(np.squeeze(np.mean(vars["Skl"][:, O, :],1)),1)
            else:
                try:
                    vars["Sk"][:,k] = np.mean(vars["Skl"][:, O, :],1)[:,0]
                except:
                    print("Cogido en el except")
                    print("Cogido en el except")
                    print("Cogido en el except")
                    print("Cogido en el except")
                    vars["Sk"][:,k] = np.nan

        else: #One or more spectra were peaked enough
            # Pre-allocate
            averS = np.zeros((vars["f"].shape[0]))

            for i_Tm in range(O.shape[0]):
                for ii in range(vars["L"]):
                    if Xkl[i_Tm,ii] == 1: # If this spectrum is considered peaky
                        # Sum all peaky spectra
                        averS = averS[:] + vars["Skl"][:, O[i_Tm], ii]
                        
                        # Store the nr of peaky spectra
                        vars["Naveraged"][k] = vars["Naveraged"][k] + 1
                        vars["used"][k,ii] = 1

            # Compute and store the averaged spectrum
            vars["Sk"][:, k] = averS/vars["Naveraged"][k]
            vars["times_used"][O,:] = vars["times_used"][O,:] + Xkl

            #Spectral peak selection
            fJmin = compute_fJmin( vars["Sk"][:, k], vars["f"], vars["bar_fr"][k_prev], d)
            
            if ~np.isnan(fJmin).any(): # Local maxima inside Omega has been found
                # Update bar_fr

                vars["bar_fr"][k] = b*vars["bar_fr"][k_prev] + (1-b)*fJmin
                
                # Update hat_fr
                if ~np.isnan(vars["hat_fr"][k_prev]).any():
                    vars["hat_fr"][k]= a*vars["hat_fr"][k_prev] + (1-a)*fJmin
                else:
                    # Use bar_fr(k-1) that always is defined, instead of hat_fr(k-1)
                    vars["hat_fr"][k]= a*vars["bar_fr"][k_prev] + (1-a)*fJmin
                
            else: # No local maxima inside Omega
                # Update bar_fr
                vars["bar_fr"][k] =  vars["bar_fr"][k_prev]
                
                # Don't Update hat_fr

    t_taver = time()

    # Extra : use bar_fr to update hat_fr when was not defined for small gaps (N_k)
    # Beginning of the intervals
    N_k = 0
    
    int_b = np.argwhere(np.isnan(vars["hat_fr"]))
    int_b1 = np.append(0,int_b)
    int_b = int_b[np.diff(int_b1)>1]

    # End of the intervals
    int_e = np.argwhere(np.isnan(vars["hat_fr"]))
    int_e1 = np.append(int_e,np.inf)
    int_e = int_e[np.diff(int_e1)>1]

    if np.isnan(vars["hat_fr"][0]) and int_e.shape[0]>1:
        
        int_e = int_e[1:]
    try:
        if (int_e[0]-int_b[0])[0] < 0:
            int_e = int_e[1:]
    except:
        print("int vacio")

    int_small = (int_e-int_b)<=(N_k-1)

    int_b = int_b[int_small]
    int_e = int_e[int_small]
    for i in range(int_small.sum()):
        vars["hat_fr"][int_b[i]:int_e[i]+1] =  vars["hat_fr"][min(int_e[i]+1,vars["hat_fr"].shape[0])]
        vars["bar_fr"][int_b[i]:int_e[i]+1] =  vars["bar_fr"][min(int_e[i]+1,vars["hat_fr"].shape[0])]



    # # Total times a signal can be used
    Ntotal = K*(vars["t_orig"].shape[0] - 2) + np.sum(np.arange(1,K))

    # Times each signal is used
    Nused = np.sum(vars["times_used"], 1)
    vars["percentage_used"] = 100*Nused/Ntotal


    vars["t_aver"] = vars["t_aver"] + ts1
    vars["t_orig"] = vars["t_orig"] + ts1
    t_fin = time()

    # if plotflag:
    #     if go is None or subplots is None:
    #         raise ModuleNotFoundError("plotly is required when plotflag=True")

    #     fig = subplots.make_subplots(rows=2,shared_xaxes=True, subplot_titles=('Peak-condition averaged EDR Spectra in '+title,"EDR/RESP signals"), row_heights=[0.7, 0.3])
        
    #     fig.add_heatmap(x=vars["t_aver"], y=vars["f"], z=vars["Sk"]/np.max(vars["Sk"]),colorscale='jet',colorbar=dict(orientation='h')) 
    #     fig.update_layout(coloraxis_showscale=False)
    #     fig.add_trace(go.Line(x=vars["t_aver"], y=vars["hat_fr"],name = 'f\u0302_r(k)'), row = 1, col=1)
    #     fig.add_trace(go.Line(x=vars["t_aver"],y=vars["bar_fr"],name= 'f\u0304_r(k)'), row = 1, col=1)              
              
    #     fig.add_trace(go.Line(x=vars["t_aver"],y=vars["used"]), row = 1, col=1)
    #     # fig.axis([vars.t_aver(1), vars.t_aver(end), vars.f(1), vars.f(end)])
    #     for i in range(signals.shape[1]):
    #         fig.add_trace(go.Line(x=ts+ts1,y=signals[:,i],name = 'Signal '+str(i)), row = 2, col=1)

    #     fig.update_layout(coloraxis_showscale=False)
    #     fig.update_yaxes(title_text="f (Hz)", row=1, col=1)
    #     fig.update_yaxes(title_text="(n.u.)", row=2, col=1)
    #     fig.update_xaxes(title_text="time (s)", row=2, col=1)
    #     if storeGraph:
    #         os.makedirs("Graphs/Peakedness/"+str(subjet), exist_ok=True)
    #         # fig.write_image(os.path.join("Graphs", "Peakedness",str(subjet),title+".png"))
    #         fig.write_html(os.path.join("Graphs", "Peakedness",str(subjet),title+".html"))
    #         # fig.write_image()
    #     else:
    #         fig.show()

    return vars["hat_fr"], vars["Sk"], vars["t_aver"], vars["used"]
    # return vars["hat_fr"], vars["Sk"], vars["bar_fr"],vars["t_aver"], vars["f"], vars["used"]