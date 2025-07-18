# This file is part of Eutropy
# Copyright (c) 2025 Burak Kaynaroglu
# This program is free software distributed under the GPL-3.0 License 
# A copy of the GPL-3.0 License can be found at 
# https://github.com/kaynarob/Eutropy/blob/main/LICENSE.md

# Eutropy model kinetics for n-dimentional configuration - X box(es)


import math
from numba import njit
import config

cache_option = config.cache_option
Altitude = config.Altitude
H1 = config.H1
H2 = config.H2

# pelagic model constants
k_growth = 0
k_resipration = 1
k_mortality = 2
k_excration = 3
k_salt_death = 4
v_set_Cpy = 5
K_SN = 6
K_SP = 7
K_Sl_salt = 8
K_Sl_ox_Cpy = 9
k_c_decomp = 10
k_n_decomp = 11
k_p_decomp = 12
v_set_Cpoc = 13
v_set_Cpon = 14
v_set_Cpop = 15
k_c_mnr_ox = 16
k_c_mnr_ni = 17
k_n_mnr_ox = 18
k_n_mnr_ni = 19
k_p_mnr_ox = 20
k_p_mnr_ni = 21
K_Sl_Cpoc_decomp = 22
K_Sl_Cpon_decomp = 23
K_Sl_Cpop_decomp = 24
K_Sl_c_mnr_ox = 25
K_Sl_c_mnr_ni = 26
K_Sl_ox_mnr_c = 27
K_Si_ox_mnr_c = 28
K_Sl_ni_mnr_c = 29
K_Sl_n_mnr_ox = 30
K_Sl_n_mnr_ni = 31
K_Sl_ox_mnr_n = 32
K_Si_ox_mnr_n = 33
K_Sl_ni_mnr_n = 34
K_Sl_p_mnr_ox = 35
K_Sl_p_mnr_ni = 36
K_Sl_ox_mnr_p = 37
K_Si_ox_mnr_p = 38
K_Sl_ni_mnr_p = 39
k_nitrification = 40
K_Sl_nitr = 41
K_Sl_nitr_ox = 42
k_denitrification = 43
K_Sl_denitr = 44
K_Si_denitr_ox = 45
k_raer = 46
K_be = 47
I_s = 48
theta_growth = 49
theta_resipration = 50
theta_mortality = 51
theta_excration = 52
theta_c_decomp = 53
theta_n_decomp = 54
theta_p_decomp = 55
theta_c_mnr_ox = 56
theta_n_mnr_ox = 57
theta_p_mnr_ox = 58
theta_c_mnr_ni = 59
theta_n_mnr_ni = 60
theta_p_mnr_ni = 61
theta_nitr = 62
theta_denitr = 63
theta_rear = 64
a_C_chl = 65
a_N_C = 66
a_P_C = 67
a_O2_C = 68
pabsorption = 69

# sediment model constants
omega_01_POC                  = 0
omega_12_POC1                 = 1
w_burial_POC1                 = 2
k_dissol_POC1                 = 3
k_dissol_POC2                 = 4
omega_01_PON                  = 5
omega_12_PON1                 = 6
w_burial_PON1                 = 7
k_dissol_PON1                 = 8
k_dissol_PON2                 = 9
omega_01_POP                  = 10
omega_12_POP1                 = 11
w_burial_POP1                 = 12
k_dissol_POP1                 = 13
k_dissol_POP2                 = 14
Kl_01_DOC                     = 15
Kl_12_DOC1                    = 16
Kl_01_DON                     = 17
Kl_12_DON1                    = 18
Kl_01_DOP                     = 19
Kl_12_DOP1                    = 20
Kl_01_NH4                     = 21
Kl_12_NH4_1                   = 22
k_nitr_NH4_1                  = 23
Kl_01_NO3                     = 24
Kl_12_NO3                     = 25
k_denitr_NO3_1                = 26
k_denitr_NO3_2                = 27
Kl_01_PO4                     = 28
Kl_12_PO4_1                   = 29
K_HS_sed_POC_1_disl           = 30
K_HS_sed_POC_2_disl           = 31
K_HS_sed_PON_1_disl           = 32
K_HS_sed_PON_2_disl           = 33
K_HS_sed_POP_1_disl           = 34
K_HS_sed_POP_2_disl           = 35
K_HS_sed_Nitr                 = 36
K_HS_sed_deNitr               = 37
k_DOC_1_Miner_dox             = 38
k_DON_1_Miner_dox             = 39
k_DOP_1_Miner_dox             = 40
K_HS_DOC_1_Minr_lim_dox       = 41
K_HS_DON_1_Minr_lim_dox       = 42
K_HS_DOP_1_Minr_lim_dox       = 43
K_HS_DOC_1_Minr_dox           = 44
K_HS_DON_1_Minr_dox           = 45
K_HS_DOP_1_Minr_dox           = 46
k_DOC_1_Miner_no3_1           = 47
K_HS_DOC_1_Minr_inhb_dox_1    = 48
K_HS_DOC_1_Minr_lim_no3_1     = 49
K_HS_DOC_1_Minr_no3_1         = 50
k_DON_1_Miner_no3_1           = 51
K_HS_DON_1_Minr_inhb_dox_1    = 52
K_HS_DON_1_Minr_lim_no3_1     = 53
K_HS_DON_1_Minr_no3_1         = 54
k_DOP_1_Miner_no3_1           = 55
K_HS_DOP_1_Minr_inhb_dox_1    = 56
K_HS_DOP_1_Minr_lim_no3_1     = 57
K_HS_DOP_1_Minr_no3_1         = 58
k_adsorp_PO4                  = 59
K_HS_adsorp_PO4               = 60


# =========================================================================== #
# =================== Pelagic process rates calculation \ =================== #
@njit(cache=cache_option)
def pelagic_process_rates(C_t, T, V, H, I_a, salinity, f_day, kmc):
    
    Cpy, Cpoc, Cpon, Cpop, Cdoc, Cdon, Cdop, Cam, Cni, Cph, Cox = C_t
    
    
    def calculate_light_limitation(ChlA, H, kmc, I_a):
        K_e = kmc[K_be] + (0.0088*ChlA) + (0.054*(ChlA**(2/3)))
        constant_e = 2.718
        X_I = (((constant_e*f_day)/(K_e*H)) * 
                (math.exp((-I_a/kmc[I_s])*math.exp(-K_e*H)) - 
                 math.exp(-I_a/kmc[I_s])))
        return X_I
    
    def calculate_nutrient_limitation(Cni, Cam, Cph, kmc):
        X_N_N = (Cni+Cam)/(kmc[K_SN]+Cni+Cam)
        X_N_P = Cph/(kmc[K_SP]+Cph)
        return X_N_P 
    # for phosporous limited systems use X_N_P; otherwise use "min(X_N_P, X_N_N)" or X_N_P*X_N_N.
    # for more information, please refer to Kaynaroglu et al. (2025; https://doi.org/10.1016/j.ecoinf.2025.103213)

    def calculate_O2_saturation(T, kmc, salinity):
        TKelvin = T + 273.15
        AltEffect = (100 - (0.0035 * 3.28083 * Altitude)) / 100
        ln_stemp = (-139.34411 + (1.575701E+5 / TKelvin) - 
                    (6.642308E+7 / (TKelvin ** 2)) + (1.243800E+10 / (TKelvin ** 3)) 
                    - (8.621949E+11 / (TKelvin ** 4)))
        ln_ssalt = (salinity * ((1.7674E-2) - (1.754E+1 / TKelvin) 
                            + (2.1407E+3 / (TKelvin ** 2))))
        O2_sat_fresh = math.exp(ln_stemp)
        O2_sat_salt = math.exp(ln_ssalt)
        
        return AltEffect * (O2_sat_fresh - O2_sat_salt)
    
    
    ChlA = (Cpy / kmc[a_C_chl]) * 1000
    X_I = calculate_light_limitation(ChlA, H, kmc, I_a)
    X_N = calculate_nutrient_limitation(Cni, Cam, Cph, kmc)
    # Cpy: Phytoplankton-Carbon processes
    r_Phyto_Death_Mortality = kmc[k_mortality] * kmc[theta_mortality]**(T-20) * Cpy
    r_Phyto_Death_Salinity = kmc[k_salt_death] * (salinity/(salinity+kmc[K_Sl_salt])) * Cpy
    R_Cpy_Growth = (kmc[k_growth] * (kmc[theta_growth]**(T - 20)) 
                     *(Cox/(Cox+kmc[K_Sl_ox_Cpy])) * X_I * X_N * Cpy)
    R_Cpy_Respiration = kmc[k_resipration] * kmc[theta_resipration]**(T-20) * Cpy
    R_Cpy_Excration = kmc[k_excration] * kmc[theta_excration]**(T-20) * Cpy
    R_Cpy_Death = (r_Phyto_Death_Mortality + r_Phyto_Death_Salinity)
    R_Cpy_Settling = (kmc[v_set_Cpy]/H) * Cpy
    R_Cpy = (+R_Cpy_Growth
             -R_Cpy_Respiration
             -R_Cpy_Excration
             -R_Cpy_Death
             -R_Cpy_Settling)

    # Cpoc: Particulate organic carbon processes
    R_Cpoc_Decomposition = kmc[k_c_decomp] * (kmc[theta_c_decomp]**(T-20)) * (Cpoc/(Cpoc+kmc[K_Sl_Cpoc_decomp])) * Cpoc
    R_Cpoc_Settling = (kmc[v_set_Cpoc]/H) * Cpoc
    R_Cpoc = (+R_Cpy_Death
              -R_Cpoc_Decomposition
              -R_Cpoc_Settling)
   
    # Cpon: Particulate organic nitrogen processes
    R_Cpon_Decomposition = kmc[k_n_decomp] * (kmc[theta_n_decomp]**(T-20)) * (Cpon/(Cpon+kmc[K_Sl_Cpon_decomp])) * Cpon
    R_Cpon_Settling = (kmc[v_set_Cpon]/H) * Cpon
    R_Cpon = (+kmc[a_N_C]*R_Cpy_Death
             -R_Cpon_Decomposition
             -R_Cpon_Settling)
    
    # Cpop: Particulate organic phosphorous processes
    R_Cpop_Decomposition = kmc[k_p_decomp] * (kmc[theta_p_decomp]**(T-20)) * (Cpop/(Cpop+kmc[K_Sl_Cpop_decomp])) * Cpop
    R_Cpop_Settling = (kmc[v_set_Cpop]/H) * Cpop
    R_Cpop = (+kmc[a_P_C]*R_Cpy_Death
             -R_Cpop_Decomposition
             -R_Cpop_Settling)

    # Mineralization of dissolved organic matters
    def mineralization_by_ox_C():
        return kmc[k_c_mnr_ox] * (kmc[theta_c_mnr_ox] ** (T - 20)) * \
            Cox / (kmc[K_Sl_ox_mnr_c] + Cox) * \
                (Cdoc / (Cdoc + kmc[K_Sl_c_mnr_ox])) * Cdoc
    
    def mineralization_by_ox_N():
        return kmc[k_n_mnr_ox] * (kmc[theta_n_mnr_ox] ** (T - 20)) * \
            Cox / (kmc[K_Sl_ox_mnr_n] + Cox) * \
                (Cdon / (Cdon + kmc[K_Sl_n_mnr_ox])) * Cdon
    
    def mineralization_by_ox_P():
        return kmc[k_p_mnr_ox] * (kmc[theta_p_mnr_ox] ** (T - 20)) * \
            Cox / (kmc[K_Sl_ox_mnr_p] + Cox) * \
                (Cdop / (Cdop + kmc[K_Sl_p_mnr_ox])) * Cdop
    
    def mineralization_by_ni_C():
        return kmc[k_c_mnr_ni] * (kmc[theta_c_mnr_ni] ** (T - 20)) * \
            (1 - Cox / (kmc[K_Si_ox_mnr_c] + Cox)) * \
                Cni / (kmc[K_Sl_ni_mnr_c] + Cni) * \
                    (Cdoc / (Cdoc + kmc[K_Sl_c_mnr_ni])) * Cdoc
    
    def mineralization_by_ni_N():
        return kmc[k_n_mnr_ni] * (kmc[theta_n_mnr_ni] ** (T - 20)) * \
            (1 - Cox / (kmc[K_Si_ox_mnr_n] + Cox)) * \
                Cni / (kmc[K_Sl_ni_mnr_n] + Cni) * \
                    (Cdon / (Cdon + kmc[K_Sl_n_mnr_ni])) * Cdon
    
    def mineralization_by_ni_P():
        return kmc[k_p_mnr_ni] * (kmc[theta_p_mnr_ni] ** (T - 20)) * \
            (1 - Cox / (kmc[K_Si_ox_mnr_p] + Cox)) * \
                Cni / (kmc[K_Sl_ni_mnr_p] + Cni) * \
                    (Cdop / (Cdop + kmc[K_Sl_p_mnr_ni])) * Cdop
    
    # Cdoc: Dissolved organic carbon processes
    R_Cdoc_Mineralization = mineralization_by_ox_C() + mineralization_by_ni_C()
    R_Cdoc = (+R_Cpy_Excration
             +R_Cpoc_Decomposition
             -R_Cdoc_Mineralization)
   
    # Cdon: Dissolved organic nitrogen processes
    R_Cdon_Mineralization = mineralization_by_ox_N() + mineralization_by_ni_N()
    R_Cdon = (+kmc[a_N_C]*R_Cpy_Excration
             +R_Cpon_Decomposition
             -R_Cdon_Mineralization)
    
    # Cdop: Dissolved organic phosphorous processes
    R_Cdop_Mineralization = mineralization_by_ox_P() + mineralization_by_ni_P()
    R_Cdop = (+kmc[a_P_C]*R_Cpy_Excration
             +R_Cpop_Decomposition
             -R_Cdop_Mineralization)
    
    # Cam: Ammonia processes
    prefam = (Cam * (Cni / ((kmc[K_SN]+Cam)*(kmc[K_SN]+Cni))) + Cam * (kmc[K_SN] / ((Cam+Cni)*(kmc[K_SN]+Cni)))) # (Thomann and Fitzpatrick,1982)
    R_Nitrification = (kmc[k_nitrification] * (kmc[theta_nitr]**(T-20)) * (Cox/(Cox+kmc[K_Sl_nitr_ox])) * (Cam/(Cam+kmc[K_Sl_nitr]))) * Cam
    R_Cam = (+R_Cdon_Mineralization
             +kmc[a_N_C]*R_Cpy_Respiration
             -kmc[a_N_C]*prefam*R_Cpy_Growth
             -R_Nitrification)
    
    # Cni: Nitrate processes
    R_Denitrification = ((kmc[k_denitrification] * (kmc[theta_denitr]**(T-20)) * 
                          (kmc[K_Si_denitr_ox]/(Cox+kmc[K_Si_denitr_ox])) * (Cni/(Cni+kmc[K_Sl_denitr]))) * Cni)
    R_Cni = (+R_Nitrification
             -R_Denitrification 
             -kmc[a_N_C]*(1-prefam)*R_Cpy_Growth)
    
    # Cph: Phosphate processes
    R_Cph = (+R_Cdop_Mineralization
             +kmc[a_P_C]*R_Cpy_Respiration
             -kmc[a_P_C]*R_Cpy_Growth
             -kmc[pabsorption]*Cph
             )
    
    # Cox: Dissolved oxygen processes
    O2_sat = calculate_O2_saturation(T, kmc, salinity)
    R_Reaeration = kmc[k_raer] * kmc[theta_rear] * (O2_sat-Cox)
    R_Cox = (+R_Reaeration
            +kmc[a_O2_C]*R_Cpy_Growth
            -kmc[a_O2_C]*R_Cpy_Respiration 
            -(32/12)*mineralization_by_ox_C()
            -(64/14)*R_Nitrification
            +(5/4)*(32/14)*R_Denitrification)
    
    
    C_t[0]  = R_Cpy
    C_t[1]  = R_Cpoc
    C_t[2]  = R_Cpon
    C_t[3]  = R_Cpop
    C_t[4]  = R_Cdoc
    C_t[5]  = R_Cdon
    C_t[6]  = R_Cdop
    C_t[7]  = R_Cam
    C_t[8]  = R_Cni
    C_t[9]  = R_Cph
    C_t[10] = R_Cox
    
    return C_t


# =================== Pelagic process rates calculation / =================== #
# =========================================================================== #


# =========================================================================== #
# ================ Bentho-Pelagic process rates calculation \ =============== #
@njit(cache=cache_option)
def pelagic_with_sediment_process_rates(C_t, CS1_t, CS2_t, T, V, H, I_a, salinity, f_day, kmc, smc):
    
    Cpy, Cpoc, Cpon, Cpop, Cdoc, Cdon, Cdop, Cam, Cni, Cph, Cox = C_t
    POC_1, PON_1, POP_1, DOC_1, DON_1, DOP_1, NH4_1, NO3_1, PO4_1 = CS1_t
    POC_2, PON_2, POP_2, DOC_2, DON_2, DOP_2, NH4_2, NO3_2, PO4_2 = CS2_t
    
    
    R_t = pelagic_process_rates(C_t, T, V, H, I_a, salinity, f_day, kmc)
    
    R_PHYC, R_POC, R_PON, R_POP, R_DOC, R_DON, R_DOP, R_NH4, R_NO3, R_PO4, R_O2 = R_t
    
# =============================================================================
#    Inorganic nutrients produced through mineralization or desorption are 
#    immediately released into the overlying water, as this approach 
#    does not account for pore water.
# =============================================================================

    # Cpoc processes
    R_POC = R_POC - smc[omega_01_POC]/H * (Cpoc - POC_1)
    # POC_1 processes
    R_POC_1 = (+ smc[omega_01_POC]/H * (Cpoc - POC_1)
              + (kmc[v_set_Cpoc]/H) * Cpoc
              + (kmc[v_set_Cpy]/H) * Cpy
              - smc[omega_12_POC1]/H * (POC_1 - POC_2)                                      # particle mixing
              - smc[w_burial_POC1] * POC_1                                                  # bruial
              - smc[k_dissol_POC1] * POC_1 * (POC_1/(POC_1+smc[K_HS_sed_POC_1_disl])) * H1  # decomposition
              ) / H1
    # POC_2 processes
    R_POC_2 = (+ smc[omega_12_POC1]/H * (POC_1 - POC_2)                                     # particle mixing
              + smc[w_burial_POC1] * (POC_1 - POC_2)                                        # bruial
              - smc[k_dissol_POC2] * POC_2 * (POC_2/(POC_2+smc[K_HS_sed_POC_2_disl])) * H2  # decomposition
              ) / H2
    
    # Cpon processes
    R_PON = R_PON - smc[omega_01_PON]/H * (Cpon - PON_1)
    # PON_1 processes
    R_PON_1 = (+ smc[omega_01_PON]/H * (Cpon - PON_1)
              + (kmc[v_set_Cpon]/H) * Cpon
              + (kmc[v_set_Cpy]/H) * Cpy * kmc[a_N_C]
              - smc[omega_12_PON1]/H * (PON_1 - PON_2)                                      # particle mixing
              - smc[w_burial_PON1] * PON_1                                                  # bruial
              - smc[k_dissol_PON1] * PON_1 * (PON_1/(PON_1+smc[K_HS_sed_PON_1_disl])) * H1  # decomposition
              ) / H1
    # PON_2 processes
    R_PON_2 = (+ smc[omega_12_PON1]/H * (PON_1 - PON_2)                                     # particle mixing
              + smc[w_burial_PON1] * (PON_1 - PON_2)                                        # bruial
              - smc[k_dissol_PON2] * PON_2 * (PON_2/(PON_2+smc[K_HS_sed_PON_2_disl])) * H2  # decomposition
              ) / H2
    
    # Cpop processes
    R_POP = R_POP - smc[omega_01_POP]/H * (Cpop - POP_1)
    # POP_1 processes
    R_POP_1 = (+ smc[omega_01_POP]/H * (Cpop - POP_1)
              + (kmc[v_set_Cpop]/H) * Cpop
              + (kmc[v_set_Cpy]/H) * Cpy * kmc[a_P_C]
              - smc[omega_12_POP1]/H * (POP_1 - POP_2)                                      # particle mixing
              - smc[w_burial_POP1] * POP_1                                                  # bruial
              - smc[k_dissol_POP1] * POP_1 * (POP_1/(POP_1+smc[K_HS_sed_POP_1_disl])) * H1  # decomposition
              ) / H1
    # POP_2 processes
    R_POP_2 = (+ smc[omega_12_POP1]/H * (POP_1 - POP_2)                                     # particle mixing
              + smc[w_burial_POP1] * (POP_1 - POP_2)                                        # bruial
              - smc[k_dissol_POP2] * POP_2 * (POP_2/(POP_2+smc[K_HS_sed_POP_2_disl])) * H2  # decomposition
              ) / H2

        
    # Mineralization of dissolved organic matters in aerobic layer
    def mineralization_by_dox_1_C():
        return smc[k_DOC_1_Miner_dox]* \
            Cox/(smc[K_HS_DOC_1_Minr_lim_dox]+Cox) *\
                (DOC_1/(DOC_1+smc[K_HS_DOC_1_Minr_dox])) * DOC_1
                
    def mineralization_by_no3_1_C():
        return smc[k_DOC_1_Miner_no3_1] * \
            (1 - Cox/(smc[K_HS_DOC_1_Minr_inhb_dox_1]+Cox)) * \
                NO3_1/(smc[K_HS_DOC_1_Minr_lim_no3_1]+NO3_1) * \
                    (DOC_1/(DOC_1+smc[K_HS_DOC_1_Minr_no3_1])) * DOC_1
                    
    def mineralization_by_dox_1_N():
        return smc[k_DON_1_Miner_dox]* \
            Cox/(smc[K_HS_DON_1_Minr_lim_dox]+Cox) *\
                (DON_1/(DON_1+smc[K_HS_DON_1_Minr_dox])) * DON_1
                
    def mineralization_by_no3_1_N():
        return smc[k_DON_1_Miner_no3_1] * \
            (1 - Cox/(smc[K_HS_DON_1_Minr_inhb_dox_1]+Cox)) * \
                NO3_1/(smc[K_HS_DON_1_Minr_lim_no3_1]+NO3_1) * \
                    (DON_1/(DON_1+smc[K_HS_DON_1_Minr_no3_1])) * DON_1
    
    def mineralization_by_dox_1_P():
        return smc[k_DOP_1_Miner_dox]* \
            Cox/(smc[K_HS_DOP_1_Minr_lim_dox]+Cox) *\
                (DOP_1/(DOP_1+smc[K_HS_DOP_1_Minr_dox])) * DOP_1
                
    def mineralization_by_no3_1_P():
        return smc[k_DOP_1_Miner_no3_1] * \
            (1 - Cox/(smc[K_HS_DOP_1_Minr_inhb_dox_1]+Cox)) * \
                NO3_1/(smc[K_HS_DOP_1_Minr_lim_no3_1]+NO3_1) * \
                    (DOP_1/(DOP_1+smc[K_HS_DOP_1_Minr_no3_1])) * DOP_1
    
    
    # Cdoc processes
    R_DOC = R_DOC - smc[Kl_01_DOC] * (Cdoc - DOC_1)
    # DOC_1 processes
    R_DOC_1_Mineralization = mineralization_by_dox_1_C() + mineralization_by_no3_1_C()
    R_DOC_1 = (+ smc[Kl_01_DOC] * (Cdoc - DOC_1)
               - smc[Kl_12_DOC1] * (DOC_1 - DOC_2)
               + smc[k_dissol_POC1] * POC_1 * (POC_1/(POC_1+smc[K_HS_sed_POC_1_disl])) * H1
               - R_DOC_1_Mineralization * H1
               ) / H1
    # DOC_2 processes
    R_DOC_2 = (+ smc[Kl_12_DOC1] * (DOC_1 - DOC_2)
               + smc[k_dissol_POC2] * POC_2 * (POC_2/(POC_2+smc[K_HS_sed_POC_2_disl])) * H2 
               ) / H2
    
    # Cdon processes
    R_DON = R_DON - smc[Kl_01_DON] * (Cdon - DON_1)
    # DON_1 processes
    R_DON_1_Mineralization = mineralization_by_dox_1_N() + mineralization_by_no3_1_N()
    R_DON_1 = (+ smc[Kl_01_DON] * (Cdon - DON_1)
               - smc[Kl_12_DON1] * (DON_1 - DON_2)
               + smc[k_dissol_PON1] * PON_1 * (PON_1/(PON_1+smc[K_HS_sed_PON_1_disl])) * H1
               - R_DON_1_Mineralization * H1
               ) / H1
    # DON_2 processes
    R_DON_2 = (+ smc[Kl_12_DON1] * (DON_1 - DON_2)
               + smc[k_dissol_PON2] * PON_2 * (PON_2/(PON_2+smc[K_HS_sed_PON_2_disl])) * H2
               ) / H2
    
    # Cdop processes
    R_DOP = R_DOP - smc[Kl_01_DOP] * (Cdop - DOP_1)
    # DOP_1 processes
    R_DOP_1_Mineralization = mineralization_by_dox_1_P() + mineralization_by_no3_1_P()
    R_DOP_1 = (+ smc[Kl_01_DOP] * (Cdop - DOP_1)
               - smc[Kl_12_DOP1] * (DOP_1 - DOP_2)
               + smc[k_dissol_POP1] * POP_1 * (POP_1/(POP_1+smc[K_HS_sed_POP_1_disl])) * H1
               - R_DOP_1_Mineralization * H1
               ) / H1
    # DOP_2 processes
    R_DOP_2 = (+ smc[Kl_12_DOP1] * (DOP_1 - DOP_2)
               + smc[k_dissol_POP2] * POP_2 * (POP_2/(POP_2+smc[K_HS_sed_POP_2_disl])) * H2
               ) / H2
    
    # Cam processes
    R_NH4 = R_NH4 - smc[Kl_01_NH4] * (Cam - NH4_1)

    # NH4_1 processes
    R_NH4_1 = (+ smc[Kl_01_NH4] * (Cam - NH4_1)
               - smc[Kl_12_NH4_1] * (NH4_1 - NH4_2)
               - smc[k_nitr_NH4_1] * NH4_1 * (NH4_1+smc[K_HS_sed_Nitr]/ (NH4_1)) * H1
               + R_DON_1_Mineralization * H1
               ) / H1
    # NH4_2 processes
    R_NH4_2 = (+ smc[Kl_12_NH4_1] * (NH4_1 - NH4_2)
               ) / H2
    
    # Cni processes
    R_NO3 = R_NO3 - smc[Kl_01_NO3] * (Cni - NO3_1)
    # NO3_1 processes
    R_Denitrification_1 = smc[k_denitr_NO3_1] * NO3_1 * (NO3_1 / (NO3_1+smc[K_HS_sed_deNitr]))
    R_NO3_1 = (+ smc[Kl_01_NO3] * (Cni - NO3_1)
               - smc[Kl_12_NO3] * (NO3_1 - NO3_2)
               + smc[k_nitr_NH4_1] * NH4_1 * (NH4_1+smc[K_HS_sed_Nitr]/ (NH4_1)) * H1
               - R_Denitrification_1 * H1
               ) / H1
    # NO3_2 processes
    R_NO3_2 = (+ smc[Kl_12_NO3] * (NO3_1 - NO3_2)
               - smc[k_denitr_NO3_2] * NO3_2 * (NO3_2 / (NO3_2+smc[K_HS_sed_deNitr])) * H2
               ) / H2
    
    # Cph processes
    R_PO4 = (+ R_PO4 
             - smc[Kl_01_PO4] * (Cph - PO4_1) 
             - smc[k_adsorp_PO4] * Cph * (Cph / (Cph + smc[K_HS_adsorp_PO4]))
             )
    # PO4_1 processes
    R_PO4_1 = (+ smc[Kl_01_PO4] * (Cph - PO4_1)
               - smc[Kl_12_PO4_1] * (PO4_1 - PO4_2)
               + R_DOP_1_Mineralization * H1
               #+ smc['k_adsorp_PO4'] * Cph * (Cph / (Cph + smc['K_HS_adsorp_PO4'])) * H1
               ) / H1
    # PO4_2 processes
    R_PO4_2 = (+ smc[Kl_12_PO4_1] * (PO4_1 - PO4_2)
               ) / H2


    CS1_t[0] = R_POC_1
    CS1_t[1] = R_PON_1
    CS1_t[2] = R_POP_1
    CS1_t[3] = R_DOC_1
    CS1_t[4] = R_DON_1
    CS1_t[5] = R_DOP_1
    CS1_t[6] = R_NH4_1
    CS1_t[7] = R_NO3_1
    CS1_t[8] = R_PO4_1

    CS2_t[0] = R_POC_2
    CS2_t[1] = R_PON_2
    CS2_t[2] = R_POP_2
    CS2_t[3] = R_DOC_2
    CS2_t[4] = R_DON_2
    CS2_t[5] = R_DOP_2
    CS2_t[6] = R_NH4_2
    CS2_t[7] = R_NO3_2
    CS2_t[8] = R_PO4_2
    
    C_t[0]   = R_PHYC
    C_t[1]   = R_POC
    C_t[2]   = R_PON
    C_t[3]   = R_POP
    C_t[4]   = R_DOC
    C_t[5]   = R_DON
    C_t[6]   = R_DOP
    C_t[7]   = R_NH4
    C_t[8]   = R_NO3
    C_t[9]   = R_PO4
    C_t[10]  = R_O2
    
    return C_t, CS1_t, CS2_t

# ================ Bentho-Pelagic process rates calculation / =============== #
# =========================================================================== #
