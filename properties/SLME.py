from math import pi
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps

class Efficiency():
    def __init__(
            self,
            solar_path="AM15G.csv"):
        self.c = constants.c
        self.h = constants.h
        self.q = constants.e
        self.k = constants.k
        data = pd.read_csv(solar_path)
        solar_data = np.array(data)
        solar_data[:,0] = solar_data[:,0] * 1E-9 #go from nm to m
        solar_data[:,1] = solar_data[:,1] / 1E-9 #go from /nm to /m
        new_Es = self.h * self.c / solar_data[:,0]
        dlam_dE = self.h * self.c / new_Es**2 #minus sign is expressed through reordering of axes
        sun_conv = np.copy(solar_data)
        sun_conv[:,0]= new_Es
        sun_conv[:,1]=solar_data[:,1] * dlam_dE / new_Es
        p_in=simps(sun_conv[::-1,0]*sun_conv[::-1,1],sun_conv[::-1,0])
        self.Es = sun_conv[::-1,0]
        self.phis = sun_conv[::-1,1]
        self.p_in = p_in

    def calculate(
        self,
        energies,
        alphas,
		absorptance = False,
		del_E = 0,
        T=300,
        d=500,
        JV_plot = False,
        display = False,
        return_PVJ = False
    ):
        energies = np.array(energies)
        alphas = np.array(alphas)
        Es = self.Es
        phis = self.phis
        if absorptance:
            As = alphas
        else:
            As = 1 - np.exp(-2 * alphas * d * 1E-7)
        Es_J = energies * self.q
        conv_As = np.interp(Es,Es_J,As)
        phi_BB = 2 * Es**2 /(self.h**3 * self.c**2 * (np.exp(Es /(self.k * T))-1))
        Jsc = self.q * simps(conv_As*phis,Es)
        f_r = np.exp(-1*del_E*self.q/(self.k*T))
        J_0 = self.q * simps(pi * conv_As * phi_BB,Es)/f_r
        def J(V):
            return Jsc - J_0 * (np.exp(self.q * V / (self.k * T)) - 1.0)
        def power(V):
            return J(V) * V
        test_voltage = 0
        voltage_step = 0.001
        while power(test_voltage + voltage_step) > power(test_voltage):
            test_voltage += voltage_step
        max_power = power(test_voltage)
        eta = max_power / self.p_in
        percentage_efficiency = eta*100
        if JV_plot:
            V = np.linspace(0, test_voltage+0.1, 200)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(V,J(V),label='J(V)')
            ax2.plot(V, power(V), color='tab:orange',label='P(V)')
            ax1.set_ylabel(r'J / A m$^{-2}$')
            ax1.set_xlabel('V / V')
            ax2.set_ylabel(r'Power / W m$^{-2}$')
            plt.savefig("JV_plot.svg")
        if display:
            print(f"Maximum predicted efficiency of {round(percentage_efficiency,2)}% corresponding to a power of {round(max_power,2)} W/m^2, short-circuit current density of {round(Jsc,2)} A/m^2 and voltage of {round(test_voltage,2)} V under AM1.5G illumination at {T} K for a {d} nm material")
        if return_PVJ:
            return max_power, test_voltage,Jsc, percentage_efficiency
        else:
            return percentage_efficiency
