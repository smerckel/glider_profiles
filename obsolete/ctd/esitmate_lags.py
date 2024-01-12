from scipy.optimize import fmin
import ndf
import numpy as np
import pylab as pl


class Estimator(object):
    def __init__(self):
        self.bins=np.arange(-5,5,0.2)
        self.range=(self.bins[1:]+self.bins[:-1])*0.5

    # bell-shape function
    @classmethod
    def gauss_curve(cls,x,mu,sigma):
        return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)

    def cost_fun(self,p,x,y):
        mu,sigma,s=p
        return ((y-self.gauss_curve(x,mu,sigma)*s)**2).sum()


    def get_mu_sigma(self,tau):
        # first make the histogram
        y=np.histogram(tau[1],self.bins,normed=True)[0]
        x=self.range
        mu,sigma,s=fmin(self.cost_fun,(1.2,1,0.9),args=(x,y),disp=False)
        return mu,sigma,s

    def make_histogram(self,tau,ax,**kwds):
        ax.hist(tau[1],self.bins,normed=True,alpha=0.5,**kwds)
        mu,sigma,s=self.get_mu_sigma(tau)
        ax.plot(self.range,Estimator.gauss_curve(self.range,mu,sigma)*s,'k--')
        ax.set_xlim(-5,5)
        ax.set_ylim(0,0.7)

estimator=Estimator()

data=ndf.NDF("short_time_mismatch_results.ndf")
fig,ax=pl.subplots(2,2,sharey=True,sharex=True)

print("CA14",estimator.get_mu_sigma(data['tau_conductivity_amadeus']))
print("TA14",estimator.get_mu_sigma(data['tau_temperature_amadeus']))
print("CS14",estimator.get_mu_sigma(data['tau_conductivity_sebastian']))
print("TS14",estimator.get_mu_sigma(data['tau_temperature_sebastian']))


estimator.make_histogram(data['tau_conductivity_amadeus'],ax[0,0],label='Conductivity Amadeus 2014',color='r')
estimator.make_histogram(data['tau_temperature_amadeus'],ax[0,1],label='Temperature Amadeus 2014',color='r')
estimator.make_histogram(data['tau_conductivity_sebastian'],ax[1,0],label='Conductivity Sebastian 2014',color='r')
estimator.make_histogram(data['tau_temperature_sebastian'],ax[1,1],label='Temperature Sebastian 2014',color='r')


data=ndf.NDF("short_time_mismatch_results_2013.ndf")


print("CA13",estimator.get_mu_sigma(data['tau_conductivity_amadeus']))
print("TA13",estimator.get_mu_sigma(data['tau_temperature_amadeus']))
print("CS13",estimator.get_mu_sigma(data['tau_conductivity_sebastian']))
print("TS13",estimator.get_mu_sigma(data['tau_temperature_sebastian']))

estimator.make_histogram(data['tau_conductivity_amadeus'],ax[0,0],label='Conductivity Amadeus 2013',color='b')
estimator.make_histogram(data['tau_temperature_amadeus'],ax[0,1],label='Temperature Amadeus 2013',color='b')
estimator.make_histogram(data['tau_conductivity_sebastian'],ax[1,0],label='Conductivity Sebastian 2014',color='b')
estimator.make_histogram(data['tau_temperature_sebastian'],ax[1,1],label='Temperature Sebastian 2013',color='b')

for i in range(2):
    ax[1,i].set_xlabel(r'$\tau$ (s)')
    ax[i,0].set_ylabel(r'Rel. freq. (-)')
    for j in range(2):
        ax[i,j].legend(fontsize='small')
        
