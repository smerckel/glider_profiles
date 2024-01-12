import numpy as np
class GeneralFilter2nd(object):
    ''' General discrete filter of the transfer function
                c0 + c1*s + c2 *s**2
        H(s) = ----------------------
                1 + d1*s + d2*s**2
        
    '''

    def __init__(self,c,d):
        self.c=c
        self.d=d

    def calculate_coefs(self,DT):
        d1,d2=self.d
        c0,c1,c2=self.c
        fn=2./DT
        fn2=fn**2
        denom=1+d1*fn+d2*fn2
        a0=(c0+c1*fn+c2*fn2)/denom
        a1=2.*(c0-c2*fn2)/denom
        a2=(c0-c1*fn+c2*fn2)/denom
        b1=2.*(1-d2*fn2)/denom
        b2=(1.-d1*fn+d2*fn2)/denom
        a=[a0,a1,a2]
        b=[1.,b1,b2]
        return a,b

    def filter(self,t,x):
        DT=np.median(np.diff(t))
        a,b=self.calculate_coefs(DT)
        return self.__filter(x,a,b)

    def __filter(self,x,a,b):
        a0,a1,a2=a
        b0,b1,b2=b
        y=np.zeros(x.shape,float)
        for i in range(2,len(x)):
            y[i]=a0*x[i]+a1*x[i-1]+a2*x[i-2]-b1*y[i-1]-b2*y[i-2]
        return y


    
class LeadLag(object):
    def __init__(self,sT,sC,verbose=False):
        self.sT=sT
        self.sC=sC
        self._verbose=verbose

    def calculate_coefs(self,DT):
        a=2./self.sT/DT
        b=2./self.sC/DT
        a0=(1.+a)/(1.+b)
        a1=(1.-a)/(1.+b)
        b1=(1.-b)/(1.+b)
        return a0,a1,b1

    def filter(self,t,x):
        DT=np.median(np.diff(t))
        a0,a1,b1=self.calculate_coefs(DT)
        if self._verbose:
            print(a0,a1,b1)
        return self.__filter(x,a0,a1,b1)
    
    def filter_direct(self,x,a,b):
        return self.__filter(x,a,-a,b)

    def __filter(self,x,a0,a1,b1):
        y=np.zeros(x.shape,float)
        for i in range(1,len(x)):
            y[i]=a0*x[i]+a1*x[i-1]-b1*y[i-1]
        return y


class GeneralFilter(LeadLag):
    ''' General discrete filter of the transfer function
                c1 + c2*s
        H(s) = -----------
                d1 + d2*s
        
        d1+d2*2/Dt <> 0.
    '''
    def __init__(self,c1,c2,d1,d2):
        LeadLag.__init__(self,1,1)
        self.set_parameters(c1,c2,d1,d2)
        
    def set_parameters(self,c1,c2,d1,d2):
        self.c1=c1
        self.c2=c2
        self.d1=d1
        self.d2=d2

    def calculate_coefs(self,DT):
        c1=self.c1
        c2=self.c2
        d1=self.d1
        d2=self.d2
        denom=(d1+d2*2./DT)
        a0=(c1+c2*2./DT)/denom
        a1=(c1-c2*2./DT)/denom
        b1=(d1-d2*2./DT)/denom
        return a0,a1,b1

class LagFilter(GeneralFilter):
    def __init__(self,gain,delay):
        GeneralFilter.__init__(self,gain,0.,1.0,delay)

    
class ThermalLagFilter(LeadLag):
    def __init__(self,gamma,alpha,beta):
        LeadLag.__init__(self,1,1)
        self.set_parameters(gamma,alpha,beta)

    def set_parameters(self,gamma,alpha,beta):
        self.c1=0.
        self.c2=gamma*alpha/beta
        self.d1=1.
        self.d2=1./beta
        
    def calculate_coefs(self,DT):
        c1=self.c1
        c2=self.c2
        d1=self.d1
        d2=self.d2
        denom=(d1+d2*2./DT)
        a0=(c1+c2*2./DT)/denom
        a1=(c1-c2*2./DT)/denom
        b1=(d1-d2*2./DT)/denom
        return a0,a1,b1
    
    def calculate_alpha_beta(self,DT,gamma):
        fn=2./DT
        a0,a1,b1=self.calculate_coefs(DT)
        print("a0,b1:",a0,b1)
        beta=fn*(1+b1)/(1-b1)
        alpha=2*a0*beta/fn/(1+b1)/gamma
        return alpha,beta

if __name__=="__main__":
    import random
    from pylab import *


    if 1:
        t=arange(1000)
        x=np.zeros(1000)
        x[500:]=1.
        f=GeneralFilter2nd([1.,0.,0.],[1.,5.,0.])
        f2=LagFilter(1.,5.)
        y=f.filter(t,x)
        y2=f2.filter(t,x)
    Q
    # test program to investigate the effect of a lead lag filter. See
    # alse the note on Ztransforms (Ztransform.pdf)

    if 0:
        # example for use in the Ztransform.pdf note.
        T=100.
        dt=1/64. # picklo and Lueck
        dt=2.    # glider
        nt=T/dt
        t=np.linspace(0.,T,nt)
        x=np.sin(t/10.)
        sT=7.5
        sC=4.35
        LL=LeadLag(sT,sC)

        plot(t,x,'r')
        y=LL.filter(t,x)
        plot(t,y,'g')
        dt=(1-sC/sT)/sC
        print("Signal shift of dt=%f"%(dt))
        plot(t-dt,y,'b')
        draw()

    if 0:
        # setting sT large (instantaneous response)
        T=100.
        dt=1/64.
        nt=T/dt
        t=np.linspace(0.,T,nt)
        x=np.sin(t/10.)
        sT=7.5*1e9
        sC=4.35
        LL=LeadLag(sT,sC)

        plot(t,x,'r')
        y=LL.filter(t,x)
        plot(t,y,'g')
        dt=(1-sC/sT)/sC
        print("Signal shift of dt=%f"%(dt))
        plot(t-dt,y,'b')
        draw()
        
    if 0:
        # adding noise to signal
        T=100.
        dt=1/64.
        #dt=2.
        nt=T/dt
        t=np.linspace(0.,T,nt)
        x=np.sin(t/10.)
        x=np.zeros(t.shape,float)
        x[nt/2:]=1.
        noise=np.array([random()*0.01 for i in x])
        x+=noise
        sT=7.5
        sC=4.35
        sT=2
        sC=1.
        LL=LeadLag(sT,sC)

        plot(t,x,'r')
        y=LL.filter(t,x)
        plot(t,y,'g')
        dt=(1-sC/sT)/sC
        print("Signal shift of dt=%f"%(dt))
        plot(t-dt,y,'b')
        xlim(49,56)
        draw()

    if 0:
        # a bode plot
        omega=logspace(0,3,100)
        I=complex(0,1)
        def H(s):
            return 1./(2.*s+100)
        h=H(omega*I)
        a=h.real
        b=h.imag
        A=sqrt(a**2+b**2)
        tanphi=b/a
        phi=arctan(tanphi)*180./pi
        subplot(211)
        semilogx(omega,20*log10(A),'k')
        semilogx(omega,omega*0-43,'k--')
        semilogx(omega*0+50,20*log10(A),'k--')
        xlabel(r'$\omega$ rad(/s)')
        ylabel('$A^2$ (dB)')
        subplot(212)
        semilogx(omega,phi,'k')
        semilogx(omega,omega*0-45,'k--') 
        semilogx(omega*0+50,phi,'k--')
        xlabel(r'$\omega$ (rad/s)')
        ylabel('$\phi$ (deg)')

    if 1:
        # a bode plot for the lead lag function
        omega=logspace(-2,1,100)
        I=complex(0,1)
        sT=7.5
        sC=4.35
        def H(s):
            return (1.+s/sT)/(1.+s/sC)
        h=H(omega*I)
        a=h.real
        b=h.imag
        A=sqrt(a**2+b**2)
        tanphi=b/a
        phi=arctan(tanphi)*180./pi
        subplot(311)
        semilogx(omega,20*log10(A),'k')
        #semilogx(omega,omega*0-43,'k--')
        #semilogx(omega*0+50,20*log10(A),'k--')
        xlabel(r'$\omega$ rad(/s)')
        ylabel('$A^2$ (dB)')
        subplot(312)
        semilogx(omega,phi,'k')
        #semilogx(omega,omega*0-45,'k--') 
        #semilogx(omega*0+50,phi,'k--')
        xlabel(r'$\omega$ (rad/s)')
        ylabel('$\phi$ (deg)')
        subplot(313)
        semilogx(omega,phi*pi/180./omega,'k')
        #semilogx(omega,omega*0-45,'k--') 
        #semilogx(omega*0+50,phi,'k--')
        xlabel(r'$\omega$ (rad/s)')
        ylabel('$\Delta t$ (s)')

        
        
