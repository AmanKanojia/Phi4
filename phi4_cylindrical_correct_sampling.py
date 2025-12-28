#Correct code for 1+1 D phi4 system
#This code imposes a circular boundary
#All units are in MeV

import numpy as np
import matplotlib.pyplot as plt
#Lattice parameters
g=1
m2=-0.4#-1.0#mass squared
#Size of the 2-D lattice(in units of number of points)
Nx=4
dx=1#0.2/197.3 # MeV^-1, or 1 fm 
T=1 # In MeV
Nt=4
dt=1#1./T/(Nt-1) # In some units, yet to be defined
Nsim=10000
#Initial random 2-d array
#lattice=np.zeros((Nt,Nx))#np.random.randn(Nt,Nx)
#lattice=np.random.randn(Nt,Nx)
lattice=np.ones((Nt,Nx))*(1)#np.random.randn(Nt,Nx)
dphix=np.zeros((len(lattice),len(lattice[0])))
dphit=np.zeros((len(lattice),len(lattice[0])))
#****************Auto-Correlation Codes******************
def acf_fft(x, max_lag=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    x -= np.mean(x)

    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.fft(x, n=nfft)
    acf = np.fft.ifft(f * np.conjugate(f)).real
    acf = acf[:n]

    acf /= np.arange(n, 0, -1)
    acf /= acf[0]

    if max_lag is not None:
        acf = acf[:max_lag + 1]

    return acf

def plot_acf(action_list, series_label, max_lag=10):
    rho = acf_fft(action_list, max_lag=max_lag)

    lags = np.arange(len(rho))
    noise = 2 / np.sqrt(len(action_list))

    #plt.figure()
    plt.plot(lags, rho,label=series_label)
    plt.axhline(+noise, linestyle='--')
    plt.axhline(-noise, linestyle='--')
    plt.axhline(0, linestyle=':')

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(r"Autocorrelation Function (ACF) for 2D $\Phi^4$ Action")
    plt.legend()
    #plt.show()

#*************************************************************
#Define next and prev indices for all points on 2 d lattice 
indices=np.zeros((Nt,Nx,6),dtype=int)
for i in range(0,len(lattice)): # i goes over time
   for j in range(0,len(lattice[0])):  # j goes over spatial
      #Spatial derivative
      if(j == (Nx-1)):
        nextx=0#j
        prevx=j#j-1
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = i
      elif(j == 0):
        nextx=j+1
        prevx=0#Nx-1#0
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = i
      else:
        nextx=j+1
        prevx=j#j-1
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = i
      #Imaginary time derivative
      if(i == (Nt-1)):
        nextt=0#i
        prevt=i#i-1
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = j
      elif(i == 0):
        nextt=i+1
        prevt=0#Nt-1#0
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = j
      else:
        nextt=i+1
        prevt=i#i-1
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = j

#Caculation of spatial and temporal derivatives for action
def dlattice(lattice):
  global dphix, dphit
  dphix[:,:]=(lattice[indices[:,:,0],indices[:,:,2]] - lattice[indices[:,:,0],indices[:,:,1]])/dx#/(indices[:,:,2]-indices[:,:,1])	
  dphit[:,:]=(lattice[indices[:,:,5],indices[:,:,3]] - lattice[indices[:,:,4],indices[:,:,3]])/dt#/(indices[:,:,5]-indices[:,:,4])	
  return  

def action(lattice):
	dlattice(lattice)
	integral_dphit2=np.sum(dphit**2) #Numerical integral over delphideltau term
	integral_dphix2=np.sum(dphix**2) #Numerical integral over delphidelx term
	integral_phi2 = np.sum(lattice**2) #Numerical integral over mass term
	integral_phi4 = np.sum(lattice**4) #Numerical integral over phi4 term
	action=(integral_dphit2/2+integral_dphix2/2+m2/2*integral_phi2+g**4/4*integral_phi4)
	action=action*dx*dt
	return action

def initialize_lattice(lattice):
	lattice=np.random.randn((Nt,Nx))
	return lattice

#Metropolis-Hastings algorithm

def mcupdate(lattice,i,j):
	global act	
	temp=lattice[i,j]	
	action1=action(lattice)
	#print(lattice)
	#print("A1",action1,lattice[i,j])
	lattice[i,j] = lattice[i,j] + np.random.randn()
	action2=action(lattice)
	#print("A2",action2,lattice[i,j])
	#exit()
	if action2<action1 :
		act+=1
		return 
	else :
		if np.random.rand()<np.exp(-(action2-action1)/T):
			act+=1
			return
		else:
			lattice[i,j]=temp
			return

chain_mean=[]
action_acf=[]
action_acf_thinning=[]
def mhchain(num):
	temp_chain_mean=[]
	action_acf_chain=[]
	action_acf_chain_thinning=[]
	for i in range(0,Nsim):
		for j in range(0,Nt):
			#print(j)
			for k in range(0,Nx):
				mcupdate(lattice,j,k)
		meanphi.append(lattice.sum()/Nt/Nx)
		print(i," ",num," ",meanphi[i]," ",act/(i+1)/Nt/Nx)
		if i>5000 :
			if i%10==0:
			#chain_mean.append(abs(np.mean(meanphi[-500:])))
				#temp_chain_mean.append(abs(np.mean(lattice)))	
				action_acf_chain_thinning.append(action(lattice))
		if (Nsim-i) <=500 :	
			temp_chain_mean.append(abs(np.mean(lattice)))	
			action_acf_chain.append(action(lattice))
	action_acf.append(action_acf_chain)
	action_acf_thinning.append(action_acf_chain_thinning)
	chain_mean.append(abs(np.mean(temp_chain_mean)))
np.random.seed(12345)
for i in range(0,201):
	act = 0
	meanphi=[]
	lattice=np.random.randn(Nt,Nx)*0.1
	#print(lattice)
	#print(action(lattice))
	#exit()
	mhchain(i)

	
#plt.figure(figsize=(8,7))
#plt.plot(meanphi,label=r"\phi^4 simulation")
#plt.show()
#plot_acf(action_acf[0],'Without Thinning')
#plot_acf(action_acf_thinning[0],'With Thinning')
plt.hist(chain_mean,bins='auto',edgecolor='black')
plt.title('Sampling without thinning')
plt.xlabel(r'$\Phi$')
plt.ylabel('Count')
plt.show()
with open('T_1by10_16x16MeV.txt','w') as file:
	for value in chain_mean:
		file.write(f"{value}\n")

