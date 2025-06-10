#Correct code for 1+1 D phi4 system
#This code imposes a circular boundary
#All units are in MeV

import numpy as np
import matplotlib.pyplot as plt
import sys

#Lattice parameters
g=6
m2=-1#mass squared
#Size of the 2-D lattice(in units of number of points)
Nx=32
dx=197.3 # MeV^-1, or 1 fm 
T=10 # In MeV
Nt=32
dt=T/(Nt-1) # In some units, yet to be defined
Nsim=10000
eps=0.0001
#Initial random 2-d array
#lattice=np.zeros((Nt,Nx))#np.random.randn(Nt,Nx)
#lattice=np.ones((Nt,Nx))*(1)#np.random.randn(Nt,Nx)
lattice=np.random.randn(Nt,Nx)
dSdphi=np.zeros((Nt,Nx))#np.random.randn(Nt,Nx)
momentum=np.random.randn(Nt,Nx)
#momentum=np.zeros((Nt,Nx))
dphix=np.zeros((len(lattice),len(lattice[0])))
dphit=np.zeros((len(lattice),len(lattice[0])))
xcutoff=50 #For excluding near boundary regions
simcutoff=0 #For excluding burnin period
#twoptcor=np.zeros((Nsim-simcutoff,Nx-2*xcutoff)))

#Define hamiltonian
def hamiltonian(lattice, momentum):
	KE = np.sum(momentum**2)/2
	PE = action(lattice)
	if np.isinf(KE+PE):
		print(KE,PE)
		sys.exit()
	return KE+PE, PE
	  
#Define next and prev indices for all points on 2 d lattice 
indices=np.zeros((Nt,Nx,6),dtype=int)
for i in range(0,len(lattice)): # i goes over time
   for j in range(0,len(lattice[0])):  # j goes over spatial
      #Spatial derivative
      if(j == (Nx-1)):
        nextx=j
        prevx=j-1
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = j
      elif(j == 0):
        nextx=j+1
        prevx=0
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = j
      else:
        nextx=j+1
        prevx=j-1
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = j
      #Imaginary time derivative
      if(i == (Nt-1)):
        nextt=i
        prevt=i-1
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = i
      elif(i == 0):
        nextt=i+1
        prevt=0
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = i
      else:
        nextt=i+1
        prevt=i-1
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = i

#Matrix of 1/delta x and 1/delta t for calculation of del (action)/del phi(x,t)
def delSdelphi(lattice):
	dSdphi[:,:]=0
			        #t derivative contribution
	dSdphi[1:-1,:] += -dphit[indices[1:-1,:,5],indices[1:-1,:,0]]/(dt*2) \
		+ dphit[indices[1:-1,:,4],indices[1:-1,:,0]]/(dt) 
	
	dSdphi[-1,:] += dphit[indices[-1,:,5],indices[-1,:,0]]/(dt) \
		+ dphit[indices[-1,:,4],indices[-1,:,0]]/(dt*2) 

	dSdphi[0,:] += -dphit[indices[0,:,5],indices[0,:,0]]/(dt*2) \
		- dphit[indices[0,:,4],indices[0,:,0]]/(dt) 
		
	#x derivative contribution
	dSdphi[:,1:-1] += -dphix[indices[:,1:-1,3],indices[:,1:-1,2]]/(dx*2) \
		+ dphix[indices[:,1:-1,3],indices[:,1:-1,1]]/(dx) 
	
	dSdphi[:,-1] += dphix[indices[:,-1,3],indices[:,-1,2]]/(dx) \
		+ dphix[indices[:,-1,3],indices[:,-1,1]]/(dx*2) 

	dSdphi[:,0] +=-dphix[indices[:,0,3],indices[:,-0,2]]/(dx*2) \
		- dphix[indices[:,0,3],indices[:,0,1]]/(dx) 
	
	#Diagonal contribution
	dSdphi[:,:] += m2*lattice[:,:] + g*lattice[:,:]**3/6 
	dSdphi[:,:]=dSdphi[:,:]*dx*dt
	return

#Caculation of spatial and temporal derivatives for action
def dlattice(lattice):
  global dphix, dphit
  dphix[:,:]=(lattice[indices[:,:,3],indices[:,:,2]] - lattice[indices[:,:,3],indices[:,:,1]])/dx/(indices[:,:,2]-indices[:,:,1])	
  dphit[:,:]=(lattice[indices[:,:,5],indices[:,:,0]] - lattice[indices[:,:,4],indices[:,:,0]])/dt/(indices[:,:,5]-indices[:,:,4])	
  return  

def action(lattice):
	dlattice(lattice)
	integral_dphit2=np.sum(dphit**2) #Numerical integral over delphideltau term
	integral_dphix2=np.sum(dphix**2) #Numerical integral over delphidelx term
	integral_phi2 = np.sum(lattice**2) #Numerical integral over mass term
	integral_phi4 = np.sum(lattice**4) #Numerical integral over phi4 term
	action=(integral_dphit2/2+integral_dphix2/2+m2/2*integral_phi2+g/24*integral_phi4)
	action=action*dx*dt
	return action

def initialize_lattice(lattice):
	lattice=np.random.randn((Nt,Nx))
	return lattice

#Hamiltonian monte carlo and Metropolis-Hastings algorithm

def leapfrog(lattice,momentum):
	dlattice(lattice)
	delSdelphi(lattice)
	p_temp = momentum - eps/2*dSdphi
	lattice_temp = lattice + eps*p_temp
	dlattice(lattice_temp)
	delSdelphi(lattice_temp)
	p_temp = p_temp - eps/2*dSdphi
	return lattice_temp, p_temp

def hmcupdate(lattice,momentum):	
	#momentum=np.random.randn(Nt,Nx)
	lattice_temp, p_temp = lattice , momentum
	for k in range(0,100):
		lattice_temp, p_temp = leapfrog(lattice_temp,p_temp)
	return lattice_temp, p_temp

def twoptcor(lattice):
	temparr=lattice[0,xcutoff:-xcutoff]
	mean=np.mean(temparr)
	sig=np.std(temparr)
	corrarr=[np.sum((temparr[0:len(temparr)-i]-mean)*(temparr[i:]-mean)) for i in range(0,len(temparr))]
	corrarr=corrarr/sig**2
	return corrarr

meanphi=[]
#propagator=np.array(arrays, dtype=object)
#propagator=np.zeros((Nsim-simcutoff,Nx-2*xcutoff))
count_acceptance=0
#correlation='twopointfunc.txt'
#Monte carlo simulation
def hamiltonian_path(count_acceptance,lattice,momentum,meanphi):
	for i in range(0,Nsim):

		if count_acceptance/(i+1) < 0.4:
			sigmatemp=100
		else:
			sigmatemp=4
		momentum=3*np.random.randn(Nt,Nx)
		hamiltonian1, _= hamiltonian(lattice,momentum)
		lattice_temp,momentum_temp = hmcupdate(lattice,momentum)
		hamiltonian2, _= hamiltonian(lattice_temp,momentum_temp)
		prat = np.exp(-(hamiltonian2-hamiltonian1)/T)
		prat2 = ((hamiltonian2-hamiltonian1)/T)
		if hamiltonian2 <= hamiltonian1 :
			lattice=lattice_temp
			momentum=momentum_temp
			count_acceptance+=1
			print("Accepted1",prat2,count_acceptance/(i+1))
		else :
			if np.random.rand() < np.exp(-(hamiltonian2-hamiltonian1)/T):
				print("Rejected",prat2,count_acceptance/(i+1))
			else: 
				lattice=lattice_temp
				momentum=momentum_temp
				count_acceptance+=1
				print("Accepted2",prat2,count_acceptance/(i+1))
			#print(np.isnan(lattice_temp).any())
			#print(lattice_temp[:,:])
	#print("Test", hamiltonian1,"|| ",hamiltonian2,"|| ",prat,"||",lattice.sum())
		meanphi.append(np.mean(lattice))
	#meanphi.append(lattice.sum())
	#with open(correlation,'a') as f:
	#	np.savetxt(f,lattice*lattice[0,0])
		print(np.mean(lattice_temp)," ",i)
#	if i>(simcutoff):
#		propagator[i-simcutoff]=twoptcor(lattice)
	return meanphi

stable_mean=[]

for j in range(0,100):
	meanphi=hamiltonian_path(count_acceptance,lattice,momentum,meanphi)
	stable_mean.append(np.mean(meanphi[-500:]))
	meanphi=[]


"""
#Auto correlation calculation
mean_propagator = np.mean(propagator,axis=0)
std_propagator = np.std(propagator,axis=0)
"""
plt.figure(figsize=(8,7))
plt.plot(meanphi,label=r"\phi^4 simulation")
plt.xlabel('Monte Carlo Time')
plt.ylabel(r'$\langle \Phi \rangle$')
plt.show()
"""with open('T_80MeV.txt','w') as file:
	for value in mean_propagator:
		file.write(f"{value}\n")

plt.figure(figsize=(8,7))
plt.plot(mean_propagator-std_propagator,label=r"\phi^4 simulation")
plt.plot(mean_propagator+std_propagator,label=r"\phi^4 simulation")
plt.plot(mean_propagator,label=r"\phi^4 simulation")
plt.xlabel('Monte Carlo Time')
plt.ylabel(r'$\langle \Phi (0) \Phi(X) \rangle$')
plt.show()
"""

with open('hmc_T_10MeV_32x32.txt','a') as file:
	for value in stable_mean:
		file.write(f"{value}\n")
