#Correct code for 1+1 D phi4 system
#This code imposes a circular boundary
#All units are in MeV

import numpy as np
import matplotlib.pyplot as plt
#Lattice parameters
g=6
m2=-1#mass squared
#Size of the 2-D lattice(in units of number of points)
Nx=64
dx=197.3 # MeV^-1, or 1 fm 
T=10 # In MeV
Nt=32
dt=T/(Nt-1) # In some units, yet to be defined
Nsim=4000
#Initial random 2-d array
#lattice=np.zeros((Nt,Nx))#np.random.randn(Nt,Nx)
lattice=np.random.randn(Nt,Nx)
#lattice=np.ones((Nt,Nx))*(-.01)#np.random.randn(Nt,Nx)
dphix=np.zeros((len(lattice),len(lattice[0])))
dphit=np.zeros((len(lattice),len(lattice[0])))

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
        indices[i,j,0] = i
      elif(j == 0):
        nextx=j+1
        prevx=0
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = i
      else:
        nextx=j+1
        prevx=j-1
        indices[i,j,2] = nextx
        indices[i,j,1] = prevx
        indices[i,j,0] = i
      #Imaginary time derivative
      if(i == (Nt-1)):
        nextt=i
        prevt=i-1
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = j
      elif(i == 0):
        nextt=i+1
        prevt=0
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = j
      else:
        nextt=i+1
        prevt=i-1
        indices[i,j,5] = nextt
        indices[i,j,4] = prevt
        indices[i,j,3] = j

#Caculation of spatial and temporal derivatives for action
def dlattice(lattice):
  global dphix, dphit
  dphix[:,:]=(lattice[indices[:,:,0],indices[:,:,2]] - lattice[indices[:,:,0],indices[:,:,1]])/dx/(indices[:,:,2]-indices[:,:,1])	
  dphit[:,:]=(lattice[indices[:,:,5],indices[:,:,3]] - lattice[indices[:,:,4],indices[:,:,3]])/dt/(indices[:,:,5]-indices[:,:,4])	
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

#Metropolis-Hastings algorithm

def mcupdate(lattice,i,j):
	global act	
	temp=lattice[i,j]	
	action1=action(lattice)
	lattice[i,j] = lattice[i,j] + np.random.randn()*0.4
	action2=action(lattice)
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
def mhchain(num):
	for i in range(0,Nsim):
		for j in range(0,Nt):
			#print(j)
			for k in range(0,Nx):
				mcupdate(lattice,j,k)
		meanphi.append(lattice.sum()/Nt/Nx)
		print(i," ",num," ",meanphi[i]," ",act/(i+1)/Nt/Nx)
	chain_mean.append(abs(np.mean(meanphi[-500:]))/Nt/Nx)

for i in range(0,101):
	act = 0
	meanphi=[]
	lattice=np.random.randn(Nt,Nx)
	mhchain(i)

	
plt.figure(figsize=(8,7))
plt.plot(meanphi,label=r"\phi^4 simulation")
plt.show()

plt.hist(chain_mean,bins='auto',edgecolor='black')
plt.show()
with open('T_10_3MeV.txt','w') as file:
	for value in chain_mean:
		file.write(f"{value}\n")

