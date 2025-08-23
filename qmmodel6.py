#phi references are removed here
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import broyden1
# ==== PARAMETERS ====
N_rho = 60             # Number of grid points in phi
vev=0.092
rho_min, rho_max = 0.0,0.1**2/2*3
dk = -0.001            # Step in RG scale (negative = toward IR)
k_init = 1.2            # Initial RG scale
k_final = 0.4          # Final RG scale
tol = 1e-12             # Newton-Raphson tolerance
max_iter = 400           # Max iterations in Newton-Raphson
T=40/1000 #Temperature, in MeV
mu=0
g=4.6
Lambda=1
# ==== GRID ====
rho_grid = np.linspace(rho_min, rho_max, N_rho)
#dphi = phi_grid[1] - phi_grid[0]
rho=rho_grid
# ==== INITIAL CONDITION: U_k=1/2 m^2 phi^2 + lambda/4 phi^4 ====
a1L = 0.582**2
c = 0.138**2*0.093
a2L = 35.2
lam = 24#a2L/2
sig0L = c/a1L
m2 = -0.0#a1L - lam*sig0L**2
U0 = m2*rho_grid +  lam * rho_grid**2

def KD(i,j,N,M):
	return np.eye(N,M,k=i)

def nB(E):
    return 1./(np.exp(E/T)-1)

def first_derivative(U):
        dU=np.zeros_like(U)
	#dU[0] = (U[1]-U[0])/(phi_grid[1]-phi_grid[0])*1./()
	#dU[1:-1] = (U[2:]-U[1:-1])/(rho[2:]-rho[1:-1])#(U[2:] - U[0:-2])/(rho[2:]-rho[0:-2])
	#dU[1:-1] = (U[2:] - U[0:-2])/(rho[2:]-rho[0:-2])
	#dU[-1] = (U[-1]-U[-2])/(rho[-1]-rho[-2])
#First point        
        delrho=rho[1] - rho[0]
        #h2=rho[2] - rho[1]
        #dU[0] = (-(2*h1 + h2)/(h1*(h1 + h2))*U[0]
        #    + (h1 + h2)/(h1*h2)*U[1]
        #    - (h1)/(h2*(h1 + h2))*U[2])
        #dU[0] = (U[1]-U[0])/(rho[1]-rho[0])
        dU[0] = (-3*U[0]+4*U[1]-U[2])/2/(rho[1]-rho[0])
        #dU[0]=(-11*U[0] + 18*U[1] - 9*U[2] + 2*U[3]) / (6*delrho)
        #dU[1]=(-11*U[1] + 18*U[2] - 9*U[3] + 2*U[4]) / (6*delrho)
        #dU[0]=(-25*U[0] + 48*U[1] - 36*U[2] + 16*U[3] -3*U[4]) / (12*delrho)
        #dU[1]=(-25*U[1] + 48*U[2] - 36*U[3] + 12*U[4] -3*U[5]) / (12*delrho)
#Last point        
        #h1=rho[-1] - rho[-2]
        #h2=rho[-2] - rho[-3]
        #dU[-1] = ((2*h1 + h2)/(h1*(h1 + h2))*U[-1]
        #    - (h1 + h2)/(h1*h2)*U[-2]
        #    + (h1)/(h2*(h1 + h2))*U[-3])
        #dU[-1] = (U[-1]-U[-2])/(rho[-1]-rho[-2])
        dU[-1] = (3*U[-1]-4*U[-2]+U[-3])/2/(rho[-1]-rho[-2])
        
        #dU[-1]=(11*U[-1] - 18*U[-2] + 9*U[-3] - 2*U[-4]) / (6*delrho)
        #dU[-2]=(11*U[-2] - 18*U[-3] + 9*U[-4] - 2*U[-5]) / (6*delrho)

        #dU[-1]=(25*U[-1] - 48*U[-2] + 36*U[-3] - 16*U[-4] + 3*U[-5]) / (12*delrho)
        #dU[-2]=(25*U[-2] - 48*U[-3] + 36*U[-4] - 16*U[-5]+3*U[-6]) / (12*delrho)
#Middle points
        #dU[1:-1] = (U[2:]-U[1:-1])/(rho[2:]-rho[1:-1])#(U[2:] - U[0:-2])/(rho[2:]-rho[0:-2])
        dU[1:-1] = (U[2:] - U[0:-2])/(rho[2:]-rho[0:-2])
        #dU[2:-2] = (-U[4:] + 8*U[3:-1] - 8*U[1:-3] + U[0:-4]) / (12*delrho)
        #h_minus = rho[1:-1] - rho[0:-2]
        #h_plus  = rho[2:] - rho[1:-1]
        #dU[1:-1] = (-h_plus / (h_minus*(h_minus+h_plus))) * U[0:-2] \
        # + ((h_plus - h_minus) / (h_minus*h_plus)) * U[1:-1] \
        # + (h_minus / (h_plus*(h_minus+h_plus))) * U[2:]
        return dU

# ==== FINITE DIFFERENCE SECOND DERIVATIVE ====
def second_derivative(U):
    U_dd = np.zeros_like(U)
    delrho = rho[1] - rho[0]
    U_dd[0] = (2*U[0]-5*U[1] + 4*U[2] - U[3])/delrho**2
    U_dd[-1] = (2*U[-1]-5*U[-2] + 4*U[-3] - U[-4])/delrho**2
    U_dd[1:-1] = (U[0:-2]-2*U[1:-1] + U[2:])/delrho**2
    return U_dd

# ==== RHS of Flow Equation ====
def frg_rhs(k, U, U_prev, U_prev2, U_prev3):
    
    #Compute the RHS of the flow equation for the quark-meson model at finite T and mu.
    
    #Parameters:
    #- k: RG scale
    #- rho: chiral invariant rho = (σ² + π²)/2
    #- U_prime: dU_k/dρ
    #- U_double_prime: d²U_k/dρ²
    #- T: temperature
    #- mu: quark chemical potential
    #- g: Yukawa coupling
    
    #Returns:
    #- RHS of ∂_k U_k(ρ)
    
    Nc = 3
    Nf = 2

    U_prime = 	first_derivative(U)
    U_double_prime = first_derivative(U)

    # Meson energies
    E_pi = (k**2 + U_prime) #np.sqrt(np.maximum((k**2 + U_prime),1e-12))
    E_sigma = (k**2 + U_prime + 2*rho*U_double_prime)#np.sqrt(np.maximum(k**2 + U_prime + 2 * rho * U_double_prime,1e-12))

    if np.any(np.isnan(E_sigma)):
     plt.plot(phi_grid,U)
     plt.show()
     print("Sigma error",E_sigma)
     print(U)
     print(U_prime)
     print(2*rho*U_double_prime)
     print(rho[0:4])
     print(rho[-3:])
     print(U[1],U[0])
     print((U[1]-U[0])/(rho[1]-rho[0]))
     exit()

    # Quark energy
    E_q = np.sqrt(k**2 + 2 * g**2 * rho)

    # Thermal distribution functions
    def nB(E): return 1.0 / (np.exp(E / T) - 1.0) if T > 0 else 0.0
    def nF(E): return 1.0 / (np.exp(E / T) + 1.0) if T > 0 else 0.0

    # Evaluate thermal parts
#    boson_term = (3 / E_pi) * (1 + 2 * nB(E_pi)) + (1 / E_sigma) * (1 + 2 * nB(E_sigma))
#    fermion_term = (4 * Nc * Nf / E_q) * (1 - nF(E_q - mu) - nF(E_q + mu))
    boson_term = (3 / E_pi) * (1) + (1 / E_sigma) * (1)
    fermion_term = (4 * Nc * Nf / E_q) * (1)

    # Full RHS
    if k < (k_init+dk):
     rhs = 11*U-18*U_prev+9*U_prev2-2*U_prev3-6*dk*(k**4 / (12 * np.pi**2)) * (boson_term - fermion_term)
    elif k == (k_init+dk):
     rhs = 3*U-4*U_prev+U_prev2-2*dk*(k**4 / (12 * np.pi**2)) * (boson_term - fermion_term)
    else:
     rhs = U-U_prev-dk*(k**4 / (12 * np.pi**2)) * (boson_term - fermion_term)
    #rhs = (k**4 / (12 * np.pi**2)) * (boson_term - 0*fermion_term)
    return rhs


# ==== JACOBIAN (Finite Difference Placeholder) ====
#This jacobian is a NxN matrix. First index represents which differential equation is being talked of, second one is the U(phi_j), which is the one with respect to which the differential equation is being differentiated after finite difference.

def jacobian_func(U,U_prev, U_prev2, U_prev3, rho, k, epsilon):	
    pi=22./7
    N = len(U)
    J = np.zeros((N, N))
    f0 = frg_rhs(k, U, U_prev, U_prev2, U_prev3)   # f(U)
    J2=np.zeros((N,N)) 
    for j in range(N):
        dU = np.zeros(N)
        if j == 0 :
         dU[j] = epsilon
        else :
         dU[j] = epsilon#*U[j]
        U_perturbed = U + dU
        f1 = frg_rhs(k,U_perturbed, U_prev, U_prev2, U_prev3) 
        U_perturbed = U - dU
        f2 = frg_rhs(k,U_perturbed, U_prev, U_prev2, U_prev3) 
        J2[:, j] = (f1 - f2)/dU[j]/2
#        if f1.all() == f0.all() :
#         print(j,"True",np.linalg.norm(U_perturbed-U)/epsilon) 
#         print(f1-f0)
#         exit()
#    print(np.linalg.det(J2))
#    print(J2[0,:])
#    print(U)
#    exit()	
    return J2

#print(jacobian_func(U0,phi_grid,k_init))


# ==== NEWTON-RAPHSON SOLVER ====
def newton_raphson(U_old,U_old2, U_old3,rho, k, dk,U_new):
    epsilon=1e-10
    #return broyden1(lambda U_new: frg_rhs(k,U_new,U_old,U_old2),U_new,f_tol=1e-8)

    for iteration in range(max_iter):
        F = frg_rhs(k, U_new, U_old,U_old2,U_old3) 
        J = jacobian_func(U_new, U_old, U_old2, U_old3, rho, k,epsilon)
        try:
            delta_U = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            #epsilon=epsilon/10
            #delta_U = 0*U_ne
            print(J[:,0])
            print(U_old)
            U_prime = 	first_derivative(U_old)
            U_double_prime = first_derivative(U_prime)
            print(U_double_prime,rho[98:102])
            raise RuntimeError("Jacobian is singular.")
            exit()
        U_new += delta_U
        #U_old2=
        F = frg_rhs(k, U_new, U_old,U_old2, U_old3) 
        norm = np.linalg.norm(F)
        print(f"  Newton iter {iteration}: ||F|| = {norm:.3e}, ",np.sqrt(rho_grid[np.argmin(U_new)])*1.414)
        if norm < tol:
            return U_new
    print(first_derivative(U_old2))
    plt.plot(np.sqrt(rho_grid)*1.414, U_old, label=str(k))
    plt.show()
    exit()
    return 0*U_old

# ==== MAIN FLOW LOOP ====
def run_flow(U0, rho, k_init, k_final, dk):
    k_vals = np.arange(k_init, k_final + dk, dk)
    U_old = U0.copy()
    U_old2 = U_old 
    U_old3 = U_old 
    history = [U_old.copy()]
    k_record = [k_init]

    for k in k_vals[1:]:
        print(f"\n== Solving for k = {k:.3f} ==")
        U_new = -0*0.000002*(rho_grid) +  lam * ((rho_grid)**2)
        U_current = newton_raphson(U_old,U_old2, U_old3, rho, k, dk,U_new)
        if np.all(U_current == 0) :
         U_new=U_old
         U_current = newton_raphson(U_old,U_old2, U_old3, rho, k, dk,U_new)
 
        history.append(U_current.copy())
        k_record.append(k)
        if k==k_init :
         U_old3=U_old2
         U_old2=U_old
         U_old=U_current
        elif k == (k_init+dk):
         U_old3=U_old2
         U_old2=U_old
         U_old=U_current
        else :
         U_old3=U_old2
         U_old2=U_old
         U_old=U_current
    """ 
    dk=-0.00001    
    k_init=k_final
    k_final=0.0001
    k_vals = np.arange(k_init, k_final + dk, dk)
    for k in k_vals[1:]:
        print(f"\n== Solving for k = {k:.5f} ==")
        U_new = -0*0.000002*(rho_grid) +  lam * ((rho_grid)**2)
        U_current = newton_raphson(U_old,U_old2, U_old3, rho, k, dk,U_new)
        if np.all(U_current == 0) :
         U_new=U_old
         U_current = newton_raphson(U_old,U_old2, U_old3, rho, k, dk,U_new)
 
        history.append(U_current.copy())
        k_record.append(k)
        if k==k_init :
         U_old3=U_old2
         U_old2=U_old
         U_old=U_current
        elif k == (k_init+dk):
         U_old3=U_old2
         U_old2=U_old
         U_old=U_current
        else :
         U_old3=U_old2
         U_old2=U_old
         U_old=U_current
      """
    return np.array(k_record), np.array(history)
# ==== RUN ====
k_vals, U_history = run_flow(U0, rho_grid, k_init, k_final, dk)
print(U_history[-1],U_history[0]/500**4)
# ==== PLOT ====
import matplotlib.pyplot as plt
#for i in range(0, len(k_vals)):
plt.plot(np.sqrt(rho_grid)*1.414, U_history[-1]-U_history[-1,0], label=f'Towards IR, k={0.3:.2f} GeV')
plt.plot(np.sqrt(rho_grid)*1.414, U_history[0]-U_history[0,0], label=f'UV, k={1.2:.2f} GeV')
#plt.plot(rho_grid, U_history[0], label=f'k={k_vals[-1]:.2f}')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$U_k(\phi)$')
#plt.xlim([0,0.1])
plt.title("Effective Potential Flow")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

