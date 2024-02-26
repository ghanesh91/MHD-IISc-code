import cython
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from mpi4py import MPI
from scipy import integrate
from scipy import special
from scipy import io
from array import array
import h5py
import sys
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

restart_num=0
Npt_ks=1e4
Npt_kz=1e4

nprocs=8
nprocz=int(size/nprocs)

if rank==0:print('np_z',nprocz,'np_s',nprocs)

nproc=size #number of processes

d=0.25000;Rm=0.01;
S=1e4;
Q=1e-2;
P=np.inf;
mu=4*np.pi*1e-7;rho=1000;
murho=csqrt(mu*rho);
g=9.8;
beta=1;
alpha=1;


tA=1/csqrt(g*alpha*np.abs(beta));
eta=d**2/(Q*S*tA);
kappa=0;#eta/P;
td=d**2/eta;
B0=murho*d*S/td;
Va=B0/murho;
tM=d/Va;
to=td/Rm;
tka=np.inf;#d**2/kappa;
nu=0;
tnu=np.inf;

if rank==0:print('tA',tA)
"""************************************************************************************************************************************ """
def distribute(kz,procs,chunks):
   """Creating "procss" number of empty chunks for storing the kz values that is to be divided across procss. """
   
   for i in range(procs):    
      actual=np.size(kz);extra=procs-1
      if i==(procs-1):
         m=actual+extra - (procs-1)*int(actual/procs);                
         m2=actual-1;m1=m2-m+1;
             
      else:
         m=int(np.size(kz)/procs)
         m1=(int(actual/procs)-1)*i;
         m2=m+m1-1;
                  
      for j in range(int(m1),int(m2+1)):
         chunks[i].append(kz[j])     
   return chunks     
"""************************************************************************************************************************************"""       
def universal_parameters(ks,kz):
   [KS,KZ]=np.meshgrid(ks,kz);
   K=(csqrt(KS**2+KZ**2));
   thetahat0=(d**3/16/csqrt(2)/np.pi**1.5)*np.exp(-K**2*d**2/8);
   omgM   = Va*KZ;
   omgnu  = nu*K**2;
   omgeta = eta*K**2;
   omgka  = kappa*K**2;
   omgA   = csqrt(g*alpha*beta*(KS/K)**2);
   L      = 9*omgA**2*(2*omgeta-4*omgka-omgnu)+2*omgeta**3-3*omgeta**2*(omgka+omgnu)-3*omgeta*(omgka**2-4*omgka*omgnu+3*omgM**2+omgnu**2)+(omgka+omgnu)*(2 *omgka**2-5*omgka*omgnu-9*omgM**2+2*omgnu**2)
   M      = 3*(omgA**2+omgeta*(omgka+omgnu)+omgka*omgnu+omgM**2)-(omgeta+omgka+omgnu)**2
   l1     =-(csqrt(-1)/12)*(2**(2/3)*(1-csqrt(-1)*csqrt(3))*(csqrt(L**2+4*M**3)+L)**(1/3)-((2*csqrt(-1)*(2)**(1/3)*(csqrt(3)-csqrt(-1))*M)/((csqrt(L**2+4*M**3)+L)**(1/3)))-4*(omgeta+omgka+omgnu))
   l2     =-(csqrt(-1)/12)*(2**(2/3)*(1+csqrt(-1)*csqrt(3))*(csqrt(L**2+4*M**3)+L)**(1/3)+((2*csqrt(-1)*(2)**(1/3)*(csqrt(3)+csqrt(-1))*M)/((csqrt(L**2+4*M**3)+L)**(1/3)))-4*(omgeta+omgka+omgnu))
   l3     =(csqrt(-1)/6)*(2**(2/3)*(csqrt(L**2+4*M**3)+L)**(1/3)-((2*(2)**(1/3)*M)/((csqrt(L**2+4*M**3)+L)**(1/3)))+2*(omgeta+omgka+omgnu))
   return KS,KZ,K,l1,l2,l3,thetahat0

"""************************************************************************************************************************************ """
def u_hat_pol(KS,KZ,K,l1,l2,l3,thetahat0,t):
   a1=0;
   a2= g*alpha*KS*thetahat0/K**2;
   a3=-g*alpha*KS*(nu+kappa)*thetahat0;
   
   A=-(a3-csqrt(-1)*a2*(l2+l3))/((l1-l2)*(l1-l3))
   
   B=-(a3-csqrt(-1)*a2*(l1+l3))/((l2-l1)*(l2-l3))
   
   C=-(a3-csqrt(-1)*a2*(l1+l2))/((l3-l1)*(l3-l2))
   
   psihat = A*np.exp(csqrt(-1)*l1*t) + B*np.exp(csqrt(-1)*l2*t)+ C*np.exp(csqrt(-1)*l3*t);
   psihatA=A*np.exp(csqrt(-1)*l1*t);
   psihatB=B*np.exp(csqrt(-1)*l2*t);
   psihatC=C*np.exp(csqrt(-1)*l3*t);
   
   uhats=-csqrt(-1)*KZ*psihat
   uhatz=KS*psihat
   uhatsA=-csqrt(-1)*KZ*psihatA
   uhatzA=KS*psihatA
   uhatsB=-csqrt(-1)*KZ*psihatB
   uhatzB=KS*psihatB
   uhatsC=-csqrt(-1)*KZ*psihatC
   uhatzC=KS*psihatC
   return uhats,uhatz,uhatsA,uhatzA,uhatsB,uhatzB,uhatsC,uhatzC
"""************************************************************************************************************************************"""
def kzksmax2(n,asB,azB,consts,constz,t):
              
       ASB=asB[n]*(Ls/(2*np.pi))*consts;  
       ksb=(2*np.pi/Ls)*np.linspace(1e-6,ASB,num=Npt_ks)

       AZB=azB[n]*(Lz/(2*np.pi))*constz;
       kzb=(2*np.pi/Lz)*np.linspace(1e-6,AZB,num=Npt_kz); 
       
       schunksb = [[] for _ in range(nprocs)]
       chunks_sb= distribute(ksb,nprocs,schunksb)
       
       zchunksb = [[] for _ in range(nprocz)]
       chunks_zb=distribute(kzb,nprocz,zchunksb)
       
       for i in range(nprocs):
           for j in range(nprocz):
               m = (i+j)+(i*(nprocz-1))
               if rank==m:
                  ks_datab=chunks_sb[i]   
                  kz_datab=chunks_zb[j]
                   
       [KSB,KZB,KB,l1b,l2b,l3b,thetahat0b]=universal_parameters(ks_datab,kz_datab)
       [uhats,uhatz,uhatsA,uhatzA,uhatsB,uhatzB,uhatsC,uhatzC]=u_hat_pol(KSB,KZB,KB,l1b,l2b,l3b,thetahat0b,t);
              
       Is=16*(np.pi)**(4)*np.trapz(np.trapz(ks_datab*np.real(uhatz)**2,ks_datab),kz_datab)
       
       gathered_Is=comm.gather(Is,root=0)                 
       
       if rank==0:
          summations=np.zeros((1,1),dtype=np.float64)
          print("Gathered qty shape",np.shape(gathered_Is))
          #summation[0][0]=0
          for i in range(np.size(gathered_Is[:])):
              summations[0][0]=summations[0][0]+gathered_Is[i]
       else:
          summations=np.zeros((1,1),dtype=np.float64)
          #summation[0][0]=None
       comm.Bcast(summations,root=0)       
       return summations[0][0],asB[n]*consts,azB[n]*constz
"""************************************************************************************************************************************"""
if __name__=='__main__':
    pdiff=0
    rank_pos=np.zeros((nproc,2))
    
    asB=np.zeros(101)
    azB=np.zeros(101)
    
    asB[0]=25
    azB[0]=25
    
    t_tM=np.zeros((101),dtype=np.float64)
    count=0;
    t_tM[0]=0; 
    for n in range(100):
       
       if n%3 == 0:
          a1=10**((n/3)-1)
                 
       if (n-1)%3 == 0:
          a1=2.5*10**(((n-1)/3)-1)   
          
       if (n-2)%3 == 0:
          a1=7.5*10**(((n-2)/3)-1)   
          
       count=count+1   
       t_tM[count]=a1

    integral_vars=['time','Eus','Euz','ks_us','kz_us','k_us','ks_uz','kz_uz','k_uz','EusA','EuzA','EusB','EuzB','EusC','EuzC','EusA2','EuzA2','EusB2','EuzB2','EusC2','EuzC2']
    variables=['time_tot','Eus_tot','Euz_tot','ks_us_tot','kz_us_tot','k_us_tot','ks_uz_tot','kz_uz_tot','k_uz_tot','EusA_tot','EuzA_tot','EusB_tot','EuzB_tot','EusC_tot','EuzC_tot','EusA2_tot','EuzA2_tot','EusB2_tot','EuzB2_tot','EusC2_tot','EuzC2_tot']
    if restart_num == 0:
       quantity=[[  ],[[  ] for _ in range(np.size(variables))]  ]
       for var in variables:
          quantity[0].append(var)
       if rank==0:
          print("initial run")
    
    if restart_num > 0:
       F=io.loadmat('S1e4_Q1e-2_PInf_stable_1e4x1e4.mat')
       quantity=[[ ],[[ ] for _ in range(np.size(variables))]]
       for i in range(np.size(variables)):
          quantity[0].append(variables[i])
          for j in range(231):
             quantity[1][i].append(F[variables[i]][0][j])      

    if rank==0:
       rank_pos=np.zeros((nproc,2),dtype=np.float64)   
       for i in range(nprocs):
          for j in range(nprocz):
             m = (i+j)+(i*(nprocz-1))
             rank_pos[m,0]=i
             rank_pos[m,1]=j 
    
    comm.Bcast(rank_pos,root=0)
    
    print('rank',rank,rank_pos[rank])      
    
    Ls=100
    Lz=100
    for n in range(restart_num,41):
       t=np.linspace(t_tM[n]*tM,t_tM[n+1]*tM,num=11)
       if n < 20:
          pdiff_criteria=1e-2
       else:
          pdiff_criteria=1e-1
       
       if rank==0:print('n=',n)   
          
       AZB=azB[0]*(Lz/(2*np.pi));
       kz=(2*np.pi/Lz)*np.linspace(1e-6,AZB,num=Npt_kz); 
       
       ASB=asB[0]*(Ls/(2*np.pi));  
       ks=(2*np.pi/Ls)*np.linspace(1e-6,ASB,num=Npt_ks);
       
       Nkzb=np.size(kz);
       
       """Distributing ks, ks and kz"""
       
       schunks = [[] for _ in range(nprocs)]
       chunks_s= distribute(ks,nprocs,schunks)
   
       zchunks = [[] for _ in range(nprocz)]
       chunks_z= distribute(kz,nprocz,zchunks)
       
       for i in range(nprocs):
          for j in range(nprocz):
             m = (i+j)+(i*(nprocz-1))
             if rank==m:
                  ks_data=chunks_s[i]   
                  kz_data=chunks_z[j]   
                                  
       [KS,KZ,K,l1,l2,l3,thetahat0]=universal_parameters(ks_data,kz_data)
       
       integral_qty=[[] for _ in range(np.size(integral_vars))]
       for i in range(np.size(t)):
          if rank==0:print("computing integrals at t/tM=",t[i]/tM,'num=',i+n*11,'kz_max',azB[n])
          [uhats,uhatz,uhatsA,uhatzA,uhatsB,uhatzB,uhatsC,uhatzC]=u_hat_pol(KS,KZ,K,l1,l2,l3,thetahat0,t[i]);
          
          integral_qty[0].append(t[i])
          integral_qty[1].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhats)**2 ,ks_data),kz_data))
          integral_qty[2].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatz)**2 ,ks_data),kz_data))
          integral_qty[3].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(KS*uhats)**2 ,ks_data),kz_data))
          integral_qty[4].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(KZ*uhats)**2 ,ks_data),kz_data))
          integral_qty[5].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(K*uhats)**2 ,ks_data),kz_data))
          integral_qty[6].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(KS*uhatz)**2 ,ks_data),kz_data))
          integral_qty[7].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(KZ*uhatz)**2 ,ks_data),kz_data))
          integral_qty[8].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(K*uhatz)**2 ,ks_data),kz_data))

          integral_qty[9].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhatsA)**2 ,ks_data),kz_data))          
          integral_qty[10].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatzA)**2 ,ks_data),kz_data))
          integral_qty[11].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhatsB)**2 ,ks_data),kz_data))
          integral_qty[12].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatzB)**2 ,ks_data),kz_data))
          integral_qty[13].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhatsC)**2 ,ks_data),kz_data))
          integral_qty[14].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatzC)**2 ,ks_data),kz_data))
          
          integral_qty[15].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatsA)**2 ,ks_data),kz_data))          
          integral_qty[16].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhatzA)**2 ,ks_data),kz_data))
          integral_qty[17].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatsB)**2 ,ks_data),kz_data))
          integral_qty[18].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhatzB)**2 ,ks_data),kz_data))
          integral_qty[19].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(uhatsC)**2 ,ks_data),kz_data))
          integral_qty[20].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(uhatzC)**2 ,ks_data),kz_data))
          
          if i==(np.size(t)-1):
             #"""Finding new kz and ks max"""
             ut_ks_maxB=0
             ut_kz_maxB=0
             consts=1;constz=1;
             [I1,ut_ks_maxB,ut_kz_maxB]=kzksmax2(n,asB,azB,consts,constz,t[i])
             
            
             if rank==0:print('I1',I1,'ksmax',ut_ks_maxB,'kzmax',ut_kz_maxB)
             if n>20:
                 consts=0.50;constz=0.50;
             else:
                 consts=0.50;constz=0.50;
                 
             [I2,ut_ks_maxB,ut_kz_maxB] = kzksmax2(n,asB,azB,consts,constz,t[i])
             
             
             if rank==0:print('I2',I2,'ksmax',ut_ks_maxB,'kzmax',ut_kz_maxB)
             pdiff=np.abs((I2-I1)/I1)*100
             
             if rank==0: print('For consts=',consts ,'constz=',constz,'pdiff=',pdiff,'kzmax=',ut_kz_maxB,'ksmax=',ut_ks_maxB)

             if pdiff < pdiff_criteria:
                if rank==0:print('pdiff < ',pdiff_criteria,' kzmax:',ut_kz_maxB,'ksmax:',ut_ks_maxB)
                asB[n+1]=ut_ks_maxB
                azB[n+1]=ut_kz_maxB
             
             if pdiff > pdiff_criteria:
                if rank==0:print("Entering while loop")
   
             while pdiff > pdiff_criteria:
                   consts=consts+(1-consts)*0.2
                   if n> 20:
                      constz=constz+(1-constz)*0.1
                   else:   
                      constz=constz+(1-constz)*0.20
                   [I2,ut_ks_maxB,ut_kz_maxB] = kzksmax2(n,asB,azB,consts,constz,t[i])             
                   pdiff=np.abs((I2-I1)/I1)*100
                   if rank==0: print('For consts=',consts ,'constz=',constz,'pdiff=',pdiff,'kzmax=',ut_kz_maxB,'ksmax=',ut_ks_maxB)
                   if pdiff < pdiff_criteria:
                      if rank==0:print('pdiff <', pdiff_criteria ,'for kzmax:',ut_kz_maxB,'ksmax:',ut_ks_maxB)
                      asB[n+1]=ut_ks_maxB
                      azB[n+1]=ut_kz_maxB
                      if rank==0:
                         print('ks_maxB',asB)
                         print('kz_maxB',azB)
                                  
          if i==(np.size(t)-1):                
             gathered_quantity = [[] for _ in range(np.size(integral_vars))]
             for i in range(np.size(integral_vars)):    
                gathered_quantity[i]=(comm.gather(integral_qty[i], root=0))
             gathered_qty=comm.gather(integral_qty,root=0)                 
             if rank==0:
                print("Gathered qty shape",np.size(gathered_qty[0][0][:]))
                count=0
                for k in range(np.size(variables)):
                   if k==0:
                      for i in range(np.size(gathered_quantity[0][0][:])):
                         quantity[1][k].append(gathered_quantity[k][0][i])
             
                   if k>0:
                      for i in range(np.size(gathered_quantity[0][0][:])):
                         summation=0
                         for j in range(nproc):    
                            summation=summation+gathered_quantity[k][j][i]
                         quantity[1][k].append(summation)

             
                print('saving gathered data')
                #os.remove('inertial_Alfven_Els1_Rm10_Ro1e-2_Pm1e-10_1e2s1e2.mat')
                io.savemat('S1e4_Q1e-2_PInf_stable_1e4x1e4.mat', mdict={quantity[0][i]:quantity[1][i][:] for i in range(np.size(quantity[0]))},appendmat=True)
                io.savemat('S1e4_Q1e-2_PInf_stable_1e4x1e4_b.mat', mdict={'ks':asB[0],'kz':azB[0],'tM':tM,'tA':tA,'td':td,'tka':tka,'Va':Va,'eta':eta,'beta':beta,'kappa':kappa},appendmat=True) 
           
           
           
           
           
      
                
"""Runing this code in AMD machine:
1. Login ghanesh@10.132.10.25, password: ghanesh123

2. source /share/apps/Dedalus/dedalus2/export_PATHS.sh

3. cd /share/apps/Dedalus/transition_analysis/stable_stratification/sample

   time /share/apps/openmpi/bin/mpirun -n 128 -H compute-0-1,compute-0-3 -x LD_LIBRARY_PATH=/share/apps/Dedalus/Dedalus_libraries/lib/ /share/apps/Dedalus/   Dedalus_libraries/bin/python3 MA_wave.py"""
           
           
"""Runing this code in Cray XC40:
1. Login esdghan@xc40.serc.iisc.ernet.in, password: esdghan@31esd
2. source /mnt/lustre/esd2/esdghan/GHANESH/dedalus2/export_PATH.sh
3. cd /mnt/lustre/esd2/esdghan/GHANESH/dedalus2/examples/alfven_diffusion/transition_analysis/forcing_Alfven/stable_stratification/sample
4. qsub script

"""           
      
