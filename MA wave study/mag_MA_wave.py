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
Npt_ks=2e3
Npt_kz=2e3

nprocs=8
nprocz=int(size/nprocs)

if rank==0:print('np_z',nprocz,'np_s',nprocs)

nproc=size #number of processes

beta=1;
if beta==0:
   d=0.25000;Rm=0.01;S=2e3;eta=1;P=1e3;
   mu=4*np.pi*1e-7;rho=1000;murho=csqrt(mu*rho);g=9.8;
   td=d**2/eta;B0=murho*d*S/td;Va=B0/murho;
   tM=d/Va;to=td/Rm;
   nu=0;tnu=np.inf;
   tA=np.inf;
   kappa=eta/P;
   tka=d**2/kappa;
   alpha=1;
   
if beta==1:
   d=0.25000;Rm=0.01;S=2e3;Q=1e-2;P=1e3;
   mu=4*np.pi*1e-7;rho=1000;murho=csqrt(mu*rho);g=9.8;
   alpha=1;tA=1/csqrt(g*alpha*np.abs(beta));
   eta=d**2/(Q*S*tA);kappa=eta/P;
   td=d**2/eta;B0=murho*d*S/td;Va=B0/murho;
   tM=d/Va;to=td/Rm;
   tka=d**2/kappa;
   nu=0;tnu=np.inf;
   
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
   L1      = 9*omgA**2*(2*omgeta-omgka-omgnu)+(omgeta-2*omgka+omgnu)*(2*omgeta**2+omgeta*(omgka-5*omgnu)-omgka**2+omgka*omgnu-9*omgM**2+2*omgnu**2)
   M1      =  3*(omgA**2+omgeta*(omgka+omgnu)+omgka*omgnu+omgM**2)-(omgeta+omgka+omgnu)**2
   
   l1     =-(csqrt(-1)/12)*(2**(2/3)*(1-csqrt(-1)*csqrt(3))*(csqrt(L1**2+4*M1**3)+L1)**(1/3)-((2*csqrt(-1)*(2)**(1/3)*(csqrt(3)-csqrt(-1))*M1)/((csqrt(L1**2+4*M1**3)+L1)**(1/3)))-4*(omgeta+omgka+omgnu))
   l2     =-(csqrt(-1)/12)*(2**(2/3)*(1+csqrt(-1)*csqrt(3))*(csqrt(L1**2+4*M1**3)+L1)**(1/3)+((2*csqrt(-1)*(2)**(1/3)*(csqrt(3)+csqrt(-1))*M1)/((csqrt(L1**2+4*M1**3)+L1)**(1/3)))-4*(omgeta+omgka+omgnu))
   l3     =(csqrt(-1)/6)*(2**(2/3)*(csqrt(L1**2+4*M1**3)+L1)**(1/3)-((2*(2)**(1/3)*M1)/((csqrt(L1**2+4*M1**3)+L1)**(1/3)))+2*(omgeta+omgka+omgnu))

   return KS,KZ,K,l1,l2,l3,thetahat0

"""************************************************************************************************************************************ """
def b_hat_pol(KS,KZ,K,l1,l2,l3,thetahat0,t):
   b1=0;
   b2=0;
   b3=alpha*g*B0*csqrt(-1)*KZ*KS*thetahat0/K**2;
   
   P=-(b3-csqrt(-1)*b2*(l2+l3))/((l1-l2)*(l1-l3))
   
   Q=-(b3-csqrt(-1)*b2*(l1+l3))/((l2-l1)*(l2-l3))
   
   R=-(b3-csqrt(-1)*b2*(l1+l2))/((l3-l1)*(l3-l2))
   
   Phihat = P*np.exp(csqrt(-1)*l1*t) + Q*np.exp(csqrt(-1)*l2*t)+ R*np.exp(csqrt(-1)*l3*t);
   PhihatP=P*np.exp(csqrt(-1)*l1*t);
   PhihatQ=Q*np.exp(csqrt(-1)*l2*t);
   PhihatR=R*np.exp(csqrt(-1)*l3*t);
   
   bhats=-csqrt(-1)*KZ*Phihat
   bhatz=KS*Phihat
   bhatsP=-csqrt(-1)*KZ*PhihatP
   bhatzP=KS*PhihatP
   bhatsQ=-csqrt(-1)*KZ*PhihatQ
   bhatzQ=KS*PhihatQ
   bhatsR=-csqrt(-1)*KZ*PhihatR
   bhatzR=KS*PhihatR
   return bhats,bhatz,bhatsP,bhatzP,bhatsQ,bhatzQ,bhatsR,bhatzR
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
       [bhats,bhatz,bhatsP,bhatzP,bhatsQ,bhatzQ,bhatsR,bhatzR]=b_hat_pol(KSB,KZB,KB,l1b,l2b,l3b,thetahat0b,t);
              
       Is=16*(np.pi)**(4)*np.trapz(np.trapz(ks_datab*np.real(bhatz)**2,ks_datab),kz_datab)
       
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
          #a2=2.5*10**((n/3)-1)
                 
       if (n-1)%3 == 0:
          a1=2.5*10**(((n-1)/3)-1)   
          #a2=7.5*10**(((n-1)/3)-1)
          
       if (n-2)%3 == 0:
          a1=7.5*10**(((n-2)/3)-1)   
          #a2=10*10**(((n-2)/3)-1)
          
       count=count+1   
       t_tM[count]=a1
       #print(count,'a1',a1,'t_tM',t_tM[count])
    integral_vars=['time','Eus','Euz','ks_us','kz_us','k_us','ks_uz','kz_uz','k_uz','EusP','EuzP','EusQ','EuzQ','EusR','EuzR','EusP2','EuzP2','EusQ2','EuzQ2','EusR2','EuzR2']
    variables=['time_tot','Eus_tot','Euz_tot','ks_us_tot','kz_us_tot','k_us_tot','ks_uz_tot','kz_uz_tot','k_uz_tot','EusP_tot','EuzP_tot','EusQ_tot','EuzQ_tot','EusR_tot','EuzR_tot','EusP2_tot','EuzP2_tot','EusQ2_tot','EuzQ2_tot','EusR2_tot','EuzR2_tot']
    if restart_num == 0:
       quantity=[[  ],[[  ] for _ in range(np.size(variables))]  ]
       for var in variables:
          quantity[0].append(var)
       if rank==0:
          print("initial run")
    
    if restart_num > 0:
       f=io.loadmat('wavenumber_Els1.mat')
       for i in range(np.size(f['t'])):
          azB[i]=f['kz'][0,i]
          asB[i]=f['ks'][0,i]  
       F=io.loadmat('inertial_Alfven_f_Els1_Rm10_Ro1e-2_Pm1e-10_1e5x1e5.mat')
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
          criteria=1e-10      
          pdiff_criteria=1e-2
       else:
          criteria=1e-10
          pdiff_criteria=1e-2
       
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
          [bhats,bhatz,bhatsP,bhatzP,bhatsQ,bhatzQ,bhatsR,bhatzR]=b_hat_pol(KS,KZ,K,l1,l2,l3,thetahat0,t[i]);
          
          integral_qty[0].append(t[i])
          integral_qty[1].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhats)**2 ,ks_data),kz_data))
          integral_qty[2].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatz)**2 ,ks_data),kz_data))
          integral_qty[3].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(KS*bhats)**2 ,ks_data),kz_data))
          integral_qty[4].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(KZ*bhats)**2 ,ks_data),kz_data))
          integral_qty[5].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(K*bhats)**2 ,ks_data),kz_data))
          integral_qty[6].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(KS*bhatz)**2 ,ks_data),kz_data))
          integral_qty[7].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(KZ*bhatz)**2 ,ks_data),kz_data))
          integral_qty[8].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(K*bhatz)**2 ,ks_data),kz_data))

          integral_qty[9].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhatsP)**2 ,ks_data),kz_data))          
          integral_qty[10].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatzP)**2 ,ks_data),kz_data))
          integral_qty[11].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhatsQ)**2 ,ks_data),kz_data))
          integral_qty[12].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatzQ)**2 ,ks_data),kz_data))
          integral_qty[13].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhatsR)**2 ,ks_data),kz_data))
          integral_qty[14].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatzR)**2 ,ks_data),kz_data))
          
          integral_qty[15].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatsP)**2 ,ks_data),kz_data))          
          integral_qty[16].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhatzP)**2 ,ks_data),kz_data))
          integral_qty[17].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatsQ)**2 ,ks_data),kz_data))
          integral_qty[18].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhatzQ)**2 ,ks_data),kz_data))
          integral_qty[19].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.real(bhatsR)**2 ,ks_data),kz_data))
          integral_qty[20].append(16*(np.pi)**(4)*np.trapz(np.trapz(ks_data*np.imag(bhatzR)**2 ,ks_data),kz_data))
          
          if i==(np.size(t)-1):
             #"""Finding new kz and ks max : new method """
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
             
                   if k > 0:
                      for i in range(np.size(gathered_quantity[0][0][:])):
                         summation=0
                         for j in range(nproc):    
                            summation=summation+gathered_quantity[k][j][i]
                         quantity[1][k].append(summation)

             
                print('saving gathered data')
                #os.remove('inertial_Alfven_Els1_Rm10_Ro1e-2_Pm1e-10_1e2s1e2.mat')
                io.savemat('mag_S2e3_Q1e-2_P1e3_stable_2e3x2e3.mat', mdict={quantity[0][i]:quantity[1][i][:] for i in range(np.size(quantity[0]))},appendmat=True)
                io.savemat('mag_S2e3_Q1e-2_P1e3_stable_2e3x2e3_b.mat', mdict={'ks':asB[0],'kz':azB[0],'tM':tM,'tA':tA,'td':td,'tka':tka,'Va':Va,'eta':eta,'beta':beta,'kappa':kappa},appendmat=True) 
           
           
           
           
           
      
                
           
           
           
           
           
      
