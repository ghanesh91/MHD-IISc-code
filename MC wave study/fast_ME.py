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

Npt_kr=2e4
Npt_kz=2e4
pdiff_criteria=1e-1
restart_num=0
nprocr=12
nprocz=int(size/nprocr)
if rank==0:print('np_r',nprocr,'np_z',nprocz)
mu=4*np.pi*1e-7;
rho=7000;
murho=csqrt(mu*rho);

nproc=size #number of processes

Els=1e-3;Rm=10;d=0.250000;Uo=0.10718*1e6;Ro=1e-3*Rm;Pm=1e-10;
L=np.sqrt((Els*Rm)/Ro)

eta=(Uo*d)/Rm;
mu=4*np.pi*1e-7;
rho=7000; 
Omega=Uo/(2*Ro*d);
Va=L*eta/d;
to=d/Uo;
tA=d/Va;
td=d**2/eta;
tau=eta/Va**2;
t_Omega=1/(2*Omega);
B0=Va*csqrt(mu*rho);
nu=Pm*eta;
Re=np.inf#Uo*d/nu;

if rank==0:print('tA',tA,'Pm',nu/eta,'Re',np.inf)
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
def universal_parameters(kz,kr):
   [KZ,KR]=np.meshgrid(kz,kr);
   K=(csqrt(KR**2+KZ**2));
   uhat0=(1e6*(KR*d**5/(16*np.pi**1.5))*np.exp(-K**2*d**2/4)); 
   l1= ((Omega*KZ)/K) + ((csqrt(-1)*(nu+eta)*K**2)/2) + csqrt(Va**2*KZ**2 +  ( ((Omega*KZ)/K) + ((csqrt(-1)*(nu - eta)*K**2)/2)  )**2  )
   l2= ((Omega*KZ)/K) + ((csqrt(-1)*(nu+eta)*K**2)/2) - csqrt(Va**2*KZ**2 +  ( ((Omega*KZ)/K) + ((csqrt(-1)*(nu - eta)*K**2)/2)  )**2  )
   l3= (-(Omega*KZ)/K) + ((csqrt(-1)*(nu+eta)*K**2)/2) + csqrt(Va**2*KZ**2 +  ( ((Omega*KZ)/K) - ((csqrt(-1)*(nu - eta)*K**2)/2)  )**2  )
   l4= (-(Omega*KZ)/K) + ((csqrt(-1)*(nu+eta)*K**2)/2) - csqrt(Va**2*KZ**2 +  ( ((Omega*KZ)/K) - ((csqrt(-1)*(nu - eta)*K**2)/2)  )**2  )
   
   return KZ,KR,K,l1,l2,l3,l4,uhat0 

"""************************************************************************************************************************************ """
def u_hat_theta(KZ,KR,K,l1,l2,l3,l4,uhat0,t):
   a1=uhat0
   a2=csqrt(-1)*nu*K**2*uhat0
   a3=(  Va**2*KZ**2+ ((4*Omega**2*KZ**2)/ K**2 ) - nu**2*K**4 )* uhat0
   a4=csqrt(-1)*( (2*nu+eta)*Va**2*KZ**2*K**2 + 12*Omega**2*nu*KZ**2 - nu**3*K**6 )* uhat0
   
   A=(a4-a3*(l2+l3+l4) + a2*(l2*(l3+l4) + l3*l4) - a1*l2*l3*l4) / ((l1-l2)*(l1-l3)*(l1-l4))
   
   B=(a4-a3*(l1+l3+l4) + a2*(l1*(l3+l4) + l3*l4) - a1*l1*l3*l4) / ((l2-l1)*(l2-l3)*(l2-l4))
   
   C=(a4-a3*(l1+l2+l4) + a2*(l1*(l2+l4) + l2*l4) - a1*l1*l2*l4) / ((l3-l1)*(l3-l2)*(l3-l4))

   D=(a4-a3*(l1+l2+l3) + a2*(l1*(l2+l3) + l2*l3) - a1*l1*l2*l3) / ((l4-l1)*(l4-l2)*(l4-l3))

   uhat_theta = A*np.exp(csqrt(-1)*l1*t) + B*np.exp(csqrt(-1)*l2*t) + C*np.exp(csqrt(-1)*l3*t) + D*np.exp(csqrt(-1)*l4*t)
   
   duhat_theta_dt = csqrt(-1)* ( A*l1*np.exp(csqrt(-1)*l1*t) + B*l2*np.exp(csqrt(-1)*l2*t) + C*l3*np.exp(csqrt(-1)*l3*t) + D*l4*np.exp(csqrt(-1)*l4*t) )

   uB = np.real(B*np.exp(csqrt(-1)*l2*t))
   uD = np.real(D*np.exp(csqrt(-1)*l4*t))
   
   return uhat_theta,duhat_theta_dt,uB,uD
"""********************************************************************************************************************************* """ 
def b_hat_theta(KZ,KR,K,l1,l2,l3,l4,uhat0,t):
   a1=0
   a2=B0*KZ*uhat0
   a3=(nu+eta)*B0*csqrt(-1)*KZ*K**2*uhat0
   a4=B0*KZ*(Va**2*KZ**2 + ((4*Omega**2*KZ**2)/K**2) - K**4*(nu**2 + eta**2 + nu*eta) )*uhat0
   
   A=(a4-a3*(l2+l3+l4) + a2*(l2*(l3+l4) + l3*l4) - a1*l2*l3*l4) / ((l1-l2)*(l1-l3)*(l1-l4))
   
   B=(a4-a3*(l1+l3+l4) + a2*(l1*(l3+l4) + l3*l4) - a1*l1*l3*l4) / ((l2-l1)*(l2-l3)*(l2-l4))
   
   C=(a4-a3*(l1+l2+l4) + a2*(l1*(l2+l4) + l2*l4) - a1*l1*l2*l4) / ((l3-l1)*(l3-l2)*(l3-l4))

   D=(a4-a3*(l1+l2+l3) + a2*(l1*(l2+l3) + l2*l3) - a1*l1*l2*l3) / ((l4-l1)*(l4-l2)*(l4-l3))

   bhat_theta = A*np.exp(csqrt(-1)*l1*t) + B*np.exp(csqrt(-1)*l2*t) + C*np.exp(csqrt(-1)*l3*t) + D*np.exp(csqrt(-1)*l4*t)
   
   dbhat_theta_dt = csqrt(-1)* ( A*l1*np.exp(csqrt(-1)*l1*t) + B*l2*np.exp(csqrt(-1)*l2*t) + C*l3*np.exp(csqrt(-1)*l3*t) + D*l4*np.exp(csqrt(-1)*l4*t) )

   jhat_r = -((csqrt(-1)*KZ)/mu)*bhat_theta
   djhat_r_dt = -((csqrt(-1)*KZ)/mu)*dbhat_theta_dt   

   jhat_z = -((csqrt(-1)*KR)/mu)*bhat_theta
   djhat_z_dt = -((csqrt(-1)*KR)/mu)*dbhat_theta_dt

   bB=B*np.exp(csqrt(-1)*l2*t)
   bA=A*np.exp(csqrt(-1)*l1*t)
   
   return jhat_r,jhat_z,bhat_theta,djhat_r_dt,djhat_z_dt,dbhat_theta_dt,bB,bA
"""************************************************************************************************************************************"""
def kzkrmax(uhat,kr_data,kz_data,name,criteria,az,ar,kz,kr):
             max_u=np.zeros((nproc,3),dtype=np.float64)
             max_proc=np.ones((1))
             max_kr_pos=np.zeros((1))
             max_kz_pos=np.zeros((1))
             
             for p in range(nproc):
                if rank==p:
                   if np.max(np.real(uhat)) == np.nan:
                      max_u[p,0]=0
                   else:
                      max_u[p,0]=np.max(np.real(uhat))
                   max_u[p,1]=rank_pos[p,0]
                   max_u[p,2]=rank_pos[p,1]               
   
             if rank==0:
                totals = np.zeros_like(max_u)
                totals_max=np.zeros((1))
             else:
                totals = None
                totals_max=np.zeros((1))
                 
             comm.Reduce([max_u, MPI.DOUBLE],[totals, MPI.DOUBLE],op = MPI.SUM,root = 0)
             if rank==0:
                totals_max[0]=np.max(totals[:,0]) 
                
                max_proc_pos=np.where(totals[:,0]==np.max(totals[:,0]))
                max_proc[0] = max_proc_pos[0][0]
             comm.Bcast(max_proc,root=0)     
             comm.Bcast(totals_max,root=0)
             
             if rank == max_proc[0]:
                umax_pos = np.where(np.real(uhat) == np.max(np.real(uhat)))
                max_kr_pos[0]=umax_pos[0][0]
                max_kz_pos[0]=umax_pos[1][0]
                #print(kr_data[umax_pos[0][0]],kz_data[umax_pos[1][0]])
             comm.Bcast(max_kr_pos,root=max_proc[0])
             comm.Bcast(max_kz_pos,root=max_proc[0])
             #print(rank,max_kr_pos,max_kz_pos)
             
             kz_max=np.zeros((1),dtype=np.float64)
             kz_max[0]=az[n]
             #if rank==0:print('kz1',kz_max[0])
             countz=np.ones((1))
             for i in range(int(max_proc[0]),int(max_proc[0]+nprocz-rank_pos[max_proc[0],1])):
                if countz[0]==0:
                   break
                else:
                   if rank==i:
                      #print("Entering proc kz",i)
                      for j in range(np.size(kz_data)):
                         #print('rank',i,name,np.real(uhat[max_kr_pos[0],j]),'kz',kz_data[j],'countz',countz)
                         if kz_data[j] > kz[max_kz_pos[0]]:
                            if np.abs(np.real(uhat[max_kr_pos[0],j]))/np.abs(totals_max) < criteria:
                               kz_max[0] = kz_data[j]
                               countz[0]=countz[0]-1
                               print('rank',i,name,'ratio',np.abs(np.real(uhat[max_kr_pos[0],j]))/np.abs(totals_max),'kzmax',kz_data[j])
                            break
                   comm.Bcast(kz_max,root=i)
                   comm.Bcast(countz,root=i)
             
             #if rank==0:print('kz2',kz_max[0])              
             
             kr_max=np.zeros((1),dtype=np.float64)
             kr_max[0]=ar[n]
             #if rank==0:print('kr1',kr_max[0])
             countr=np.ones((1))
             for i in range(int(max_proc[0]),int(max_proc[0]+(nprocr-1-rank_pos[max_proc[0],0])*nprocz+1),nprocz):
                if countr[0]==0:
                   break
                else:
                   if rank==i:
                      #print("Entering proc kr",i)
                      for j in range(np.size(kr_data)):
                         #print('rank',i,name,np.real(uhat[j,max_kz_pos[0]]),'kr',kr_data[j],'countr',countr)
                         if kr_data[j] > kr[max_kr_pos[0]]:
                            if np.abs(np.real(uhat[j,max_kz_pos[0]]))/np.abs(totals_max) < criteria:
                               kr_max[0] = kr_data[j]
                               countr[0]=countr[0]-1
                               print('rank',i,name,'ratio',np.abs(np.real(uhat[j,max_kz_pos[0]]))/np.abs(totals_max),'kr_max',kr_data[j])
                            break
                   comm.Bcast(kr_max,root=i)
                   comm.Bcast(countr,root=i)
             
             return kr_max[0],kz_max[0] 
"""************************************************************************************************************************************"""
def kzkrmax2(n,arB,azB,ARmin,AZmin,constr,constz,t):
       
       ARB=arB[n]*(Lr/(2*np.pi))*constr;  
       krb=(2*np.pi/Lr)*np.linspace(ARmin,ARB,num=Npt_kr)
       
       AZB=azB[n]*(Lz/(2*np.pi))*constz;
       kzb=(2*np.pi/Lz)*np.linspace(AZmin,AZB,num=Npt_kz); 
       
       rchunksb = [[] for _ in range(nprocr)]
       chunks_rb = distribute(krb,nprocr,rchunksb)
       
       zchunksb = [[] for _ in range(nprocz)]
       chunks_zb=distribute(kzb,nprocz,zchunksb)
        
       for i in range(nprocr):
          for j in range(nprocz):
                m = (i+j)+(i*(nprocz-1))
                if rank==m:
                   kr_datab=chunks_rb[i]
                   kz_datab=chunks_zb[j]
                   
       [KZB,KRB,KB,l1b,l2b,l3b,l4b,uhat0b]=universal_parameters(kz_datab,kr_datab)
       [jhat_r,jhat_z,bhat_theta,djhat_r_dt,djhat_z_dt,dbhat_theta_dt,bB,bA]=b_hat_theta(KZB,KRB,KB,l1b,l2b,l3b,l4b,uhat0b,t)
       I=32*np.pi**4*np.trapz(kr_datab*np.trapz(np.imag(bA)**2,kz_datab),kr_datab)
       
       gathered_I=comm.gather(I,root=0)                 
       if rank==0:
          summation=np.zeros((1,1),dtype=np.float64)
          print("Gathered qty shape",np.shape(gathered_I))
          #summation[0][0]=0
          for i in range(np.size(gathered_I[:])):
              summation[0][0]=summation[0][0]+gathered_I[i]
       else:
          summation=np.zeros((1,1),dtype=np.float64)
          #summation[0][0]=None
       comm.Bcast(summation,root=0)       
       
       return summation[0][0],arB[n]*constr,azB[n]*constz
"""************************************************************************************************************************************"""
if __name__=='__main__':
    pdiff=0
    rank_pos=np.zeros((nproc,2))
    azB=np.zeros(101)
    arB=np.zeros(101)
    
    azB[0]=50
    arB[0]=50
    
    t_tA=np.zeros((101),dtype=np.float64)
    count=0;
    t_tA[0]=0; 
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
       t_tA[count]=a1
       #print(count,'a1',a1,'t_tA',t_tA[count])
    
    integral_vars=['time','toroidal_uB','kr_u','kz_u','k_u']
    variables=['time_tot','toroidal_uB_tot','kr_u_tot','kz_u_tot','k_u_tot']
    if restart_num == 0:
       quantity=[ [  ]  ,  [[  ] for _ in range(np.size(variables))]  ]
       for var in variables:
          quantity[0].append(var)
       if rank==0:
          print("initial run")
          f=h5py.File('inertial_Alfven_f_Els1e-3_Rm10_Ro1e-2_Pm1e-10_2e4x2e4.h5','w')
          time_tot=f.create_dataset('time_tot',(1,101),dtype=np.float64)
          kz_maxB=f.create_dataset('kz_maxB',(1,101),dtype=np.float64)
          kr_maxB=f.create_dataset('kr_maxB',(1,101),dtype=np.float64)
    
    if restart_num > 0:
       f=io.loadmat('wavenumber_Els1e-3.mat')
       for i in range(np.size(f['t'])):
          azB[i]=f['kz'][0,i]
          arB[i]=f['kr'][0,i]  
       F=io.loadmat('inertial_Alfven_f_Els1e-3_Rm10_Ro1e-2_Pm1e-10_2e4x2e4.mat')
       quantity=[ [  ]  ,  [[  ] for _ in range(np.size(variables))]]
       for i in range(np.size(variables)):
          quantity[0].append(variables[i])
          for j in range(33):
             quantity[1][i].append(F[variables[i]][0][j])      
    if rank==0:
       rank_pos=np.zeros((nproc,2),dtype=np.float64)               
       for i in range(nprocr):
          for j in range(nprocz):
             m = (i+j)+(i*(nprocz-1))
             rank_pos[m,0]=i
             rank_pos[m,1]=j
    comm.Bcast(rank_pos,root=0)
    print('rank',rank,rank_pos[rank])      
    Lr=100
    Lz=5*10**10*.25
    for n in range(restart_num,41):
       t=np.linspace(t_tA[n]*t_Omega,t_tA[n+1]*t_Omega,num=11)
       if n < 25:
          criteria=1e-10      
          ARmin=1e-5*(Lr/(2*np.pi));
          AZmin=1e-5*(Lz/(2*np.pi)); 
       else:
          criteria=1e-10
          ARmin=1e-5*(Lr/(2*np.pi));
          AZmin=1e-5*(Lz/(2*np.pi)); 
       
       if rank==0:print('n=',n)   
       
       ARB=arB[n]*(Lr/(2*np.pi));  
       krb=(2*np.pi/Lr)*np.linspace(ARmin,ARB,num=Npt_kr)
       
       """kz vector"""
       
       AZB=azB[n]*(Lz/(2*np.pi));
       kzb=(2*np.pi/Lz)*np.linspace(AZmin,AZB,num=Npt_kz); 
       Nkzb=np.size(kzb);
       
       """ Distributing kz and kr"""
       
       rchunksb = [[] for _ in range(nprocr)]
       chunks_rb = distribute(krb,nprocr,rchunksb)
       
       zchunksb = [[] for _ in range(nprocz)]
       chunks_zb=distribute(kzb,nprocz,zchunksb)
        
       for i in range(nprocr):
          for j in range(nprocz):
                m = (i+j)+(i*(nprocz-1))
                if rank==m:
                   kr_datab=chunks_rb[i]
                   kz_datab=chunks_zb[j]
                   
       [KZB,KRB,KB,l1b,l2b,l3b,l4b,uhat0b]=universal_parameters(kz_datab,kr_datab)
       
       integral_qty=[[] for _ in range(np.size(integral_vars))]
       for i in range(np.size(t)):
          if rank==0:print("computing integrals at t/t_Omega=",t[i]/t_Omega,'num=',i+n*11,'kz_max',azB[n])
          [jhat_r,jhat_z,bhat_theta,djhat_r_dt,djhat_z_dt,dbhat_theta_dt,bB,bA]=b_hat_theta(KZB,KRB,KB,l1b,l2b,l3b,l4b,uhat0b,t[i])
          
          integral_qty[0].append(t[i])
          integral_qty[1].append(32*np.pi**4*np.trapz(kr_datab*np.trapz(np.imag(bA)**2,kz_datab),kr_datab))
          integral_qty[2].append(32*np.pi**4*np.trapz(kr_datab*np.trapz(np.abs(KRB*np.imag(bA))**2,kz_datab),kr_datab))
          integral_qty[3].append(32*np.pi**4*np.trapz(kr_datab*np.trapz(np.abs(KZB*np.imag(bA))**2,kz_datab),kr_datab))
          integral_qty[4].append(32*np.pi**4*np.trapz(kr_datab*np.trapz(np.abs(KB*np.imag(bA))**2,kz_datab),kr_datab))
          
          if i==(np.size(t)-1):
      
             """ Storing current max(kr,kz) values"""
             #"""Finding new kz and kr max : old method """
             #ut_kr_maxB =0
             #ut_kz_maxB =0
             
             #[ut_kr_maxB,ut_kz_maxB]=kzkrmax(np.real(uB1),kr_datab,kz_datab,'uhat_thetaB',criteria,azB,arB,kzb,krb)
             #if rank==0:print('ut_kr_maxB',ut_kr_maxB,'ut_kz_maxB',ut_kz_maxB,criteria)
             
             #arB[n+1]=ut_kr_maxB
             #azB[n+1]=ut_kz_maxB
                    
             """Finding new kz and kr max : new method """
             ut_kr_maxB  = 0
             ut_kz_maxB = 0
             constr=1;constz=1;
             [I1,ut_kr_maxB,ut_kz_maxB] = kzkrmax2(n,arB,azB,ARmin,AZmin,constr,constz,t[i])
             
             constr=0.5;constz=0.5;
             [I2,ut_kr_maxB,ut_kz_maxB] = kzkrmax2(n,arB,azB,ARmin,AZmin,constr,constz,t[i])
             
             pdiff=np.abs((I2-I1)/I1)*100
             
             if rank==0: print('For constr=',constr ,'constz=',constz,'pdiff=',pdiff,'kzmax=',ut_kz_maxB,'krmax=',ut_kr_maxB)
             if pdiff < pdiff_criteria:
                if rank==0:print('pdiff < ',pdiff_criteria,' kzmax:',ut_kz_maxB,'krmax:',ut_kr_maxB)
                arB[n+1]=ut_kr_maxB
                azB[n+1]=ut_kz_maxB
             if pdiff > pdiff_criteria:
                if rank==0:print("Entering while loop")
   
             while pdiff > pdiff_criteria:
                   constr=constr+(1-constr)*0.5
                   constz=constz+(1-constz)*0.5
                   [I2,ut_kr_maxB,ut_kz_maxB] = kzkrmax2(n,arB,azB,ARmin,AZmin,constr,constz,t[i])             
                   pdiff=np.abs((I2-I1)/I1)*100
                   if rank==0: print('For constr=',constr ,'constz=',constz,'pdiff=',pdiff,'kzmax=',ut_kz_maxB,'krmax=',ut_kr_maxB)
                   if pdiff < pdiff_criteria:
                      if rank==0:print('pdiff <', pdiff_criteria ,'for kzmax:',ut_kz_maxB,'krmax:',ut_kr_maxB)
                      arB[n+1]=ut_kr_maxB
                      azB[n+1]=ut_kz_maxB
                      if rank==0:
                         print('kr_maxB',arB)
                         print('kz_maxB',azB)
   
                               
          if i==(np.size(t)-1):#i+n*11 == 65:                
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
                io.savemat('inertial_Alfven_f_Els1e-3_Rm10_Ro1e-2_Pm1e-10_2e4x2e4_ME.mat', mdict={quantity[0][i]:quantity[1][i][:] for i in range(np.size(quantity[0]))},appendmat=False) 
                io.savemat('wavenumber_Els1e-3.mat', mdict={'kr':arB,'kz':azB,'t':t_tA},appendmat=False)
             
           
           
           
           
           
      
