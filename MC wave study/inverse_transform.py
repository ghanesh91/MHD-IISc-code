import cython
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from mpi4py import MPI
from scipy import integrate
from scipy.special import jn
from scipy import io
from array import array
import h5py
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Npt_kr=3.5e3
Npt_kz=3.8e3

nprocr=12
nprocz=int(size/nprocr)
if rank==0:print('np_r',nprocr,'np_z',nprocz)
mu=4*np.pi*1e-7;
rho=7000;
murho=csqrt(mu*rho);

nproc=size #number of processes

d=0.25000;Uo=0.10718*1e6;
Ro=1e-2;Pm=1e-10;
Rm=10;
Els=1000;
L=np.sqrt(Els*(Rm/Ro));

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
Re=Uo*d/nu;
tnu = d**2/nu;

if rank==0:print('tA',tA,'Pm',nu/eta,'Re',Uo*d/nu)
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
def u_hat_theta(t):
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

   return uhat_theta,duhat_theta_dt
"""************************************************************************************************************************************ """
def gamma_hat(t):
   gamma0= ((KR*d**6)/(128*np.pi))*(np.exp( - (KZ**2 + (KR**2/2))*(d**2/4)  ))* ( (KR**2*d**2-2)*special.iv(1,KR**2*d**2/8) - (KR**2*d**2-6)*special.iv(0,KR**2*d**2/8)    )
   a1=gamma0
   a2=csqrt(-1)*nu*K**2*gamma0
   a3=(  Va**2*KZ**2+ ((4*Omega**2*KZ**2)/ K**2 ) - nu**2*K**4 )* gamma0
   a4=csqrt(-1)*( (2*nu+eta)*Va**2*KZ**2*K**2 + 12*Omega**2*nu*KZ**2 - nu**3*K**6 )* gamma0
   
   A=(a4-a3*(l2+l3+l4) + a2*(l2*(l3+l4) + l3*l4) - a1*l2*l3*l4) / ((l1-l2)*(l1-l3)*(l1-l4))
   
   B=(a4-a3*(l1+l3+l4) + a2*(l1*(l3+l4) + l3*l4) - a1*l1*l3*l4) / ((l2-l1)*(l2-l3)*(l2-l4))
   
   C=(a4-a3*(l1+l2+l4) + a2*(l1*(l2+l4) + l2*l4) - a1*l1*l2*l4) / ((l3-l1)*(l3-l2)*(l3-l4))

   D=(a4-a3*(l1+l2+l3) + a2*(l1*(l2+l3) + l2*l3) - a1*l1*l2*l3) / ((l4-l1)*(l4-l2)*(l4-l3))

   gammahat = A*np.exp(csqrt(-1)*l1*t) + B*np.exp(csqrt(-1)*l2*t) + C*np.exp(csqrt(-1)*l3*t) + D*np.exp(csqrt(-1)*l4*t)
   
   return gammahat, gamma0

"""************************************************************************************************************************************ """
def u_hat_poloidal(t):
   a1=0
   a2=2*Omega*KZ*uhat0
   a3=4*Omega*nu*csqrt(-1)*KZ*K**2*uhat0
   a4=2*Omega*KZ*(2*Va**2*KZ**2 + ((4*Omega**2*KZ**2)/K**2) - 3*nu**2*K**4)*uhat0
   
   A=(a4-a3*(l2+l3+l4) + a2*(l2*(l3+l4) + l3*l4) - a1*l2*l3*l4) / ((l1-l2)*(l1-l3)*(l1-l4))
   
   B=(a4-a3*(l1+l3+l4) + a2*(l1*(l3+l4) + l3*l4) - a1*l1*l3*l4) / ((l2-l1)*(l2-l3)*(l2-l4))
   
   C=(a4-a3*(l1+l2+l4) + a2*(l1*(l2+l4) + l2*l4) - a1*l1*l2*l4) / ((l3-l1)*(l3-l2)*(l3-l4))

   D=(a4-a3*(l1+l2+l3) + a2*(l1*(l2+l3) + l2*l3) - a1*l1*l2*l3) / ((l4-l1)*(l4-l2)*(l4-l3))

   omghat_theta = A*np.exp(csqrt(-1)*l1*t) + B*np.exp(csqrt(-1)*l2*t) + C*np.exp(csqrt(-1)*l3*t) + D*np.exp(csqrt(-1)*l4*t)
   domghat_theta_dt = csqrt(-1)* ( A*l1*np.exp(csqrt(-1)*l1*t) + B*l2*np.exp(csqrt(-1)*l2*t) + C*l3*np.exp(csqrt(-1)*l3*t) + D*l4*np.exp(csqrt(-1)*l4*t) )

   uhat_r = -csqrt(-1)*(KZ/K**2)*omghat_theta
   duhat_r_dt= -csqrt(-1)*(KZ/K**2)*domghat_theta_dt

   uhat_z = csqrt(-1)*(KR/K**2)*omghat_theta
   duhat_z_dt = csqrt(-1)*(KR/K**2)*domghat_theta_dt

   return uhat_r, uhat_z, omghat_theta,duhat_r_dt,duhat_z_dt,domghat_theta_dt
"""************************************************************************************************************************************"""              
def b_hat_theta(t):
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

   return jhat_r,jhat_z,bhat_theta,djhat_r_dt,djhat_z_dt,dbhat_theta_dt
"""************************************************************************************************************************************"""              
def b_hat_poloidal(t):
   a1=0
   a2=0
   a3=(2*Omega*B0*KZ**2)*(uhat0/mu)
   a4=csqrt(-1)*(2*Omega*(2*nu+eta)*B0*KZ**2*K**2)*(uhat0/mu)
   
   A=(a4-a3*(l2+l3+l4) + a2*(l2*(l3+l4) + l3*l4) - a1*l2*l3*l4) / ((l1-l2)*(l1-l3)*(l1-l4))
   
   B=(a4-a3*(l1+l3+l4) + a2*(l1*(l3+l4) + l3*l4) - a1*l1*l3*l4) / ((l2-l1)*(l2-l3)*(l2-l4))
   
   C=(a4-a3*(l1+l2+l4) + a2*(l1*(l2+l4) + l2*l4) - a1*l1*l2*l4) / ((l3-l1)*(l3-l2)*(l3-l4))

   D=(a4-a3*(l1+l2+l3) + a2*(l1*(l2+l3) + l2*l3) - a1*l1*l2*l3) / ((l4-l1)*(l4-l2)*(l4-l3))

   jhat_theta = A*np.exp(csqrt(-1)*l1*t) + B*np.exp(csqrt(-1)*l2*t) + C*np.exp(csqrt(-1)*l3*t) + D*np.exp(csqrt(-1)*l4*t)
   djhat_theta_dt = csqrt(-1)* ( A*l1*np.exp(csqrt(-1)*l1*t) + B*l2*np.exp(csqrt(-1)*l2*t) + C*l3*np.exp(csqrt(-1)*l3*t) + D*l4*np.exp(csqrt(-1)*l4*t) )

   bhat_r = - (csqrt(-1)*mu*KZ*jhat_theta)/K**2
   dbhat_r_dt = - (csqrt(-1)*mu*KZ*djhat_theta_dt)/K**2   


   bhat_z =   (csqrt(-1)*mu*KR*jhat_theta)/K**2
   dbhat_z_dt =   (csqrt(-1)*mu*KR*djhat_theta_dt)/K**2
   
   return bhat_r,bhat_z,jhat_theta, dbhat_r_dt, dbhat_z_dt, djhat_theta_dt
"""************************************************************************************************************************************"""              

def distribute(kz,nproc,chunks):
   """Creating "nprocs" number of empty chunks for storing the kz values that is to be divided across nprocs. """
   for i in range(nproc):    
      actual=np.size(kz);extra=nproc-1
      if i==(nproc-1):
         m=actual+extra - (nproc-1)*int(actual/nproc);                
         m2=actual-1;m1=m2-m+1;
             
      else:
         m=int(np.size(kz)/nproc)
         m1=(int(actual/nproc)-1)*i;
         m2=m+m1-1;
                  
      for j in range(int(m1),int(m2+1)):
         chunks[i].append(kz[j])     
   return chunks     
"""************************************************************************************************************************************ """
def inverse(uhat,kr,kz,r,z): 
   Nr=np.size(r)
   Nz=np.size(z)
   Nkr=np.size(kr)
   Nkz=np.size(kz)
   I=np.zeros((Nkr,Nz), np.complex128);
   u=np.zeros((Nr,Nz), np.complex128);
   if rank==0:print('ir',r)
   for i in range(Nkr):
      if rank==0:print('kr',i,kr[i])
      for j in range(Nz):
          I[i,j]=np.trapz(uhat[i,:]*np.exp(csqrt(-1)*np.multiply(kz,z[j])),kz);
          
       
   for i in range(Nr):
      if rank==0:print('r',i,r[i])
      for j in range(Nz):
         u[i,j]=4*np.pi*np.trapz(kr*jn(1,np.multiply(kr,r[i]))*I[:,j],kr);
   return u      
"""************************************************************************************************************************************ """         
if __name__=='__main__':
   t_tA=np.zeros((41),dtype=np.float64)
   rank_pos=np.zeros((nproc,2))
   count=0;
   t_tA[0]=0; 
   for i in range(40):
       
      if i%3 == 0:
         a1=10**((i/3)-1)
         #a2=2.5*10**((n/3)-1)
                 
      if (i-1)%3 == 0:
         a1=2.5*10**(((i-1)/3)-1)   
         #a2=7.5*10**(((n-1)/3)-1)
          
      if (i-2)%3 == 0:
         a1=7.5*10**(((i-2)/3)-1)   
         #a2=10*10**(((n-2)/3)-1)
          
      count=count+1   
      t_tA[count]=a1
      if rank==0:print(count,'a1',a1,'t_tA',t_tA[count])
    
   #if rank==0: print('read data')
   
   #f=h5py.File('S100_Rm0.01_Rm1e-6_2e4x2e4.h5','r')
   
   t=0.5*td
   
   #for i in range(1,40):
   #   if t/tA >= t_tA[i-1] and t/tA <= t_tA[i]:
   #      n=i-1
   #      kzmax=f['kz_max'][0,n]
   #      krmax=f['kr_max'][0,n]
   #      if rank==0:print('Accessing',n,t_tA[n],kzmax,krmax)
         
   
   if rank==0:
       rank_pos=np.zeros((nproc,2),dtype=np.float64)               
       for i in range(nprocr):
          for j in range(nprocz):
             m = (i+j)+(i*(nprocz-1))
             rank_pos[m,0]=i
             rank_pos[m,1]=j
      
      
   comm.Bcast(rank_pos,root=0)
   print('rank',rank,rank_pos[rank])      
   Lz=5*10**10*.25;Lr=100
   kr=np.linspace(1e-6,22,num=Npt_kr)
   kz=np.linspace(1e-6,20,num=Npt_kz);  
   
   r=np.linspace(0,20*.25,num=((20-0)*.25)/0.01)
   z=np.linspace((0)*0.25,(1500)*0.25,num=((1500-0)*0.25)/0.025)
   
   """Distributing kz and kr"""
       
   rchunks = [[] for _ in range(nprocr)]
   chunks_r= distribute(r,nprocr,rchunks)
       
   zchunks = [[] for _ in range(nprocz)]
   chunks_z= distribute(z,nprocz,zchunks)
   
   for i in range(nprocr):
       for j in range(nprocz):
             m = (i+j)+(i*(nprocz-1))
             if rank==m:
                r_data=chunks_r[i]
                z_data=chunks_z[j]
   print('rank',rank,'r_shape',np.shape(r_data))
   [KZ,KR,K,l1,l2,l3,l4,uhat0]=universal_parameters(kz,kr)
   
   [uhat_theta,duhat_theta_dt]=u_hat_theta(t)
   if rank==0:print("Inverse transform")
   u=inverse((uhat_theta),kr,kz,r_data,z_data)
   
   u_gathered=comm.gather(u,root=0)
   r_gathered=comm.gather(r_data,root=0)
   z_gathered=comm.gather(z_data,root=0)
   #u_theta=np.zeros((np.size(r),np.size(z)),dtype=np.float64)
   if rank==0:
      print("Gathering data")
      print(rank,np.shape(u_gathered[0]),np.shape(r_gathered),np.shape(z_gathered))
      num_r=0;num_z=0;j=0;p=0;
      f=h5py.File('Els1000_0_5_td.h5','w')
      u_theta=f.create_dataset('u_theta',(np.size(r),np.size(z)),dtype=np.float64)
      uht_theta=f.create_dataset('uhat_theta',(np.size(kr),np.size(kz)),dtype=np.float64) 
      r1=f.create_dataset('r',(np.size(r),1),dtype=np.float64)
      z1=f.create_dataset('z',(np.size(z),1),dtype=np.float64)
      kr1=f.create_dataset('kr',(np.size(kr),1),dtype=np.float64)
      kz1=f.create_dataset('kz',(np.size(kz),1),dtype=np.float64)
      
      r1[0:np.size(r),0]=r[0:np.size(r)]
      kr1[0:np.size(kr),0]=kr[0:np.size(kr)]
      kz1[0:np.size(kz),0]=kz[0:np.size(kz)]
      z1[0:np.size(z),0]=z[0:np.size(z)]
      uht_theta[0:np.size(kr),0:np.size(kz)]=np.real(uhat_theta[0:np.size(kr),0:np.size(kz)])
      
      for i in range(nproc):
         print('proc',i)
         for j in range(np.size(r_gathered[i])-1):
            for k in range(np.size(z_gathered[i])-1):
               u_theta[j + rank_pos[i,0]*(np.size(r_gathered[0])-1) , k + rank_pos[i,1]*(np.size(z_gathered[0])-1) ]=np.real(u_gathered[i][j][k])
              
      f.close()
      print("end")
      #io.savemat('test_1t_Omega.mat',{'u_theta':u_theta,'r':r,'z':z,'tA':tA,'t_Omega':t_Omega,'td':td,'tau':tau,'tnu':tnu,'uhat_theta':uhat_theta,'kr':kr,'kz':kz})
   
      
   
   
