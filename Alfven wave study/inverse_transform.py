from numpy.lib.scimath import sqrt as csqrt
#import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import io
from scipy.special import jn
from mpi4py import MPI
import h5py
import sys
#n=6
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nproc=size
nprocr=8
nprocz=int(size/nprocr)

Npt_kr=3e3
Npt_kz=3.5e3
L=2000;Rm=0.01;d=0.25000;Uo=0.10718*1e6;Pm=1e-6
eta=Uo*d/Rm;mu=4*np.pi*1e-7;rho=7000; 
Va=L*eta/d;to=d/Uo;tA=d/Va;td=d**2/eta;tau=eta/Va**2;
B=Va*csqrt(mu*rho);
nu=Pm*eta
if rank==0:print('tA',tA,'Pm',nu/eta,'Re',Uo*d/nu)

"""************************************************************************************************************************************"""       
def universal_parameters(kz,kr):
   [KZ,KR]=np.meshgrid(kz,kr);
   K=(csqrt(KR**2+KZ**2));
   lamda=(KZ**2/(tau*K**2))+nu*K**2; 
   omg_qs=(csqrt(-1)**0*lamda);
   omg_p=(csqrt((Va*KZ)**2-(((eta-nu)*K**2)**2/4))+csqrt(-1)*((eta+nu)*K**2/2));
   omg_n=(-csqrt((Va*KZ)**2-(((eta-nu)*K**2)**2/4))+csqrt(-1)*((eta+nu)*K**2/2));
   uhat0=(1e6*(KR*d**5/(16*np.pi**1.5))*np.exp(-K**2*d**2/4)); 
   alpha=(csqrt((Va*KZ)**2-(((eta-nu)*K**2)**2/4)));
   eps=(csqrt(1-((4*alpha**2)/((eta+nu)**2*K**4))));

   return KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0 

"""************************************************************************************************************************************ """
def uhatbhat(KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0,t):
   A_factor=(-uhat0/(omg_p-omg_n))*(omg_n-csqrt(-1)*nu*K**2)
   B_factor=(uhat0/(omg_p-omg_n))*(omg_p-csqrt(-1)*nu*K**2)
   
   uhat=A_factor*np.exp(csqrt(-1)*omg_p*t) + B_factor*np.exp(csqrt(-1)*omg_n*t);
   duhat_dt=A_factor*csqrt(-1)*omg_p*np.exp(csqrt(-1)*omg_p*t) + B_factor*csqrt(-1)*omg_n*np.exp(csqrt(-1)*omg_n*t);
   d2uhat_dt2=A_factor*csqrt(-1)**2*omg_p**2*np.exp(csqrt(-1)*omg_p*t) + B_factor*csqrt(-1)**2*omg_n**2*np.exp(csqrt(-1)*omg_n*t);
   #bhat=(duhat_dt+nu*K**2*uhat)*((mu*rho)/(B*csqrt(-1)*KZ));
   bhat=((B*KZ*uhat0)/(omg_p-omg_n))*(np.exp(csqrt(-1)*omg_p*t)-np.exp(csqrt(-1)*omg_n*t))
   dbhat_dt=((B*KZ*uhat0)/(omg_p-omg_n))*(csqrt(-1)*omg_p*np.exp(csqrt(-1)*omg_p*t)-csqrt(-1)*omg_n*np.exp(csqrt(-1)*omg_n*t))
   
   u_jxB        = (csqrt(-1)*KZ*(B/(mu*rho))*bhat*uhat); 
   b_conv       = csqrt(-1)*KZ*B*uhat*bhat;
   b_diff_hat   = -eta*K**2*bhat**2;
   #diff_hat   = -eta*K**2*bhat
   #jxB=(csqrt(-1)*KZ*(B/(mu*rho))*bhat);
   #dbhat_dt=csqrt(-1)*KZ*B*uhat-eta*K**2*bhat  
   #dbhat_dt=((mu*rho)/(csqrt(-1)*KZ*B))*(d2uhat_dt2+(nu*K**2*duhat_dt))   
   omgnbhat=omg_n*bhat*csqrt(-1)
   return uhat,duhat_dt,d2uhat_dt2,bhat,u_jxB,b_conv,b_diff_hat,dbhat_dt,omgnbhat     


"""************************************************************************************************************************************ """

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
   
   t=1.5*tA
   
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
   kr=np.linspace(1e-10,26,num=Npt_kr)
   kz=np.linspace(1e-10,26,num=Npt_kz);  
   
   r=np.linspace(0,20*.25,num=((20-0)*.25)/0.01)
   z=np.linspace((-5)*0.25,(5)*0.25,num=((10-0)*0.25)/0.01)
   
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
   [KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0]=universal_parameters(kz,kr)
   
   [uhat,duhat_dt,d2uhat_dt2,bhat,u_jxB,b_conv,b_diff_hat,dbhat_dt,omgnbhat]=uhatbhat(KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0,t)
   if rank==0:print("Inverse transform")
   u=inverse(uhat,kr,kz,r_data,z_data)
   b=inverse(bhat,kr,kz,r_data,z_data)
   
   u_gathered=comm.gather(u,root=0)
   b_gathered=comm.gather(b,root=0)
   r_gathered=comm.gather(r_data,root=0)
   z_gathered=comm.gather(z_data,root=0)
   if rank==0:
      print("Gathering data")
      print(rank,np.shape(u_gathered[0]),np.shape(r_gathered),np.shape(z_gathered))
      num_r=0;num_z=0;j=0;p=0;
      f=h5py.File('Alfven_1.5_tA.h5','w')
      u_theta=f.create_dataset('u_theta',(np.size(r),np.size(z)),dtype=np.float64)
      b_theta=f.create_dataset('b_theta',(np.size(r),np.size(z)),dtype=np.float64)
      uht_theta=f.create_dataset('uhat',(np.size(kr),np.size(kz)),dtype=np.float64) 
      bht_theta=f.create_dataset('bhat',(np.size(kr),np.size(kz)),dtype=np.float64) 
      r1=f.create_dataset('r',(np.size(r),1),dtype=np.float64)
      z1=f.create_dataset('z',(np.size(z),1),dtype=np.float64)
      kr1=f.create_dataset('kr',(np.size(kr),1),dtype=np.float64)
      kz1=f.create_dataset('kz',(np.size(kz),1),dtype=np.float64)
      
      r1[0:np.size(r),0]=r[0:np.size(r)]
      kr1[0:np.size(kr),0]=kr[0:np.size(kr)]
      kz1[0:np.size(kz),0]=kz[0:np.size(kz)]
      z1[0:np.size(z),0]=z[0:np.size(z)]
      uht_theta[0:np.size(kr),0:np.size(kz)]=np.real(uhat[0:np.size(kr),0:np.size(kz)])
      bht_theta[0:np.size(kr),0:np.size(kz)]=np.imag(bhat[0:np.size(kr),0:np.size(kz)])
      
      for i in range(nproc):
         print('proc',i)
         for j in range(np.size(r_gathered[i])-1):
            for k in range(np.size(z_gathered[i])-1):
               u_theta[j + rank_pos[i,0]*(np.size(r_gathered[0])-1) , k + rank_pos[i,1]*(np.size(z_gathered[0])-1) ]=np.real(u_gathered[i][j][k])
               b_theta[j + rank_pos[i,0]*(np.size(r_gathered[0])-1) , k + rank_pos[i,1]*(np.size(z_gathered[0])-1) ]=np.real(b_gathered[i][j][k])
      f.close()
      print("end")
      #io.savemat('test_1t_Omega.mat',{'u_theta':u_theta,'r':r,'z':z,'tA':tA,'t_Omega':t_Omega,'td':td,'tau':tau,'tnu':tnu,'uhat':uhat,'kr':kr,'kz':kz})
      
   
   
