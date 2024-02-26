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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Npt_kr=1e3
Npt_kz=1e4

nprocr=4
nprocz=int(size/nprocr)
if rank==0:print('np_r',nprocr,'np_z',nprocz)
mu=4*np.pi*1e-7;
rho=7000;
murho=csqrt(mu*rho);

nproc=size #number of processes

L=1000;Rm=0.01;d=0.25000;Uo=0.10718*1e6;Pm=0
eta=Uo*d/Rm;mu=4*np.pi*1e-7;rho=7000; 
Va=L*eta/d;to=d/Uo;tA=d/Va;td=d**2/eta;tau=eta/Va**2;
B=Va*csqrt(mu*rho);
nu=0#Pm*eta
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
def universal_parameters(kz,kr,scale_z,scale_r):
   kr1=np.zeros((np.size(kr)),dtype=np.float64)
   kz1=np.zeros((np.size(kz)),dtype=np.float64)
   for i in range(np.size(kr)):
      kr1[i]=kr[i]*scale_r
   
   for i in range(np.size(kz)):
      kz1[i]=kz[i]*scale_z
   
   
   [KZ,KR]=np.meshgrid(kz1,kr1);
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
def uhatbhat_qs(KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0,t):
   uhat_qs=(uhat0)*(np.exp(-omg_qs*t));
   duhat_dt_qs=-omg_qs*uhat_qs;
   d2uhat_dt2_qs=-omg_qs*duhat_dt_qs;
   bhat_qs=(B*csqrt(-1)*KZ*uhat_qs)/(eta*K**2);
   dbhat_dt_qs=(B*csqrt(-1)*KZ*duhat_dt_qs)/(eta*K**2)
   return uhat_qs,duhat_dt_qs,d2uhat_dt2_qs,bhat_qs,dbhat_dt_qs     

"""************************************************************************************************************************************"""              
def kzmax(k_r,k_z,u,name,rank_pos,criteria):
   kz_loc=az[n]
   
   for l in range(nproc):
      umax[l]=np.max((u[l][:,:]))
   max_umax=np.where(umax==np.max(umax))   
   print("The maximum value of",name,"is in proc:",max_umax[0][0])
   
   
   max_umax_loc=np.where(u[max_umax[0][0]][:,:]==np.max(umax))   
   max_umax_r_loc=k_r[max_umax[0][0]][max_umax_loc[0][0]]
   max_umax_z_loc=k_z[max_umax[0][0]][max_umax_loc[1][0]]
   print("In proc",max_umax[0][0],"overall maxima of ",name,"is at location: (kr,kz)",max_umax_r_loc,max_umax_z_loc)
   #print("The ratio of maxima with overall maxima is :",umax/np.max(umax))
   
   """Finding the ratio wrt overall umax of values at the r location"""       
   count=1
   for i in range(int(max_umax[0][0]),int(max_umax[0][0]+nprocz-rank_pos[max_umax[0][0],1])):
      #print("rank",i,"rank_pos,i=",rank_pos[max_umax[0][0],0],"rank_pos,j=",rank_pos[max_umax[0][0],1])
      if count==0:
         break
      else:
         for j in range(np.size(k_z[i][:])):
            if k_z[i][j] > k_z[max_umax[0][0]][max_umax_loc[1][0]]:
               #print("proc rank",i,"ratio at kz",k_z[i][j],"is :",np.abs(u[i][max_umax_loc[0][0],j]/np.max(umax)))
               if np.abs(u[i][max_umax_loc[0][0],j]/np.max(umax)**1)  < criteria:
                  #u_loc=np.where(u[i][:,:]==np.max(u[i][max_umax_loc[0][0],:]))
                  print("kzmax is at proc",i,"location:",k_r[i][max_umax_loc[0][0]],k_z[i][j],"ratio value:",np.abs(u[i][max_umax_loc[0][0],j]/np.max(umax)))
                  print(" ")
                  kz_loc=k_z[i][j]
                  count=count-1
                  break
   return kz_loc
"""************************************************************************************************************************************ """
def krmax(k_r,k_z,u,name,rank_pos,criteria):
   kr_loc=ar[n]
   if name == 'dbhat_dt':
      for l in range(nproc):
         u[l][:,:]=-1*u[l][:,:]   
   for l in range(nproc):
      umax[l]=np.max((u[l][:,:]))
   max_umax=np.where(umax==np.max(umax))   
   print("The maximum value of",name,"is in proc:",max_umax[0][0])

   
   max_umax_loc=np.where(u[max_umax[0][0]][:,:]==np.max(umax))   
   max_umax_r_loc=k_r[max_umax[0][0]][max_umax_loc[0][0]]
   max_umax_z_loc=k_z[max_umax[0][0]][max_umax_loc[1][0]]
   print("In proc",max_umax[0][0],"overall maxima of ",name,"is at location: (kr,kz)",max_umax_r_loc,max_umax_z_loc)
   #print("The ratio of maxima with overall maxima is :",umax/np.max(umax))
   
   """Finding the ratio wrt overall umax of values at the r location"""       
   count=1
   for i in range(int(max_umax[0][0]),int(max_umax[0][0]+(nprocr-1-rank_pos[max_umax[0][0],0])*nprocz+1),nprocz):
      #print("rank",i,"rank_pos,i=",rank_pos[max_umax[0][0],0],"rank_pos,j=",rank_pos[max_umax[0][0],1])
      if count==0:
         break
      else:
         for j in range(np.size(k_r[i][:])):
            if k_r[i][j] > k_r[max_umax[0][0]][max_umax_loc[0][0]]:
               #print("proc rank",i,"ratio at kz",k_z[i][j],"is :",np.abs(u[i][max_umax_loc[0][0],j]/np.max(umax)))
               if np.abs(u[i][j,max_umax_loc[1][0]]/np.max(umax)**1)  < criteria:
                  #u_loc=np.where(u[i][:,:]==np.max(u[i][max_umax_loc[0][0],:]))
                  print("krmax is at proc",i,"location:",k_r[i][j],k_z[i][max_umax_loc[1][0]],"ratio value:",np.abs(u[i][j,max_umax_loc[1][0]]/np.max(umax)))
                  print(" ")
                  kr_loc=k_r[i][j]
                  count=count-1
                  break
   return kr_loc
"""************************************************************************************************************************************"""              
def kzkrmax(uhat,kr_data,kz_data,name,criteria):
             max_u=np.zeros((nproc,3),dtype=np.float64)
             max_proc=np.ones((1))
             max_kr_pos=np.zeros((1))
             max_kz_pos=np.zeros((1))
             
             for p in range(nproc):
                if rank==p:
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
                      
                         
             #az[n+1]=kz_max[0]
             #ar[n+1]=kr_max[0]
             #if rank==0:print('kr2',kr_max[0])
             
             return kr_max[0],kz_max[0] 

"""************************************************************************************************************************************"""
if __name__=='__main__':
    pdiff=0
    rank_pos=np.zeros((nproc,2))
    az=np.zeros(41)
    ar=np.zeros(41)
    az[0]=50
    ar[0]=50
    t_tA=np.zeros((41),dtype=np.float64)
    count=0;
    t_tA[0]=0; 
    for n in range(40):
       
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
    
    if rank==0:
          f=h5py.File('test.h5','w')
          #bhat_tot=f.create_dataset('bhat',(np.size(kr),1e4,41),dtype=np.float64)
          #dbdt_tot=f.create_dataset('dbdt',(np.size(kr),1e4,41),dtype=np.float64)
          #diffhat_tot=f.create_dataset('diff_hat',(np.size(kr),1e4,41),dtype=np.float64)
          #bconv_tot=f.create_dataset('b_conv',(np.size(kr),1e4,41),dtype=np.float64)
          #kz_tot=f.create_dataset('kz',(1e4,41),dtype=np.float64)
          #kr_tot=f.create_dataset('kr',(np.size(kr),41),dtype=np.float64)
          time_tot=f.create_dataset('time',(1,41),dtype=np.float64)
          kz_max=f.create_dataset('kz_max',(1,41),dtype=np.float64)
          kr_max=f.create_dataset('kr_max',(1,41),dtype=np.float64)
   
    
    integral_vars=['KE','time','ME','lorentz_work','conv_work','Joule_dissip','dudt','dbdt','diff','kz_u','kz_b','kr_u','kr_b','omgbn','omgbp','omgun','omgup',
'kz_u2','kz_b2','kr_u2','kr_b2','k_u','k_b','k_u2','k_b2','k2_u2','k2_b2','kz2_u2','kz2_b2','kr2_u2','kr2_b2']

    variables=['KE_tot','time_tot','ME_tot','lorentz_work_tot','conv_work_tot','Joule_dissip_tot','dudt_tot','dbdt_tot','diff_tot','kz_u_tot','kz_b_tot','kr_u_tot','kr_b_tot','omgbn_tot','omgbp_tot','omgun_tot','omgup_tot',
'kz_u2_tot','kz_b2_tot','kr_u2_tot','kr_b2_tot','k_u_tot','k_b_tot','k_u2_tot','k_b2_tot','k2_u2_tot','k2_b2_tot','kz2_u2_tot','kz2_b2_tot','kr2_u2_tot','kr2_b2_tot']       
    integral_qty=[[] for _ in range(np.size(integral_vars))]
    
    if rank==0:
       rank_pos=np.zeros((nproc,2),dtype=np.float64)               
       for i in range(nprocr):
          for j in range(nprocz):
             m = (i+j)+(i*(nprocz-1))
             rank_pos[m,0]=i
             rank_pos[m,1]=j
       #for i in range(nproc):
       #   print("rank_pos",i,rank_pos[i])
    comm.Bcast(rank_pos,root=0)
    print('rank',rank,rank_pos[rank])      
    #sys.exit()
    for n in range(2):
       scale_r=ar[n]/ar[0]
       scale_z=az[n]/az[0]
       
       if n < 25:
          criteria=1e-10
       else:
          criteria=1e-35
       t=np.linspace(t_tA[n]*tA,t_tA[n+1]*tA,num=11)
       
       if rank==0:print('n=',n)   
       Lr=100
       AR=(ar[n]/scale_r)*(Lr/(2*np.pi)); 
       kr=(2*np.pi/Lr)*np.linspace(0,AR,num=Npt_kr)
       if rank==0:print("krmin",min(kr),"krmax",max(kr),scale_r)

       """kz vector"""
       Lz=5*10**10*.25
       AZ=(az[n]/scale_z)*(Lz/(2*np.pi));
       kz=(2*np.pi/Lz)*np.linspace(1e-9/scale_z,AZ,num=Npt_kz); 
       if rank==0:print("kzmin",min(kz),"kzmax",max(kz),scale_z)
       
       Nkz=np.size(kz);
       
       """ Distributing kz and kr"""
       
       rchunks = [[] for _ in range(nprocr)]
       chunks_r = distribute(kr,nprocr,rchunks)
       
       zchunks = [[] for _ in range(nprocz)]
       chunks_z=distribute(kz,nprocz,zchunks)
       
      
       for i in range(nprocr):
          for j in range(nprocz):
                m = (i+j)+(i*(nprocz-1))
                if rank==m:
                   kr_data=chunks_r[i]
                   kz_data=chunks_z[j]
      
       #print("rank",rank,"min(kr_data)",np.min(kr_data),"max(kr_data)",np.max(kr_data)) 
       #print("rank",rank,"min(kz_data)",np.min(kz_data),"max(kz_data)",np.max(kz_data)) 
       [KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0]=universal_parameters(kz_data,kr_data,scale_z,scale_r)
       for i in range(np.size(t)):
          if rank==0:print("computing integrals at t/tA=",t[i]/tA)
          [uhat,duhat_dt,d2uhat_dt2,bhat,u_jxB,b_conv,b_diff_hat,dbhat_dt,omgnbhat]=uhatbhat(KZ,KR,K,lamda,omg_qs,omg_p,omg_n,uhat0,t[i])
                    
          integral_qty[0].append(32*np.pi**4*np.trapz(kr_data*np.trapz(np.real(uhat)*np.real(uhat),kz_data),kr_data)/(scale_r*scale_z))
          integral_qty[1].append(t[i])
          integral_qty[2].append(32*np.pi**4*np.trapz(kr_data*np.trapz(np.imag(bhat)*np.imag(bhat),kz_data),kr_data)/(scale_r*scale_z))
          integral_qty[3].append(32*np.pi**4*np.trapz(kr_data*np.trapz(u_jxB,kz_data),kr_data)/(scale_r*scale_z))
          integral_qty[4].append(32*np.pi**4*np.trapz(kr_data*np.trapz(b_conv/murho**2,kz_data),kr_data)/(scale_r*scale_z))
          integral_qty[5].append(32*np.pi**4*np.trapz(kr_data*np.trapz(b_diff_hat/murho**2,kz_data),kr_data)/(scale_r*scale_z))
          integral_qty[6].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(duhat_dt)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[7].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(dbhat_dt)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[8].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(-eta*K**2*bhat)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[9].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KZ*uhat)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[10].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KZ*bhat)**2,kz_data),kr_data))/(scale_r*scale_z))#/(32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(bhat)**2,kz_data),kr_data)))#sqrt
          integral_qty[11].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KR*uhat)**2,kz_data),kr_data))/(scale_r*scale_z))#/(32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(uhat)**2,kz_data),kr_data)))#sqrt
          integral_qty[12].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KR*bhat)**2,kz_data),kr_data))/(scale_r*scale_z))#/(32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(bhat)**2,kz_data),kr_data)))#sqrt
          integral_qty[13].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(csqrt(-1)*bhat*omg_n)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[14].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(csqrt(-1)*bhat*omg_p)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[15].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(csqrt(-1)*uhat*omg_n)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[16].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(csqrt(-1)*uhat*omg_p)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          
          integral_qty[17].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KZ*uhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[18].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KZ*bhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[19].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KR*uhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[20].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KR*bhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
      
          integral_qty[21].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(K*uhat)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[22].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(K*bhat)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          
          integral_qty[23].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(K*uhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[24].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(K*bhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt 
          
          integral_qty[25].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(K**2*uhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[26].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(K**2*bhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt 
          integral_qty[27].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KZ**2*uhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[28].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KZ**2*bhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[29].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KR**2*uhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          integral_qty[30].append((32*np.pi**4*np.trapz(kr_data*np.trapz(np.abs(KR**2*bhat**2)**2,kz_data),kr_data))/(scale_r*scale_z))#sqrt
          
          
          if  i==(np.size(t)-1):
             """ Finding I_old"""
             I_old=(comm.gather(integral_qty[7], root=0))
             time_old=(comm.gather(integral_qty[1], root=0))
             
             if rank==0:
                print(np.shape(I_old),(n+1)*(i+1),time_old[0][(n+1)*(i+1)-1]/tA)
                dbdt_old=0
                for j in range(nproc):    
                   dbdt_old=dbdt_old+I_old[j][(n+1)*(i+1)-1]
                print('old',dbdt_old,time_old[0][(n+1)*(i+1)-1]/tA)
 
             
             """ Storing current max(kr,kz) values"""
             if rank==0:   
                kz_max[0,n]=az[n]
                kr_max[0,n]=ar[n]
                time_tot[0,n]=t[i]
             

             """Finding new kz and kr max : new method """
             max_kr_array=np.zeros((3))
             max_kz_array=np.zeros((3))
             
             u_kr_max=0
             u_kz_max=0
             
             b_kr_max=0
             b_kz_max=0
             
             dbdt_kr_max=0
             dbdt_kz_max=0
             
             [u_kr_max,u_kz_max]=kzkrmax(np.real(uhat),kr_data,kz_data,'uhat',criteria)
             if rank==0:print('u_kr_max',u_kr_max,'u_kz_max',u_kz_max,'criteria',criteria)
             
             if n > 0:
                [b_kr_max,b_kz_max]=kzkrmax(np.imag(bhat),kr_data,kz_data,'bhat',criteria)
                if rank==0:print('b_kr_max',b_kr_max,'b_kz_max',b_kz_max)
             
                [dbdt_kr_max,dbdt_kz_max]=kzkrmax(np.imag(dbhat_dt),kr_data,kz_data,'dbhat_dt',criteria)
                if rank==0:print('dbdt_kr_max',dbdt_kr_max,'dbdt_kz_max',dbdt_kz_max)
             
             
             max_kr_array[0]=u_kr_max
             max_kr_array[1]=b_kr_max     
             max_kr_array[2]=dbdt_kr_max 
             
             max_kz_array[0]=u_kz_max
             max_kz_array[1]=b_kz_max     
             max_kz_array[2]=dbdt_kz_max 
             
             if rank==0:print(np.max(max_kz_array[:])*scale_z,np.max(max_kz_array[:]))
             if n==1:sys.exit()
             ar[n+1]=np.max(max_kr_array[:])*scale_r
             az[n+1]=np.max(max_kz_array[:])*scale_z
             scale_r=ar[n+1]/ar[0]
             scale_z=az[n+1]/az[0]
             
             if rank==0:
                print('kr_max',ar)
                print('kz_max',az)
                print('scale_r',scale_r,'scale_z',scale_z)
                
             Lr=100;Lz=5*10**10*.25
             AR=(ar[n+1]/scale_r)*(Lr/(2*np.pi)); 
             AZ=(az[n+1]/scale_z)*(Lz/(2*np.pi));
             
             kr_new = (2*np.pi/Lr)*np.linspace(0,AR,num=Npt_kr)
             kz_new = (2*np.pi/Lz)*np.linspace(1e-9/scale_z,AZ,num=Npt_kz); 
             
             
             rchunks_new = [[] for _ in range(nprocr)]
             chunks_r_new = distribute(kr_new,nprocr,rchunks_new)
             zchunks_new = [[] for _ in range(nprocz)]
             chunks_z_new=distribute(kz_new,nprocz,zchunks_new)
             for l in range(nprocr):
                for j in range(nprocz):
                   m = (l+j)+(l*(nprocz-1))
                   if rank==m:
                      kr_data_new=chunks_r_new[l]
                      kz_data_new=chunks_z_new[j]
             
             #print('rank',rank,'kz_data_new',min(kz_data_new),max(kz_data_new),np.shape(kz_data_new))
             #if rank==0:print(" ")
             [KZ_new,KR_new,K_new,lamda_new,omg_qs_new,omg_p_new,omg_n_new,uhat0_new]=universal_parameters(kz_data_new,kr_data_new,scale_z,scale_r)
             
             #print('rank',rank,'KZ_new',np.shape(KZ_new),'KR_new',np.shape(KR_new))
             #sys.exit()
             
             [uhat_new,duhat_dt_new,d2uhat_dt2_new,bhat_new,u_jxB_new,b_conv_new,b_diff_hat_new,dbhat_dt_new,omgnbhat_new]=uhatbhat(KZ_new,KR_new,K_new,lamda_new,omg_qs_new,omg_p_new,omg_n_new,uhat0_new,t[i])
             
             #print('rank',rank,'kz_data_new',np.shape(kz_data_new),'kr_data_new',np.shape(kr_data_new),'dbdt_new',np.shape(dbhat_dt_new))
             
             
             I_dbdt_new=(32*np.pi**4*np.trapz(kr_data_new*np.trapz(np.abs(dbhat_dt_new)**2,kz_data_new),kr_data_new)/(scale_r*scale_r))       
             
             """ Finding I_new"""
             I_new=(comm.gather(I_dbdt_new, root=0))
             
             pdiff=np.zeros((1))
             if rank==0:
                print(np.shape(I_new))
                dbdt_new=0
                for j in range(nproc):    
                   dbdt_new=dbdt_new+I_new[j]
                print('new',dbdt_new,t[i]/tA)
                print('% diff',100*np.abs((dbdt_new-dbdt_old)/(dbdt_old))) 
                pdiff=100*np.abs((dbdt_new-dbdt_old)/(dbdt_old))
             comm.Bcast(pdiff,root=0) 
             count=1
             krloc = np.where(kr[0:Npt_kr] == ar[n+1])
             kzloc = np.where(kz[0:Npt_kz] == az[n+1])          
             if rank==0:print(krloc,'kr',kr[krloc[0]],'ar',ar[n+1])
             if rank==0:print(kzloc,'kz',kz[kzloc[0]],'az',az[n+1]) 

             while pdiff > 1e-3:
                   if (krloc[0]+count) > Npt_kr-1:
                      if rank==0:print("Entered Npt_kr",krloc[0]+count)
                      ar[n+1]=ar[n]#kr[krloc[0]]
                   else:
                      if rank==0:print(krloc[0]+count,"Entered Npt_kr")
                      ar[n+1]=kr[krloc[0]+count]
                      
                   if (kzloc[0]+count) > Npt_kz-1:   
                      if rank==0:print("Entered Npt_kz",kzloc[0]+count)
                      az[n+1]=az[n]#kz[kzloc[0]]
                   else:
                      if rank==0:print(kzloc[0]+count,"Entered Npt_kz")
                      az[n+1]=kz[kzloc[0]+count]
                   if rank==0:print('kr_max_new',ar[n+1],'kz_max_new',az[n+1])
                   if az[n+1] <= kz[Npt_kz-1] and ar[n+1] <= kr[Npt_kr-1]:
                      Lr=100;Lz=5*10**10*.25
                      AR=ar[n+1]*(Lr/(2*np.pi)); 
                      AZ=az[n+1]*(Lz/(2*np.pi));
                      kr_new = (2*np.pi/Lr)*np.linspace(0,AR,num=Npt_kr)
                      kz_new = (2*np.pi/Lz)*np.linspace(1e-8,AZ,num=Npt_kz); 
                      rchunks_new = [[] for _ in range(nprocr)]
                      chunks_r_new = distribute(kr_new,nprocr,rchunks_new)
                      zchunks_new = [[] for _ in range(nprocz)]
                      chunks_z_new=distribute(kz_new,nprocz,zchunks_new)
                      for l in range(nprocr):
                         for j in range(nprocz):
                            m = (l+j)+(l*(nprocz-1))
                            if rank==m:
                               kr_data_new=chunks_r_new[l]
                               kz_data_new=chunks_z_new[j]
                      [KZ_new,KR_new,K_new,lamda_new,omg_qs_new,omg_p_new,omg_n_new,uhat0_new]=universal_parameters(kz_data_new,kr_data_new,scale_z,scale_r)
                      [uhat_new,duhat_dt_new,d2uhat_dt2_new,bhat_new,u_jxB_new,b_conv_new,b_diff_hat_new,dbhat_dt_new,omgnbhat_new]=uhatbhat(KZ_new,KR_new,K_new,lamda_new,omg_qs_new,omg_p_new,omg_n_new,uhat0_new,t[i])
                      I_dbdt_new=(32*np.pi**4*np.trapz(kr_data_new*np.trapz(np.abs(dbhat_dt_new)**2,kz_data_new),kr_data_new))       
                      I_new=(comm.gather(I_dbdt_new, root=0))
                      
                      if rank==0:
                         #print(np.shape(I_new))
                         dbdt_new=0
                         for j in range(nproc):    
                            dbdt_new=dbdt_new+I_new[j]
                         print('% diff=',100*np.abs((dbdt_new-dbdt_old)/(dbdt_old)),'for kr_max=',ar[n+1],'and kz_max=',az[n+1]) 
                         pdiff=100*np.abs((dbdt_new-dbdt_old)/(dbdt_old))
                      count=count+100
                      comm.Bcast(pdiff,root=0) 
                   
                   else:
                   
                      if rank==0:print("Entered")
                      ar[n+1]=ar[n]#kr[krloc[0]]
                      az[n+1]=az[n]#kz[kzloc[0]]
                      scale_r=ar[n+1]/ar[0]
                      scale_z=az[n+1]/az[0]
                      Lr=100;Lz=5*10**10*.25
                      AR=(ar[n+1]/scale_r)*(Lr/(2*np.pi)); 
                      AZ=(az[n+1]/scale_z)*(Lz/(2*np.pi));
                      kr_new = (2*np.pi/Lr)*np.linspace(0,AR,num=Npt_kr)
                      kz_new = (2*np.pi/Lz)*np.linspace(1e-9/scale_z,AZ,num=Npt_kz); 
                      rchunks_new = [[] for _ in range(nprocr)]
                      chunks_r_new = distribute(kr_new,nprocr,rchunks_new)
                      zchunks_new = [[] for _ in range(nprocz)]
                      chunks_z_new=distribute(kz_new,nprocz,zchunks_new)
                      for l in range(nprocr):
                         for j in range(nprocz):
                            m = (l+j)+(l*(nprocz-1))
                            if rank==m:
                               kr_data_new=chunks_r_new[l]
                               kz_data_new=chunks_z_new[j]
                      [KZ_new,KR_new,K_new,lamda_new,omg_qs_new,omg_p_new,omg_n_new,uhat0_new]=universal_parameters(kz_data_new,kr_data_new,scale_r,scale_z)
                      [uhat_new,duhat_dt_new,d2uhat_dt2_new,bhat_new,u_jxB_new,b_conv_new,b_diff_hat_new,dbhat_dt_new,omgnbhat_new]=uhatbhat(KZ_new,KR_new,K_new,lamda_new,omg_qs_new,omg_p_new,omg_n_new,uhat0_new,t[i])
                      I_dbdt_new=(32*np.pi**4*np.trapz(kr_data_new*np.trapz(np.abs(dbhat_dt_new)**2,kz_data_new),kr_data_new))       
                      I_new=(comm.gather(I_dbdt_new, root=0))
                      if rank==0:
                         #print(np.shape(I_new))
                         dbdt_new=0
                         for j in range(nproc):    
                            dbdt_new=dbdt_new+I_new[j]
                         print('% diff=',100*np.abs((dbdt_new-dbdt_old)/(dbdt_old)),'for kr_max=',ar[n+1],'and kz_max=',az[n+1]) 
                         pdiff=100*np.abs((dbdt_new-dbdt_old)/(dbdt_old))
                      count=count+100
                      comm.Bcast(pdiff,root=0)
                      break 
                          
       gathered_quantity = [[] for _ in range(np.size(integral_vars))]
       for i in range(np.size(integral_vars)):    
          gathered_quantity[i]=(comm.gather(integral_qty[i], root=0))
       gathered_qty=comm.gather(integral_qty,root=0)                 
       if rank==0:
          #print("Gathered quantity shape",np.shape(gathered_quantity))
          #
          print("Gathered qty shape",np.shape(gathered_qty))
          print("Gathered qty shape",np.size(gathered_qty[0][0][:]))
     
    """*********************************************************************************************************************************"""
    if rank==0:
       quantity=[ [  ]  ,  [[  ] for _ in range(np.size(variables))]  ]
       for var in variables:
          quantity[0].append(var)
       count=0
       for k in range(np.size(variables)):
          if k==1:
             for i in range(np.size(gathered_quantity[0][0][:])):
                quantity[1][k].append(gathered_quantity[k][0][i])
             
          if k !=1 and k <= 5:
             for i in range(np.size(gathered_quantity[0][0][:])):
                summation=0
                for j in range(nproc):    
                   summation=summation+gathered_quantity[k][j][i]
                quantity[1][k].append(summation)

          if k >= 6:
             for i in range(np.size(gathered_quantity[0][0][:])):
                summation=0
                for j in range(nproc):    
                   summation=summation+gathered_quantity[k][j][i]
                quantity[1][k].append(np.sqrt(summation/(scale_r*scale_z)))

       print('saving gathered data')
       io.savemat('test.mat', mdict={quantity[0][i]:quantity[1][i][:] for i in range(np.size(quantity[0]))},appendmat=True)      
