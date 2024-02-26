%clear
%% Parameters %%
d=0.25000;Rm=0.01;
S=1e7;
Q=1e-2;
P=Inf;
mu=4*pi*1e-7;rho=1000;
murho=sqrt(mu*rho);
g=9.8;
beta=1;
alpha=1;

tA=1/sqrt(g*alpha*abs(beta));
eta=d^2/(Q*S*tA);
kappa=eta/P;
td=d^2/eta;
B0=murho*d*S/td;
Va=B0/murho;
tM=d/Va;
to=td/Rm;
tka=d^2/kappa;
nu=0;
tnu=Inf;

t=.25*tM;n=1;
c=2;
ksmax=25;
kzmax=25;
kswav(1)=25;kzwav(1)=25;timetM(1)=0;
ksmin(1)=1e-5;kzmin(1)=1e-5;

kswav(c)=ksmax;
kzwav(c)=kzmax;
timetM(c)=t/tM;
ksmin(c)=ksmin(1);
kzmin(c)=kzmin(1);


save('kskz.mat','timetM','kswav','kzwav','ksmin','kzmin','-append');
ks =linspace(1e-5,ksmax,2000);
kz =linspace(1e-5,kzmax,2000);
[KS,KZ]=meshgrid(ks,kz);
K=sqrt(KS.^2+KZ.^2);
thetahat0=(d.^3./16./sqrt(2)./pi.^1.5).*exp(-K.^2.*d.^2./8);
omgM   = Va.*KZ;
omgnu  = nu.*K.^2;
omgeta = eta.*K.^2;
omgka  = kappa.*K.^2;
omgA   = sqrt((g.*alpha.*(beta)).*(KS./K).^2);

%% Velocity field solution %%
L      = 9.*omgA.^2.*(2.*omgeta-4.*omgka-omgnu)+2.*omgeta.^3-3.*omgeta.^2.*(omgka+omgnu)-3.*omgeta.*(omgka.^2-4.*omgka.*omgnu+3.*omgM.^2+omgnu.^2)+(omgka+omgnu).*(2 .*omgka.^2-5.*omgka.*omgnu-9.*omgM.^2+2.*omgnu.^2);
M      = 3.*(omgA.^2+omgeta.*(omgka+omgnu)+omgka.*omgnu+omgM.^2)-(omgeta+omgka+omgnu).^2;
l1     = -(sqrt(-1)./12).*(2.^(2./3).*(1-sqrt(-1).*sqrt(3)).*(sqrt(L.^2+4.*M.^3)+L).^(1./3)-((2.*sqrt(-1).*(2).^(1./3).*(sqrt(3)-sqrt(-1)).*M)./((sqrt(L.^2+4.*M.^3)+L).^(1./3)))-4.*(omgeta+omgka+omgnu));
l2     = -(sqrt(-1)./12).*(2.^(2./3).*(1+sqrt(-1).*sqrt(3)).*(sqrt(L.^2+4.*M.^3)+L).^(1./3)+((2.*sqrt(-1).*(2).^(1./3).*(sqrt(3)+sqrt(-1)).*M)./((sqrt(L.^2+4.*M.^3)+L).^(1./3)))-4.*(omgeta+omgka+omgnu));
l3     = (sqrt(-1)./6).*(2.^(2./3).*(sqrt(L.^2+4.*M.^3)+L).^(1./3)-((2.*(2).^(1./3).*M)./((sqrt(L.^2+4.*M.^3)+L).^(1./3)))+2.*(omgeta+omgka+omgnu));
a2 = g.*alpha.*KS.*thetahat0./K.^2;
a3 =-g.*alpha.*KS.*(nu+kappa).*thetahat0;
A =-(a3-sqrt(-1).*a2.*(l2+l3))./((l1-l2).*(l1-l3));
B =-(a3-sqrt(-1).*a2.*(l1+l3))./((l2-l1).*(l2-l3));
C =-(a3-sqrt(-1).*a2.*(l1+l2))./((l3-l1).*(l3-l2));
psihat  = A.*exp(sqrt(-1).*l1.*t) + B.*exp(sqrt(-1).*l2.*t) + C.*exp(sqrt(-1).*l3.*t);
uhats  = -sqrt(-1).*KZ.*psihat;
uhatz  = KS.*psihat;

%% Magnetic field solution %%
L1     = 9.*omgA.^2.*(2.*omgeta-omgka-omgnu)+(omgeta-2.*omgka+omgnu).*(2.*omgeta.^2+omgeta.*(omgka-5.*omgnu)-omgka.^2+omgka.*omgnu-9.*omgM.^2+2.*omgnu.^2);
M1     = 3.*(omgA.^2+omgeta.*(omgka+omgnu)+omgka.*omgnu+omgM.^2)-(omgeta+omgka+omgnu).^2;
l1b    = -(sqrt(-1)./12).*(2.^(2./3).*(1-sqrt(-1).*sqrt(3)).*(sqrt(L1.^2+4.*M1.^3)+L1).^(1./3)-((2.*sqrt(-1).*(2).^(1./3).*(sqrt(3)-sqrt(-1)).*M1)./((sqrt(L1.^2+4.*M1.^3)+L1).^(1./3)))-4.*(omgeta+omgka+omgnu));
l2b    = -(sqrt(-1)./12).*(2.^(2./3).*(1+sqrt(-1).*sqrt(3)).*(sqrt(L1.^2+4.*M1.^3)+L1).^(1./3)+((2.*sqrt(-1).*(2).^(1./3).*(sqrt(3)+sqrt(-1)).*M1)./((sqrt(L1.^2+4.*M1.^3)+L1).^(1./3)))-4.*(omgeta+omgka+omgnu));
l3b    = (sqrt(-1)./6).*(2.^(2./3).*(sqrt(L1.^2+4.*M1.^3)+L1).^(1./3)-((2.*(2).^(1./3).*M1)./((sqrt(L1.^2+4.*M1.^3)+L1).^(1./3)))+2.*(omgeta+omgka+omgnu));
b1=0;b2=0;b3=alpha.*g.*B0.*sqrt(-1).*KZ.*KS.*thetahat0./K.^2;
P  =-(b3-sqrt(-1).*b2.*(l2+l3))./((l1-l2).*(l1-l3));
Q  =-(b3-sqrt(-1).*b2.*(l1+l3))./((l2-l1).*(l2-l3));
R  =-(b3-sqrt(-1).*b2.*(l1+l2))./((l3-l1).*(l3-l2));
Phihat   = P.*exp(sqrt(-1).*l1b.*t) + Q.*exp(sqrt(-1).*l2b.*t) + R.*exp(sqrt(-1).*l3b.*t);
bhats   = -sqrt(-1).*KZ.*Phihat;
bhatz   = KS.*Phihat;

figure;contourf(ks,kz,real(uhatz),20,'linestyle','none');colorbar;
title(sprintf('uz, t=%0.2f tM,ks=[%0.2d,%0.2d],kz=[%0.2d,%0.2d]',t/tM,ks(1),ks(numel(ks)),kz(1),kz(numel(kz))))
colormap(bwr)
savefig(sprintf('%0.0d_uz_t%0.2etM.fig',c,t/tM))

break
%% Approximate frequencies from effective wavenumbers %%
%%% ksus=||ks*us||/||us|| similarly for other wavenumbers%%%

g=9.8;
ksus=sqrt(ks_us_tot./Eus_tot);
kzus=sqrt(kz_us_tot./Eus_tot);
kus=sqrt(k_us_tot./Eus_tot);

ksuz=sqrt(ks_uz_tot./Euz_tot);
kzuz=sqrt(kz_uz_tot./Euz_tot);
kuz=sqrt(k_uz_tot./Euz_tot);

figure
loglog(time_tot/tM,Va*kzus,'r');hold on
loglog(time_tot/tM,1e0*sqrt(g*abs(1))*(ksus./kus),'b');hold on
loglog(time_tot/tM,eta*kus.^2./2,'k');hold on
loglog(time_tot/tM,kappa*kus.^2./2,'m');hold on

figure
loglog(time_tot/tM,Va*kzuz,'r');hold on
loglog(time_tot/tM,sqrt(g*abs(1))*(ksuz./kuz),'k');hold on
loglog(time_tot/tM,eta*kuz.^2./2,'b');hold on
loglog(time_tot/tM,kappa*kuz.^2./2,'m');hold on

omgMa   = Va.*kzus;
omgnua  = nu.*kus.^2;
omgetaa = eta.*kus.^2;
omgkaa  = kappa.*kus.^2;
omgAa   = 1e0*sqrt((g.*alpha.*(beta)).*(ksus./kus).^2);
L      = 9.*omgAa.^2.*(2.*omgetaa-4.*omgkaa-omgnua)+2.*omgetaa.^3-3.*omgetaa.^2.*(omgkaa+omgnua)-3.*omgetaa.*(omgkaa.^2-4.*omgkaa.*omgnua+3.*omgMa.^2+omgnua.^2)+(omgkaa+omgnua).*(2 .*omgkaa.^2-5.*omgkaa.*omgnua-9.*omgMa.^2+2.*omgnua.^2);
M      = 3.*(omgAa.^2+omgetaa.*(omgkaa+omgnua)+omgkaa.*omgnua+omgMa.^2)-(omgetaa+omgkaa+omgnua).^2;
l1     = -(sqrt(-1)./12).*(2.^(2./3).*(1-sqrt(-1).*sqrt(3)).*(sqrt(L.^2+4.*M.^3)+L).^(1./3)-((2.*sqrt(-1).*(2).^(1./3).*(sqrt(3)-sqrt(-1)).*M)./((sqrt(L.^2+4.*M.^3)+L).^(1./3)))-4.*(omgetaa+omgkaa+omgnua));
l2     = -(sqrt(-1)./12).*(2.^(2./3).*(1+sqrt(-1).*sqrt(3)).*(sqrt(L.^2+4.*M.^3)+L).^(1./3)+((2.*sqrt(-1).*(2).^(1./3).*(sqrt(3)+sqrt(-1)).*M)./((sqrt(L.^2+4.*M.^3)+L).^(1./3)))-4.*(omgetaa+omgkaa+omgnua));
l3     = (sqrt(-1)./6).*(2.^(2./3).*(sqrt(L.^2+4.*M.^3)+L).^(1./3)-((2.*(2).^(1./3).*M)./((sqrt(L.^2+4.*M.^3)+L).^(1./3)))+2.*(omgetaa+omgkaa+omgnua));



%% contour plots %%
clear
figure
filename='psi_S2e3_Q1e-2_P1e3_100tM.h5';
u=h5read(filename,'/psi');
r=h5read(filename,'/r');
z=h5read(filename,'/z');
ymax=100;%max(z)/.25;
ymin=-100;%min(z)/.25;
nlines=24;
num=numel(z);
const=1e0;

%# centimeters units
X = 21.0;                  %# A4 paper size
Y = 29.7;                  %# A4 paper size
xMargin = 1;               %# left/right margins from page borders
yMargin = 0;               %# bottom/top margins from page borders
xSize = X - 2*xMargin;     %# figure size on paper (widht & hieght)
ySize = Y - 2*yMargin;     %# figure size on paper (widht & hieght)
hFig=gcf;
set(hFig, 'Units','centimeters', 'Position',[0 0 xSize/3.1 ySize/2.75]/1)
movegui(hFig, 'center')
%# figure size printed on paper
set(hFig, 'PaperUnits','centimeters')
set(hFig, 'PaperSize',[X Y])
set(hFig, 'PaperPosition',[xMargin yMargin xSize ySize])
set(hFig, 'PaperOrientation','portrait')
contourf(r/.25,z(1:num)/.25,const*u(1:num,:),nlines,'linewidth',.001)
set(gca,'DataAspectratio',[(100/20) (10/1000)*(ymax-ymin) 1])
set(gcf,'Color','w')
%set(gca,'XTick',[0,4,8])%,10,12,14,16,18,20])
set(gca,'pos',[0.0102    0.1100    0.7750    0.8150])
colorbar('position',[.58 .11 .05 .815])
%caxis([min(min(u(1:num,:))) max(max(u(1:num,:)))])
colormap(bwr)
set(gca,'XLim',[0 100])
set(gca,'YLim',[ymin ymax])
%title('$\Lambda/S=0.1, \ Ro/Rm=1\times 10^{-3}$','interpreter','latex','fontsize',12)
set(gca,'fontsize',12,'fontname','times')
text(9.8,114,'x10^{-4}','interpreter','tex','fontsize',12,'fontname','times','Color','k')

New_folder='../print2eps/private';
current_folder=pwd;
cd(New_folder)
filename='psi_10M.eps';%sprintf('uhatz_%0.0dtd_2.eps',100000);
print2eps(filename,gcf)
movefile(filename,current_folder)
cd(current_folder)