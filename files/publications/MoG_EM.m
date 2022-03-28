%%%%%%%%%%%%%%%%%
%%% Written by Omar Alhussein, 28th July 2015.
%%% Generation and Approximation of Fading channels using MoG model
%%% References = 
%%%%%%%%%%%%%%%%%
%% Global Variables
snr_dB = 0 ; % in dB
snr = 10.^(snr_dB./10); %linear
%% Generate a NL random variate and Validate its PDF.
N = 1e6 ; % # of samples
m = 3; % fading parameter
theta = 1./m ; 
data1 = sqrt(gamrnd(1,theta,1,N));  % Since it is sqrt(). this is amplitude.
[rpdf,h1] = hist(data1,100); 
rpdf = rpdf/(h1(2)-h1(1))/length(data1) ; %normalized amplitude pdf.

datax = data1.^2 ; % This is the instantaneous snr RV.
[rpdfx,hx] = hist(datax,100); 
rpdfx = rpdfx/(hx(2)-hx(1))/length(datax) ; %normalized snr pdf.

figure;
plot(h1,rpdf,'b*-'); hold on ; 
plot(hx,rpdfx,'ko-');
legend('empirical amplitude PDF','empirical SNR PDF');
% AoF = var(datax)./(mean(datax))^2
%% Determination of the optimal number of components
                                    tic
N = 1e4;
data = data1 ;
C = 10 ; % upper number of components
e = 1e-6 ; % error of stopping criterion
num_iter = 1e4; % number of iterations.
options = statset('MaxIter',num_iter,'TolFun',e,'Display','final');
BIC = zeros(1,C);
obj = cell(1,C);
for k = 1:C
    obj{k} = gmdistribution.fit(data',k,'Options',options);
    BIC(k)= obj{k}.BIC; %one can also employ AIC as well.
end
[minBIC,numComponents] = min(BIC);
disp('minimum number of components is: ') ; 
numComponents
                                    toc;
figure; 
semilogy(1:C,BIC-min(BIC)+1);
title('BIC Behavior');
xlabel('Number of Components');
ylabel('BIC');
%% Fitting MoG parameters using EM code
                                    tic
N=1e6;
data = data1 ; 
C = numComponents; %number of components
e = 1e-6 ; %tolerance of stopping criterion
options = statset('MaxIter',num_iter,'TolFun',e,'Display','final');
obj = gmdistribution.fit(data',C,'Options',options);
BIC = obj.BIC;

w_est = obj.PComponents ;
mu_est = obj.mu;
for i =1:C
sigma_est(i) = sqrt(obj.Sigma(:,:,i)); 
end
toc;

x = h1 ; 
%.% Building the GMM Model
fhat = zeros(size(x));
for i = 1:C
fhat = fhat + w_est(i) * normpdf(x,mu_est(i),sigma_est(i));
end
x = hx ; 
fhat_snr = zeros(size(x));
for i = 1:C
fhat_snr = fhat_snr + w_est(i) ./ (sqrt(8*pi*snr.*x)*sigma_est(i)) ...
    .* exp(-(sqrt(x./snr)-mu_est(i)).^2 ./(2*sigma_est(i)^2)) ;
end


%Check Amplitude Distribution
fig = figure; 
plot(h1,fhat,'k');
hold on ; 
plot(h1,rpdf,'r*');
legend('MoG amplitude PDF','Empirical amplitude PDF');
title(['EM Based approximation with' num2str(C) ' components']);

%Visualize SNR Distribution
fig = figure; 
plot(hx,fhat_snr,'k');
hold on ; 
plot(hx,rpdfx,'b*');
legend('MoG SNR PDF','Empirical SNR PDF');
title(['EM Based approximation with' num2str(C) ' components']);

%writing MoG parameters to a file.
dlmwrite('NEW_MoG_KappaMu_w_Mu_Sig.txt',...
    [w_est', mu_est,sigma_est'],'delimiter', '\t');

MSE = mean((fhat_snr-rpdfx).^2) ; %Mean square error for SNR PDF.

%% If the delay bothered you, consider using Variational Bayes method
%% for determining the number of components as well as getting parameters
%% simulataneously. refer to Chapter~3.