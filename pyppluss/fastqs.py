#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:27:45 2017

@author: edanrein
Vectorized FastQS as described in PeterStrobach (2010), Journal of Computational and Applied Mathematics 234
Based on MATLAB port by Aviv Ofir

v10-Oct-2018
-Cleaned up

v03-Sep-2018
-Increased MaxIter back to 50, to handle special cases.

v02-Aug-2018
-Cleaned up
-Reduced MaxIter back to 16
"""
import numpy as np

def QuarticSolverVec(a,b,c,d,e):
    """
     function [x1, x2, x3, x4]=QuarticSolverVec(a,b,c,d,e)
     v.0.2 - Python Port
     - Added condition in size sorting to avoid floating point errors.
     - Removed early loop abortion when stuck in loop (Inefficient)
     - Improved numerical stability of analytical solution
     - Added code for the case of S==0
     ============================================
     v.0.1 - Nearly identical to QuarticSolver v. 0.4, the first successful vectorized implimentation 
             Changed logic of ChosenSet to accomudate simultaneous convergence of sets 1 & 2
           - Note the periodicity in nearly-convergent solutions can other
             than four (related to text on step 4 after table 3). examples:
             period of 5: [a,b,c,d,e]=[0.111964240308252 -0.88497524334712 -0.197876116344933 -1.07336408259262 -0.373248675102065];
             period of 6: [a,b,c,d,e]=[-1.380904438798326 0.904866918945240 -0.280749330818231 0.990034312758900 1.413106456228119];
             period of 22: [a,b,c,d,e]=[0.903755513939902 0.490545114637739 -1.389679906455410 -0.875910689438623 -0.290630547104907];
             Therefore condition was changed from epsilon1(iiter)==0 to epsilon1(iiter)<8*eps (and similarl for epsilon2)
           - Special case criterion of the analytical formula was changed to
             ind=abs(4*Delta0**3./Delta1**2)<2*eps;  (instead of exact zero)
           - vectorized
     ============================================
     - Solves for the x1-x4 roots of the quartic equation y(x)=ax^4+bx^3+cx^2+dx+e.
       Multiple eqations can be soved simultaneously by entering same-sized column vectors on all inputs.
     - Note the code immediatly tanslates the input parameters ["a","b","c","d","e"] to the reference paper parameters [1,a,b,c,d] for consistency,
       and the code probably performes best when "a"=1.
    
    Parameters
    ----------
    a,b,c,d,e : ``1-D arrays``
        Quartic polynomial coefficients
    
    Returns
    ------
    - x1-x4 : ``2-D array``
        Concatenated array of the polynomial roots. The function always returns four (possibly complex) values. Multiple roots, if exist, are given multiple times. An error will result in four NaN values.
        No convergence may result in four inf values (still?)
    
    Reference: 
    Peter Strobach (2010), Journal of Computational and Applied Mathematics 234
        http://www.sciencedirect.com/science/article/pii/S0377042710002128
    """
#    MaxIter=16;
    MaxIter=50;
    eps = np.finfo(float).eps
    #INPUT CONTROL
    #Note: not all input control is implemented.
    # all-column vectors only
#    if size(a,1)~=size(b,1) or size(a,1)~=size(c,1) or size(a,1)~=size(d,1) or size(a,1)~=size(e,1) or ...
#       size(a,2)~=1 or size(b,2)~=1 or size(c,2)~=1 or size(d,2)~=1 or size(e,2)~=1:
#        fprintf('ERROR: illegal input parameter sizes.\n');
#        x1=inf; x2=inf; x3=inf; x4=inf;    
#        return
    
    # translate input variables to the paper's
    if np.any(a==0):
       print('ERROR: a==0. Not a quartic equation.\n')
       x1=np.NaN; x2=np.NaN; x3=np.NaN; x4=np.NaN;    
       return x1,x2,x3,x4
    else:
        input_a=a;
        input_b=b;
        input_c=c;
        input_d=d;
        input_e=e;
        a=input_b/input_a;
        b=input_c/input_a;
        c=input_d/input_a;
        d=input_e/input_a;
    
    # PRE-ALLOCATE MEMORY
    # ChosenSet is used to track which input set already has a solution (=non-zero value)
    ChosenSet=np.zeros_like(a);
    x1 = np.empty_like(a,complex)
    x1[:] = np.nan
    x2=x1.copy(); x3=x1.copy(); x4=x1.copy(); x11=x1.copy(); x12=x1.copy(); x21=x1.copy(); x22=x1.copy(); alpha01=x1.copy(); alpha02=x1.copy(); beta01=x1.copy(); beta02=x1.copy(); gamma01=x1.copy(); gamma02=x1.copy(); delta01=x1.copy(); delta02=x1.copy(); e11=x1.copy(); e12=x1.copy(); e13=x1.copy(); e14=x1.copy(); e21=x1.copy(); e22=x1.copy(); e23=x1.copy(); e24=x1.copy(); alpha1=x1.copy(); alpha2=x1.copy(); beta1=x1.copy(); beta2=x1.copy(); gamma1=x1.copy(); gamma2=x1.copy(); delta1=x1.copy(); delta2=x1.copy(); alpha=x1.copy(); beta=x1.copy(); gamma=x1.copy(); delta=x1.copy();
    # check multiple roots -cases 2 & 3. indexed by ChosenSet=-2
    test_alpha=0.5*a;
    test_beta=0.5*(b-test_alpha**2);
    test_epsilon=np.stack((c-2*test_alpha*test_beta, d-test_beta**2)).T;
    ind=np.all(test_epsilon==0,1);
    if np.any(ind):
        x1[ind], x2[ind]=SolveQuadratic(np.ones_like(test_alpha[ind]),test_alpha[ind],test_beta[ind]);
        x3[ind]=x1[ind]; x4[ind]=x2[ind];
        ChosenSet[ind]=-2;
    
    # check multiple roots -case 4. indexed by ChosenSet=-4
    i=ChosenSet==0;
    x11[i], x12[i]=SolveQuadratic(np.ones(np.sum(i)),a[i]/2,b[i]/6);
    x21[i]=-a[i]-3*x11[i];    
    test_epsilon[i,:2]=np.stack((c[i]+x11[i]**2*(x11[i]+3*x21[i]), d[i]-x11[i]**3*x21[i])).T;
    ind[i]=np.all(test_epsilon[i]==0,1);
    if np.any(ind[i]):
        x1[ind[i]]=x11[ind[i]]; x2[ind[i]]=x11[ind[i]]; x3[ind[i]]=x11[ind[i]]; x4[ind[i]]=x12[ind[i]];
        ChosenSet[ind[i]]=-4;
    x22[i]=-a[i]-3*x12[i];
    test_epsilon[i,:2]=np.stack((c[i]+x12[i]**2*(x12[i]+3*x22[i]), d[i]-x12[i]**3*x22[i])).T;
    ind[i]=np.all(test_epsilon[i]==0,1);
    if np.any(ind[i]):
        x1[ind[i]]=x21[ind[i]]; x2[ind[i]]=x21[ind[i]]; x3[ind[i]]=x21[ind[i]]; x4[ind[i]]=x22[ind[i]];
        ChosenSet[ind[i]]=-4;
    # General solution
    # initilize
    epsilon1=np.empty((np.size(a),MaxIter))
    epsilon1[:]=np.inf
    epsilon2=epsilon1.copy();
    
    i=ChosenSet==0;
    fi=np.nonzero(i)[0];
    x=np.empty((fi.size,4),complex)
    ii = np.arange(fi.size)
    #Calculate analytical root values
    x[:,0], x[:,1], x[:,2], x[:,3]=AnalyticalSolution(np.ones(np.sum(i)),a[i],b[i],c[i],d[i],eps);
    #Sort the roots in order of their size
    ind=np.argsort(abs(x))[:,::-1]; #'descend'
    x1[i]=x.flatten()[4*ii+ind[:,0]];
    x2[i]=x.flatten()[4*ii+ind[:,1]];
    x3[i]=x.flatten()[4*ii+ind[:,2]];
    x4[i]=x.flatten()[4*ii+ind[:,3]];
    #Avoiding floating point errors.
    #The value chosen is somewhat arbitrary. See Appendix C for details.
    ind = abs(x1)-abs(x4)<8*10**-12;
    x2[ind] = np.conj(x1[ind])
    x3[ind] = -x1[ind]
    x4[ind] = -x2[ind]
    #Initializing parameter values
    alpha01[i]=-np.real(x1[i]+x2[i]);
    beta01[i]=np.real(x1[i]*x2[i]);
    alpha02[i]=-np.real(x2[i]+x3[i]);
    beta02[i]=np.real(x2[i]*x3[i]);
    gamma01[i], delta01[i]=FastGammaDelta(alpha01[i],beta01[i],a[i],b[i],c[i],d[i]);
    gamma02[i], delta02[i]=FastGammaDelta(alpha02[i],beta02[i],a[i],b[i],c[i],d[i]);
    
    alpha1[i]=alpha01[i]; beta1[i]=beta01[i]; gamma1[i]=gamma01[i]; delta1[i]=delta01[i];
    alpha2[i]=alpha02[i]; beta2[i]=beta02[i]; gamma2[i]=gamma02[i]; delta2[i]=delta02[i];
    
    #Backward Optimizer Outer Loop
    e11[i]=a[i]-alpha1[i]-gamma1[i];
    e12[i]=b[i]-beta1[i]-alpha1[i]*gamma1[i]-delta1[i];
    e13[i]=c[i]-beta1[i]*gamma1[i]-alpha1[i]*delta1[i];
    e14[i]=d[i]-beta1[i]*delta1[i];
    
    e21[i]=a[i]-alpha2[i]-gamma2[i];
    e22[i]=b[i]-beta2[i]-alpha2[i]*gamma2[i]-delta2[i];
    e23[i]=c[i]-beta2[i]*gamma2[i]-alpha2[i]*delta2[i];
    e24[i]=d[i]-beta2[i]*delta2[i];
    iiter=0;
    while iiter<MaxIter and np.any(ChosenSet[i]==0):
        i=np.nonzero(ChosenSet==0)[0];
        
        alpha1[i], beta1[i], gamma1[i], delta1[i], e11[i], e12[i], e13[i], e14[i], epsilon1[i,iiter]=BackwardOptimizer_InnerLoop(a[i],b[i],c[i],d[i],alpha1[i],beta1[i],gamma1[i],delta1[i],e11[i],e12[i],e13[i],e14[i]);
        alpha2[i], beta2[i], gamma2[i], delta2[i], e21[i], e22[i], e23[i], e24[i], epsilon2[i,iiter]=BackwardOptimizer_InnerLoop(a[i],b[i],c[i],d[i],alpha2[i],beta2[i],gamma2[i],delta2[i],e21[i],e22[i],e23[i],e24[i]);
    
        j = np.ones_like(a[i])
        j[(epsilon2[i,iiter]<epsilon1[i,iiter]).flatten()] = 2
        BestEps = np.nanmin(np.stack([epsilon1[i,iiter].flatten(), epsilon2[i,iiter].flatten()]),0);
        ind=BestEps<8*eps;
        ChosenSet[i[ind]]=j[ind];
        ind=np.logical_not(ind);
#        if iiter>0 and np.any(ind):
#            ii=i[ind];
#            LimitCycleReached = np.empty((ii.size,2),bool)
#            LimitCycleReached[:,0] = np.any(epsilon1[ii,:iiter]==epsilon1[ii,iiter],0)
#            LimitCycleReached[:,1] = np.any(epsilon2[ii,:iiter]==epsilon2[ii,iiter],0)
##            LimitCycleReached=[any(bsxfun(@eq,epsilon1(i(ind),max(1,iiter-4):max(1,iiter-1)),epsilon1(i(ind),iiter)),2) any(bsxfun(@eq,epsilon2(i(ind),max(1,iiter-4):max(1,iiter-1)),epsilon2(i(ind),iiter)),2)];
#            ChosenSet[ii[np.logical_and(LimitCycleReached[:,0] , np.logical_not(LimitCycleReached[:,1]))]]=1;
#            ChosenSet[ii[np.logical_and(LimitCycleReached[:,1] , np.logical_not(LimitCycleReached[:,0]))]]=2;
##            ChosenSet(ii(~LimitCycleReached(:,1) & LimitCycleReached(:,2)))=2;
##            ind=find(ind);
#            cond = np.logical_and(LimitCycleReached[:,1],LimitCycleReached[:,0])
#            ChosenSet[ii[cond]]=j[ind][cond]
##            ChosenSet(ii(LimitCycleReached(:,1) & LimitCycleReached(:,2)))=j(ind(LimitCycleReached(:,1) & LimitCycleReached(:,2)));
        iiter=iiter+1;
        
    #Checking which of the chains is relevant
    i=np.nonzero(ChosenSet==0)[0];
    ind=epsilon1[i,-1]<epsilon2[i,-1];
#    ind=np.logical_and(epsilon1[i,-1]<epsilon2[i,-1],np.logical_not(np.isnan(epsilon2[i,-1])));
    ChosenSet[i[ind]]=1;
    ChosenSet[i[np.logical_not(ind)]]=2;
    
    # Output
    i=ChosenSet==1;
    alpha[i]=alpha1[i];
    beta[i]=beta1[i];
    gamma[i]=gamma1[i];
    delta[i]=delta1[i];
    
    i=ChosenSet==2;
    alpha[i]=alpha2[i];
    beta[i]=beta2[i];
    gamma[i]=gamma2[i];
    delta[i]=delta2[i];
    
    i=ChosenSet>0;
    x1[i], x2[i]=SolveQuadratic(np.ones(np.sum(i)),alpha[i],beta[i]);
    x3[i], x4[i]=SolveQuadratic(np.ones(np.sum(i)),gamma[i],delta[i]);

    return np.array([x1,x2,x3,x4])


def AnalyticalSolution(a,b,c,d,e,eps):
    """
    Source: https://en.wikipedia.org/wiki/Quartic_function#General_formula_for_roots
    Calculates the value of the analytical solution.
    """
    p=(8*a*c-3*b**2)/(8*a**2)
    q=(b**3-4*a*b*c+8*a**2*d)/(8*a**3)
    
    Delta0=c**2-3*b*d+12*a*e
    Delta1=2*c**3 -9*b*c*d +27*b**2*e +27*a*d**2 -72*a*c*e
    Q = np.empty_like(Delta1, complex)
    cond = Delta1>=0
    Q[cond]=(0.5*(Delta1[cond]+np.sqrt(Delta1[cond]**2-4*Delta0[cond]**3)))**(1/3)
    #Improved numerical stability of calculation of Q by noting (when d1<0):
    #(a+b)(a-b)=a**2-b**2; here (d1+sqrt(...))(d1-sqrt(...))=4*d0**3.
    cond = Delta1<0
    Q[cond]=Delta0[cond]*(2/(Delta1[cond]-np.sqrt(Delta1[cond]**2-4*Delta0[cond]**3)))**(1/3)
    #In this case we need to set the sign of sqrt equal to that of delta1
    ind = abs(Delta0)<eps
    Q[ind] = Delta1[ind]**(1/3)
    #The case of triple root+singular is not properly handled by the code.
#    cond = np.logical_and(ind,Delta1<eps)
#    aa = a[cond]; bb=b[cond]; cc=c[cond];dd = d[cond];
#    xparam = dd-bb*cc/(12*aa)-0.5*bb/aa*(5/6*cc-bb**2/(4*aa))
#    x1[cond] = xparam/(cc/(6*aa)*(5/6*cc-bb**2/(4*aa))-e[cond])
#    x2[cond] = x1[cond]
#    x3[cond] = x1[cond]
#    x4[cond] = -bb/aa-3*x1[cond]
#    cond = np.logical_not(cond)
#    ind=abs(4*Delta0**3)<2*eps*abs(Delta1**2)    
#    if np.any(ind):
#        ind2=np.zeros_like(a,bool);
#        Delta = np.empty_like(Q, complex)
#        Delta[ind]=256*a[ind]**3*e[ind]**3-192*a[ind]**2*b[ind]*d[ind]*e[ind]**2-128*a[ind]**2*c[ind]**2*e[ind]**2+144*a[ind]**2*c[ind]*d[ind]**2*e[ind]-27*a[ind]**2*d[ind]**4
#        +144*a[ind]*b[ind]**2*c[ind]*e[ind]**2-6*a[ind]*b[ind]**2*d[ind]**2*e[ind]-80*a[ind]*b[ind]*c[ind]**2*d[ind]*e[ind]+18*a[ind]*b[ind]*c[ind]*d[ind]**3+16*a[ind]*c[ind]**4*e[ind] 
#        -4*a[ind]*c[ind]**3*d[ind]**2-27*b[ind]**4*e[ind]**2+18*b[ind]**3*c[ind]*d[ind]*e[ind]-4*b[ind]**3*d[ind]**3-4*b[ind]**2*c[ind]**3*e[ind]+b[ind]**2*c[ind]**2*d[ind]**2;
#        ind2[ind]=np.logical_not(Delta[ind]==0);
#        if np.any(Delta[np.logical_and(ind,ind2)]):
#            Q[np.logical_and(ind, ind2)]=(Delta1[np.logical_and(ind, ind2)])**(1/3);
    
    S=0.5*np.sqrt(-2/3*p+1/(3*a)*(Q+Delta0/Q))
    cond = S==0
    Q[cond] *= np.exp(1j*np.pi/3)
    S=0.5*np.sqrt(-2/3*p+1/(3*a)*(Q+Delta0/Q))
    cond = S==0
    Q[cond] *= np.exp(1j*np.pi/3)
    S=0.5*np.sqrt(-2/3*p+1/(3*a)*(Q+Delta0/Q))
    cond = S==0
    if np.any(cond):
        raise NotImplementedError("A root is (probably) singular or repeats 3 times. Code does not treat it (yet)")
    x1=-b/(4*a)-S+0.5*np.sqrt(-4*S**2-2*p+q/S);
    x2=-b/(4*a)-S-0.5*np.sqrt(-4*S**2-2*p+q/S);
    x3=-b/(4*a)+S+0.5*np.sqrt(-4*S**2-2*p-q/S);
    x4=-b/(4*a)+S-0.5*np.sqrt(-4*S**2-2*p-q/S);
    return x1,x2,x3,x4
   
def FastGammaDelta(alpha0,beta0,a,b,c,d):
    # Table 3
    phi1=1+alpha0**2+beta0**2;
    phi2=alpha0*(1+beta0);
    c1=a-alpha0+alpha0*(b-beta0)+beta0*c;
    c2=b-beta0+alpha0*c+beta0*d;
    L1=np.sqrt(phi1);
    L3=phi2/L1;
    L2=np.sqrt(phi1-phi2**2/phi1);
    y1=c1/L1;
    y2=(c2-y1*L3)/L2;
    delta0=y2/L2;
    gamma0=(y1-delta0*L3)/L1;
    return gamma0, delta0

def BackwardOptimizer_InnerLoop(a,b,c,d,alpha,beta,gamma,delta,e1,e2,e3,e4):
    U23=alpha-gamma;
    U33=beta-delta-gamma*U23;
    L43=-delta*U23/U33;
    U44=beta-delta-L43*U23;
   
    x1=e1;
    x2=e2-gamma*x1;
    x3=e3-delta*x1-gamma*x2;
    x4=e4-delta*x2-L43*x3;
    y4=x4/U44;
    y3=(x3-U23*y4)/U33;
    y2=x2-U23*y3-y4;
    y1=x1-y3;
       
    alpha=alpha+y1;
    beta=beta+y2;
    gamma=gamma+y3;
    delta=delta+y4;
   
    e1=a-alpha-gamma;
    e2=b-beta-alpha*gamma-delta;
    e3=c-beta*gamma-alpha*delta;
    e4=d-beta*delta;   
    epsilon=abs(e1)+abs(e2)+abs(e3)+abs(e4);
    return alpha, beta, gamma, delta, e1, e2, e3, e4, epsilon

def SolveQuadratic(a,b,c):
    # Chapter 5.6 from Numerical Recepies
    i=np.all(np.imag(np.stack([a, b, c]))==0,0);
    ii = b==0
    q = np.empty_like(a,complex)
    q[i]=-0.5*(b[i]+np.sign(b[i])*np.sqrt(b[i]**2-4*a[i]*c[i]));
    #Correct for sign(b)=0
    q[ii]=-0.5*np.sqrt(-4*a[ii]*c[ii])
    i=np.logical_not(i);
    if np.any(i):
        s = np.empty_like(a)
        s[i]=abs(np.real(np.conj(b[i])*np.sqrt(b[i]**2-4*a[i]*c[i])));
#        s[i[s[i]<0]]=-s[i[s[i]<0]];
        q[i]=-0.5*(b[i]+s[i]);
    
    i = q==0
    if np.any(i):
        x1=np.zeros_like(a,complex);x2=np.zeros_like(a,complex);
        i = np.logical_not(i)
        x1[i]=q[i]/a[i];
        x2[i]=c[i]/q[i];
    else:
        x1=q/a;
        x2=c/q;
    return x1,x2