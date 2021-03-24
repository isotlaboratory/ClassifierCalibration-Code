__all__ = ['LogisticRegressionCalibrator', 'LogisticRegressionCalibratorPlatt', 'IsotonicRegressionCalibrator',"BinningCalibrator", "GeneralizedLambdaDistribution","mannWhitney", "setMeanFromMannWhitney"]

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, norm, beta
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, mean_squared_error, auc, roc_curve


class LogisticRegressionCalibratorPlatt:
    A = 0
    B = 0

    def __init__(self):
        pass

    def train(self, h0, h1, withLinFix=True):
        #fits the LR model, self.Calibrator, to scores h0 and h1. Assumes labels of h0 is 0 (normal) and labels of h1 is 1 (anamolous)
        #expects single dimensional arrays of anyform (list, numpy, mx1, 1xm, etc.)

        n0 = np.size(h0)
        n1 = np.size(h1)
        H = np.append(np.reshape(h0, n0), np.reshape(h1, n1))
        H = H.reshape(n0 + n1, 1)
        tp = (n1 + 1) / (n1 + 2)
        tn = 1/(n0+2)
        t = np.array(([tn] * n0) + ([tp] * n1))
        self.fit(H, t, n0, n1)
        
    def test(self, H):
        #Reutrns the predicted posterioir probabilities of the list of scores H using the fitted LR model, self.Calibrator
        #expects single dimensional array of anyform (list, numpy, mx1, 1xm, etc.)
        #   Arguments:
        #       H - classifier scores
        
        H = np.array(H)
        n = np.size(H)
        H = H.reshape(n, 1)
        return 1/(1+np.exp(H*self.A + self.B))
        
    def fit(self, H, t, n0, n1):
        """
        Input parameters:
            H = array of scores
            t = array of adjusted targets
            n1 = number of positive examples
            n0 = number of negative examples
        Outputs:
            A, B = parameters of sigmoid
        """

        #Parameter setting
        maxiter=10000 #Maximum number of iterations
        minstep=1e-10 #Minimum step taken in line search
        sigma=1e-12 #Set to any value > 0

        len=n0+n1 #total data set size
        
        A=0.0
        B=np.log((n0+1.0)/(n1+1.0))
        fval=0.0
        for i in range(len):
            fApB=H[i]*A+B #Ay+B
            if (fApB >= 0):
                fval += t[i]*fApB+np.log(1+np.exp(-fApB)) # y(Ay+B) + log(1+e^(-Ay+B)) 
            else:
                fval += (t[i]-1)*fApB+np.log(1+np.exp(fApB)) # (y-1)(Ay+B) + log(1+e^(Ay+B)
        
        for it in range(maxiter): 
            #Update Gradient and Hessian (use H’ = H + sigma I)
            h11=h22=sigma
            h21=g1=g2=0.0
            for i in range(len):
                fApB=H[i]*A+B #Ay+B
                if (fApB >= 0):
                    p=np.exp(-fApB)/(1.0+np.exp(-fApB))     # p = e^(-Ay+B)/1+e^(-Ay+B)
                    q=1.0/(1.0+np.exp(-fApB))               # q = 1/1+e^(-Ay+B)
                else:
                    p=1.0/(1.0+np.exp(fApB))                # p = 1/1+e^(Ay+B)
                    q=np.exp(fApB)/(1.0+np.exp(fApB))       # q = e^(Ay+B)/1+e^(Ay+B)
                d2=p*q
                h11 += H[i]*H[i]*d2
                h22 += d2
                h21 += H[i]*d2
                d1 = t[i]-p
                g1 += H[i]*d1
                g2 += d1
            if ( abs(g1)<1e-5 and abs(g2)<1e-5 ): #Stopping criteria
                break
        
            #Compute modified Newton directions
            det=h11*h22-h21*h21
            dA=-(h22*g1-h21*g2)/det
            dB=-(-h21*g1+h11*g2)/det
            gd=g1*dA+g2*dB
            stepsize=1
            while (stepsize >= minstep): #Line search
                newA=A+stepsize*dA
                newB=B+stepsize*dB
                newf=0.0
                for i in range(len):
                    fApB=H[i]*newA+newB #A'y+B'
                    if (fApB >= 0):
                        newf += t[i]*fApB+np.log(1+np.exp(-fApB)) # y(A'y+B') + log(1+e^(-(A'y+B')))
                    else:
                        newf += (t[i]-1)*fApB+np.log(1+np.exp(fApB)) # (y-1)(A'y+B') + log(1+e^(A'y+B'))
            
                if (newf < (fval+0.0001*stepsize*gd) ):
                    A=newA
                    B=newB
                    fval=newf
                    break #Sufficient decrease satisfied
                else:
                    stepsize /= 2.0
            if (stepsize < minstep):
                print("Line search fails")
                break
        if (it >= maxiter):
            print("Reaching maximum iterations")
        
        self.A = A
        self.B = B

        return 0

    def toString(self):
        return "Platt"


class LogisticRegressionCalibrator:
    Ext=False
    scaler=None

    def __init__(self,  exstensible=False):
        #set extensible to True to add extra feature(s) x^2 (mono clf case) and x1^2, x2^2, x1*x2 (multi clf case) 
        self.Calibrator = LogisticRegression(solver='lbfgs', n_jobs=1)
        self.Ext = exstensible

    def train(self, h0, h1,):
        #fits the LR model, self.Calibrator, to scores h0 and h1. Assumes labels of h0 is 0 (normal) and labels of h1 is 1 (anamolous)
        #expects single dimensional arrays of anyform (list, numpy, mx1, 1xm, etc.)

        n0 = h0.shape[0]
        n1 = h0.shape[0]
        H = np.append(h0, h1, axis=0)
        if not len(H.shape) > 1: #add column dimension
            H = H.reshape(H.shape[0], 1)

        y = np.append(np.zeros(n0), np.ones(n1)) #labels
        
        if self.Ext == True:
            H = np.concatenate( (H, H**2), 1) #add squared column(s)
            if H.shape[1] > 2: #if multi classifier calibration
                H = np.concatenate( (H, np.reshape(H[:,0]*H[:,1], (H.shape[0],1) ) ), 1) #add x1*x2
            mms = MinMaxScaler()
            mms.fit(H.astype(float))
            H = mms.transform(H) #scale data to 0,1 range
            self.scaler = mms #save scaler
            
        self.Calibrator.fit(H,y)


    def test(self, H):
        #Reutrns the predicted posterioir probabilities of the list of scores H using the fitted LR model, self.Calibrator
        #expects single dimensional array of anyform (list, numpy, mx1, 1xm, etc.)
        #   Arguments:
        #       H - classifier scores
        
        if not len(H.shape) > 1: #add column dimension
            H = H.reshape(H.shape[0], 1)

        if self.Ext:
            H = np.concatenate( (H, H**2), 1) #add squared column
            if H.shape[1] > 2: #if multi classifier calibration
                H = np.concatenate( (H, np.reshape(H[:,0]*H[:,1], (H.shape[0],1) ) ), 1) #add squared column
            H = self.scaler.transform(H) #scale data
        return( self.Calibrator.predict_proba(H)[:,1] )

    def toString(self):
        if self.Ext == True:
            return "LogisticExtensible"
        else:
            return "Logistic"


class IsotonicRegressionCalibrator:
    
    def __init__(self):
      self.Calibrator = IsotonicRegression(out_of_bounds='clip')

    def train(self, h0, h1):
        #fits the IR model, self.Calibrator, to scores h0 and h1. Assumes labels of h0 is 0 (normal) and labels of h1 is 1 (anamolous)
        #expects single dimensional arrays of anyform (list, numpy, mx1, 1xm, etc.)

        n0 = np.size(h0)
        n1 = np.size(h1)
        H = np.append(np.reshape(h0, n0), np.reshape(h1, n1))

        y = np.append(np.zeros(n0), np.ones(n1)) #labels

        self.Calibrator.fit(H,y)


    def test(self, H):
        #Reutrns the predicted posterioir probabilities of the list of scores H using the fitted IR model in self.Calibrator.
        #expects single dimensional array of anyform (list, numpy, mx1, 1xm, etc.)
        #   Keyword arguments:
        #       H - classifier scores

        return self.Calibrator.predict(H)

    def toString(self):
        return "Isotonic"


class BinningCalibrator:
    descrete = False
    EPS = np.finfo(float).eps #min. precision: y/(x+eps) will guard against DBZ error when x=0.


    def __init__(self, descrete=False, n_bins=10):
        #set descrete = True for descrete valued scores, and false for continious valued scores
        #n_bins determines the number of equal width bins only for contious valued scores, otherwise this variable is ignored

        self.descrete = descrete
        self.n_bins=n_bins

    def train(self, h0, h1):
        #Gets nonparametric MLE posterioir for a set of equal width bings from scores h0 and h1. Assumes labels of h0 is 0 (normal) and labels of h1 is 1 (anamolous)
        #expects single dimensional arrays of anyform (list, numpy, mx1, 1xm, etc.)

        n0 = np.size(h0)
        n1 = np.size(h1)
        H = np.append(np.reshape(h0, n0), np.reshape(h1, n1))
        H = H.reshape(n0 + n1, 1)
        Hmin =  np.amin(H)
        Hmax = np.amax(H)

        if self.descrete:
            #   descrete scores assume to have take on all integer values from Hmin to Hmax
            bw = 1 #bin width
            self.n_bins = np.unique(H).shape[0] #number of bins equal to number of classes
            self.bin_edges = np.arange(0, self.n_bins+1, bw) - bw/2 #bins 
        
        else:
            self.bin_edges, _ = np.linspace(Hmin, Hmax, self.n_bins+1, retstep=True) 
        lhr = np.zeros(self.n_bins)
        counts0, bins0 = np.histogram(h0, bins=self.bin_edges, density=True) #histogram of nefative class
        counts1, bins1 = np.histogram(h1, bins=self.bin_edges, density=True) #histogram of positive class
        
        p = n1/(n1 + n0) #prior
        for i in range(lhr.shape[0]):
            if counts0[i] == 0 and counts1[i] != 0:# 1/0 = ∞
                lhr[i] = np.inf
            elif counts0[i] != 0 and counts1[i] == 0:# 0/1 = 0
                lhr[i] = 0
            elif counts0[i] == 0 and counts1[i] == 0:# 0/0 = ( (leftPosCount + rightPosCount)/(2+n) )  /  ( (leftNegCount + rightNedCount)/(2+n) ) # where n is number of 0/0 bins. Note: 2+n cancels out thus is irrelevant
                lhr[i] = -1 #replaced with true value later
            else:
                lhr[i] = counts1[i]/counts0[i]         #nonparametric MLE lhr ratio of number of observations in each bin.

        j = 0
        for i in range(lhr.shape[0]):
            if lhr[i] == -1: #count number of bins which evaluated to 0/0 in current run
                j+=1
            elif j > 0: #if end of run of 0/0 bins 
                if (counts0[i-j-1]+counts0[i]) == 0: #if rightBin and leftBin both = ∞
                    for k in range(1,j+1): #all 0/0 bins' lhr estimate set to ∞
                       lhr[i-k] = np.inf
                else:
                    for k in range(1,j+1): #use counts in left and right bin to set all 0/0 bins' lhr estimate
                        lhr[i-k] = (counts1[i-j-1]+counts1[i])/(counts0[i-j-1]+counts0[i])
                j = 0 #no longer in run of 0/0 bins

        self.pmle = 1/(1+ (lhr + self.EPS)**-1 * (1-p)/p)  #nonparametric MLE posterioir for each bin.

    def test(self, H):
        #Reutrns the predicted posterioir probabilities of the list of scores H using the fitted IR model in self.Calibrator.
        #expects single dimensional array of anyform (list, numpy, mx1, 1xm, etc.)
        #   Keyword arguments:
        #       H - classifier scores

        H = np.array(H)
        n = np.size(H)
        H = H.reshape(n, 1)

        tempPmle = np.concatenate(([self.pmle[0]],self.pmle,[self.pmle[-1]]))
        y = tempPmle[np.searchsorted(self.bin_edges, H)]
        return y
    
    def toString(self):
        return "Binning("+str(self.n_bins)+")"


class GeneralizedLambdaDistribution:

    #References: [1] Ramberg, J. S., Tadikamalla, P. R., Dudewicz, E. J., and Mykytka, E. F. (1979), “A Probability Distribution and Its Uses in FittingData,” Technometrics, 21, 201.

    lam1, lam2, lam3, lam4 = 0.0, 0.0, 0.0, 0.0 

    def __init__(self, lam1, lam2, lam3, lam4, resolution=10000000):
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.resolution = resolution #the precision the pdf is approximated with since no closed for the pdf exists
        self.calcPDF()  #precalculate the PDF for resultion many points

    def calcPDF(self): #calculates points on the pdf plot to be used in pdf()
        x = np.linspace(0, 1, self.resolution)
        self.ppfX = self.ppf(x)
        self.pdfX = self.fx(x)
        
    def sample(self, n=1): #sample n times from the distribution defined by lambda parameters
        U =  np.random.uniform(0, 1, n)
        x = self.ppf(U)
        return x
    
    def rvs(self, n=1): # alias for sample() to match the name of Scipy distribution's function for sampling from a distribution
        return self.sample(n)

    def ppf(self, q): #percent-point/qauntile function/inverse distribution function
        return self.lam1 + ( (q**self.lam3 - (1 - q)**self.lam4)/self.lam2 ) #Ramberg and Schmeiser parameterization
        #return self.lam1 + ( ( ((q**self.lam3)/self.lam3) -   (((1-q)**self.lam4 - 1)/self.lam4) ) / self.lam2  ) # Freimer, Mudholkar, Kollia and Lin parameterization

    def pdf(self, x): #determines the value of the pdf at a each in vector (x)
        i = np.searchsorted(self.ppfX, x)
        index = np.append( self.pdfX, [self.pdfX[self.resolution-1], self.pdfX[self.resolution-1]])
        pd = (index[i]+index[i+1])/2
        return pd

    def fx(self, x): # density function
        Px = self.lam2 / ( self.lam3*((x)**(self.lam3-1)) + self.lam4*(((1 - x))**(self.lam4-1)) )
        return Px

    def plot(self, xlim=(-3,3), ylim=(0,1), title="", wait=False, lineNames=None): #plot the pdf of the lambda function
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.plot(self.ppfX, self.pdfX)
        if wait == False:
            plt.title(title)
            if lineNames != None:
                plt.legend(lineNames)
            plt.show()
            plt.close()
    
    def toString(self, bare=False): #returns string description of the GLD's parameters
        if bare:
            return str(self.lam1)+"_"+str(self.lam2)+"_"+str(self.lam3)+"_"+str(self.lam4)
        else:
            return "λ1:"+str(self.lam1)+", λ2:"+str(self.lam2)+", λ3:"+str(self.lam3)+", λ4:"+str(self.lam4)

    def saveDist(self, dir, mw=None):
        if mw is None:
            fp = open(dir+"/"+"Dist_"+str(self.lam1)+"_"+str(self.lam2)+"_"+str(self.lam3)+"_"+str(self.lam4)+".csv","w+") #create evauation file for distribution pair
        else:
            fp = open(dir+"/"+"Dist_("+str(mw)+")_"+str(self.lam1)+"_"+str(self.lam2)+"_"+str(self.lam3)+"_"+str(self.lam4)+".csv","w+") #create evauation file for distribution pair
        for i in range(self.ppfX.shape[0]):
            fp.write(str(self.ppfX[i])+","+str(self.pdfX[i])+"\n")
        fp.close()

    def setMeanFromMannWhitneyGLD(self, target, x_0, normalize=False, tol=0.001, step=0.1): #if normalize = True, assumes that x_0 already has unit variance and normalize x_1 to unit variance
        lb =  -1
        ub = 10
        ubStep = 1
        
        #get current AUC
        x_1 = self.sample(x_0.shape[0])
        std1 = None
        if normalize:
            std1 = np.std(x_1)
            x_1 = x_1/std1
        mw = mannWhitney(x_0, x_1)
        
        while mw < target: #find upper bound for lambda 1
            lb = self.lam1
            ub += ubStep
            ubStep *=1.5 #take a larger step in AUC at upper bound < target
            self.lam1 = ub

            #get new AUC
            x_1 = self.sample(x_0.shape[0])
            if normalize:
                std1 = np.std(x_1)
                x_1 = x_1/std1
            mw = mannWhitney(x_0, x_1)
        
        #begin binary search, get AUC at midpoint
        self.lam1 = (ub + lb)/2 
        x_1 = self.sample(x_0.shape[0])
        if normalize:
            std1 = np.std(x_1)
            x_1 = x_1/std1
        mw = mannWhitney(x_0, x_1)


        while not ((target - tol) <= mw and mw <= (target + tol)): #continue search if AUC not close enough to target
            if mw < target: #search upper half
                lb = self.lam1 
                self.lam1  = (lb + ub)/2 
            else: #search lower half
                ub = self.lam1 
                self.lam1  = (lb + ub)/2 

            #update AUC   
            x_1 = self.sample(x_0.shape[0])
            if normalize:
                std1 = np.std(x_1)
                x_1 = x_1/std1
            mw = mannWhitney(x_0, x_1)

        #update PDF
        self.calcPDF()
        return mw, std1, np.mean(x_1)


class exponentialDistribution:

    lam = 1.0
    flip = False

    def __init__(self, lam=1.0, flip=False):
        self.lam = lam
        self.flip = flip

    def rvs(self, n=1): #sample n times from the distribution defined by lambda parameter
        u = np.random.uniform(0, 1, n)
        x = self.ppf(u)
        return x
    
    def pdf(self, x):
        if self.flip:
            d = (self.lam * np.exp(self.lam*(x-1))) / (1-np.exp(-self.lam))
        else:
            d = (self.lam * np.exp(-self.lam*x)) / (1-np.exp(-self.lam))
        return d

    def ppf(self, p):
        if self.flip:
            x = ( np.log( p * (1 - np.exp(-self.lam)) + np.exp(-self.lam)) / self.lam ) + 1
        else:
            x = -np.log(1 - ((1-np.exp(-self.lam))*p) )/self.lam
        return x

    def toString(self):
        return "expo"


def setLambdasFromMannWhitneyExpon(target, negDist, posDist, tol=0.0001, step=0.1): #if normalize = True, assumes that x_0 already has unit variance and normalize x_1 to unit variance
    
    lam = 1
    lb =  0.0000000000001
    ub = 10
    ubStep = 2
    n = 1000000

    
    #get current AUC
    x_1 = posDist.rvs(n)
    x_0 = negDist.rvs(n)
    mw = mannWhitney(x_0, x_1)
    print(mw, "\t", lam)

    while mw < target: #find upper bound for lambda 1
        ub += ubStep
        ubStep *=1.25 #take a larger step in AUC at upper bound < target
        lam = ub

        #get new AUC
        posDist.lam = lam
        negDist.lam = lam
        x_1 = posDist.rvs(n) 
        x_0 = negDist.rvs(n)
        mw = mannWhitney(x_0, x_1)
        print(mw, "\t", lam)

    #begin binary search, get AUC at midpoint
    lam = (ub + lb)/2
    posDist.lam = lam
    negDist.lam = lam
    x_1 = posDist.rvs(n)
    x_0 = negDist.rvs(n)
    mw = mannWhitney(x_0, x_1)
    

    while not ((target - tol) <= mw and mw <= (target + tol)): #continue search if AUC not close enough to target
        if mw < target:  #search upper half
            lb = lam
            lam  = (lb + ub)/2 
        else: #search bottom half
            ub = lam
            lam  = (lb + ub)/2 

        #update AUC 
        lam = (ub + lb)/2
        posDist.lam = lam
        negDist.lam = lam
        x_1 = posDist.rvs(n)
        x_0 = negDist.rvs(n)
        mw = mannWhitney(x_0, x_1)
        print(mw, "\t", lam)

    return mw, lam


def mannWhitney(x_0, x_1):

    n = x_0.shape[0]

    fpr, tpr, thresholds = roc_curve(np.append(np.zeros(n), np.ones(n)), np.append(x_0,x_1), pos_label=1)
    return auc(fpr, tpr)



def setMeanFromMannWhitney(dist, x_0, target, p1, p2, normalize=False, tol=0.001, step=0.1): #if normalize = True, assumes that x_0 already has unit variance and normalize x_1 to unit variance
        lb = 0
        ub = 1
        ubStep = 1
        
        #get current AUC
        std = None
        x_1 = dist.rvs(p1, p2, size=x_0.shape[0])
        if normalize:
            std1 = np.std(x_1)
            x_1 = x_1/std1
        mw = mannWhitney(x_0, x_1)

        while mw < target: #find upper bound for p1
            lb = p1
            ub += ubStep
            ubStep *=1.5 #take a larger step in AUC at upper bound < target
            p1 = ub

            #get new AUC
            x_1 = dist.rvs(p1, p2, size=x_0.shape[0])
            if normalize:
                std1 = np.std(x_1)
                x_1 = x_1/std1
            mw = mannWhitney(x_0, x_1)

        #begin binary search, get AUC at midpoint
        p1 = (ub + lb)/2
        x_1 = dist.rvs(p1, p2, size=x_0.shape[0])
        if normalize:
            std1 = np.std(x_1)
            x_1 = x_1/std1
        mw = mannWhitney(x_0, x_1)

        while not ((target - tol) <= mw and mw <= (target + tol)): #continue search if AUC not close enough to target
            if mw < target: #search upper half
                lb = p1
                p1 = (lb + ub)/2 
            else: #search lower half
                ub = p1
                p1 = (lb + ub)/2

            #update AUC   
            x_1 = dist.rvs(p1, p2, size=x_0.shape[0])
            if normalize:
                std1 = np.std(x_1)
                x_1 = x_1/std1
            mw = mannWhitney(x_0, x_1)
        
        return p1, mw, std1, np.mean(x_1)