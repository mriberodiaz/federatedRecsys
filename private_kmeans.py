import numpy as np
from utils import give_percentiles_all, give_percentiles, avg_rank
from utilsMovie import give_percentiles_m, give_percentiles_all_m


###### PARTITION SCRIPT ###### 
def partition( X_data,n,p,epsilon , side):
# Partitioning using the regular grid.
    gridpoints=[]  #list with squares centers to be kept as candidates. e.g. [(0,0), (1/2,1/2), (-1/2, -1/2)....]
    depth=1
    #print('            What partition received as eps: {}'.format(epsilon))
    epss=epsilon/(2*np.log(n))
    gamma=2*np.log(n)/epss
    thegrid= [{'coordinate': np.zeros(p), 'points':np.arange(n)}]
    #print('      Gamma = {},  epss = {} '.format(gamma,epss))

    while (depth<=np.log(n)) and (len(thegrid) > 0) :
        #print('.       Depth: {}'.format(depth))
        #print('entered while')
        #print(gridpoints)
        L=len(thegrid);
        side=side/2;
        #print('         Length: {}.  --   Side length: {}'.format(L, side))
        newgrid=[] ;
        for j in range(L):
            gridpoints.append(thegrid[j]['coordinate']);
            npoints=len(thegrid[j]['points']);
            directions_numeric=np.sign(X_data[:,thegrid[j]['points']].T-thegrid[j]['coordinate']*np.ones((npoints,1)));
            keyset = np.unique(directions_numeric, axis = 0);
            valset = []
            for i in range(len(keyset)):
                valset.append(list(np.where([np.all(keyset[i]==directions_numeric[k]) for k in range(len(directions_numeric)) ])[0]))
            #valset=values(cubemap);
            #print('LEN VALSET:  ')
            #print([len(valset[i]) for i in range(len(valset))])
            for i in range(len(keyset)):
                activesize=len(valset[i]);
                if(activesize>gamma):
                    #print('                 large size')
                    prob=1-0.5*np.exp(-epss*(activesize-gamma));
                else:
                    prob=0.5*np.exp(epss*(activesize-gamma));
                if np.random.binomial(1,prob)>0.5 :
                    tempobj = {}
                    tempobj['coordinate'] = thegrid[j]['coordinate']+side/2*keyset[i];
                    tempobj['points']=valset[i];
                    newgrid.append(tempobj);
        thegrid=newgrid;
        depth=depth+1;
    return gridpoints
###### CANDIDATE SCRIPT ###### 

def candidate(X_data,n,p,epsilon, side_length, T):
    # Find candidate centers given data points.
    #print('  What candidates received as eps: {}'.format(epsilon))
    #print('  What candidates received as T: {}'.format(T))
    candidates=[];
    newpart=partition(X_data,n,p,epsilon/T, side_length);
    candidates.append(newpart);
    #########################################
    ###### THIS IS TO MAKE EQUAL TO MATLAB: REMOVE LATER
    #T=2
    #########################################
    #########################################
    print('     {}   -  trials for candidate sets'.format(T))
    for t in range(T):
        if t % 500 ==0:
        	print('     {}-th trial for candidate set\n'.format(t));
        offset=np.random.uniform(-side_length/2,side_length/2,p);
        #print('             offset: {}'.format(offset))
        shifted=X_data.T+offset
        newpart=partition(shifted.T,n,p,epsilon/T, side_length);
        newpart=[newpart[i]-offset for i in range(len(newpart))]
        candidates.append(newpart)
    return np.concatenate(candidates)

def sample_discrete(A):
    probs = A.flatten()/np.sum(A)
    samp = np.random.choice(A.size, size = 1, p=probs)
    row = samp//A.shape[1]
    col = samp%A.shape[1]
    return row,col

def localsearch( data,candidate,n,p,k,epsilon ,delta,RANGE, fast):
    #Private local search k-means given candidate centers.
    #p is dimension
    #global range;
    m =len(candidate)
    print('{} candidates, {} clusters'.format(m,k))
    # w[i,j] <- distance from point j to center i
    weightmat=np.zeros((m,n))
    Lambda=2*RANGE;
    # Estimate distance to candidates
    for j in range(m):
        weightmat[j,:]=np.sum((candidate[j].reshape((p,1))-data)**2, axis=0)
        
    # initialize output
    
    # if not enough clusters then repeat!!!!!
    centerid=np.random.choice(m, size=k)
    
    # in paper, T = 100*klog(n/delta). This can be huge... have a better T
    #########################################
    ###### THIS IS TO MAKE EQUAL TO MATLAB: REMOVE LATER
    # T=k;
    # if(T>20):
    #     T=20;
    # AFTER PLAYING UNCOMMENT THIS
    #T = min(20,int(100*k*np.log(n/delta)))
    T = min(100,int(100*k*np.log(n/delta)))
    if fast:
    	T = 5

    #########################################
    #########################################
    
    recordid=np.zeros((k,T), dtype=np.int)
    recordloss=np.zeros((T,1))
    print(' Total iterations local search: {}'.format(T))
    loss=sum(np.min(weightmat[centerid], axis = 0))
    for it in range(T):
        if it % 500 ==0:
            print('{}-th iteration for local search\n'.format(it))
        gains=np.zeros((k,m));
        for i in range(k):
            for j in range(m):
                tmpcenterid=centerid.copy();
                tmpcenterid[i]=j;
                newloss=sum(np.min(weightmat[tmpcenterid], axis = 0))
                gains[i,j]=newloss-loss;
        raw_exp=np.exp(-epsilon*gains/((Lambda**2)*(T+1)));
        i,j=sample_discrete(raw_exp);
        centerid[i]=j;
        recordloss[it]=sum(np.min(weightmat[centerid], axis = 0))
        loss=recordloss[it];
        recordid[:,it]=centerid;
    it,_ = sample_discrete(np.exp(-epsilon*recordloss/((Lambda**2)*(T+1))))
    centerid=recordid[:,it].flatten();
    #print(centerid)
    centers=candidate[centerid, :];
    return centers.T
def recover( X_data,d,epsilon, RANGE):
    #Recover the centers privately from a given cluster

    if(X_data.size ==0):
        print(' found emty set \n')
        z=np.random.uniform(-RANGE,RANGE,size=d)
        randi = 1
    else:
        print('    found NON empty set \n')
        z=np.mean(X_data,axis = 1);
        z=z+np.random.exponential(RANGE/(epsilon*X_data.shape[1]),d)*(2*np.random.binomial(1,0.5,size = d)-1)
        randi = 0
    return z, randi



def recover_sparse(X_data, mu_mean, side_length, s, epsilon, delta, eta = 1, noise = 'bin'):
    n = max(X_data.shape[1],1)
    I = list(np.arange(X_data.shape[0]))
    mu_mean = mu_mean.flatten()
    #maxi = int(np.ceil(2*s/eta))
    maxi = int(np.ceil(s/eta))
    #print('  Maxi: {}'.format(maxi))
    v=np.zeros(X_data.shape[0])

    probs = np.exp(epsilon*eta*n*np.abs(mu_mean[I])/(4*side_length*s)).flatten()
    probs = probs/np.sum(probs)
    samp = np.random.choice(I, size = maxi, p=probs)
    if noise=='bin':
        v[samp]=(mu_mean[samp] + np.random.binomial(1,p=1/(1+epsilon), size=maxi)[0])%2
    else: 
        v[samp]=mu_mean[samp] + np.random.exponential(4*side_length*s/(epsilon*n*eta), size=maxi)*(2*np.random.binomial(1,0.5, size=maxi)-1)
    # for j in range(maxi):
    #     probs = np.exp(epsilon*eta*n*np.abs(mu_mean[I])/(4*side_length*s)).flatten()
    #     #print(probs)
    #     probs = probs/np.sum(probs)
    #     samp = np.random.choice(I, size = 1, p=probs)
    #     I.remove(samp)
    #     if noise=='bin':
    #     	v[samp]=(mu_mean[samp] + np.random.binomial(1,p=1/(1+epsilon), size=1)[0])%2
    #     else: 
    #     	v[samp]=mu_mean[samp] + np.random.exponential(4*side_length*s/(epsilon*n*eta))*(2*np.random.binomial(1,0.5)-1)

    return v


    ###### CLUSTERING SCRIPT ###### 

# Global variables:
#  RANGE=l_2 radius of the data space. Need to specify before using.
#  side_length>2*l_infty norm of data space. Need to specify before using.

# x_data should be d*n matrix, where each column is a data point
# k: number of clusters.
# epsilon: privacy parameter
# delta: failure probability, not used in current implementation (assume constant failure probability)

def clustering( x_data,k,epsilon,delta,  RANGE,side_length,JLcoef, T = None, fast = False):
    d,n = x_data.shape
    if T == None:
        T = int(2*np.log(1/delta))
    if fast:
    	T=1
    sparsity = np.amax(np.sum(x_data>0, axis = 0))
    print(' sparsity:  {}'.format(sparsity))
    #Verify range
    mu_mean = np.mean(x_data, 1).reshape((d,1))
    print(' DATA SHAPE: {}'.format(x_data.shape))
    tmpR = np.sqrt(np.amax(np.sum((x_data-mu_mean)**2, 0)))
    if RANGE<tmpR:
        RANGE=tmpR
    print('Range : {} \n'.format(RANGE))
    results = [{} for i in range(T)]
    loss_iter =np.zeros((T,1))

    #########################################
    ###### THIS IS TO MAKE EQUAL TO MATLAB: REMOVE LATER
    epsilon = epsilon/T
    #########################################
    #########################################
    for iterat in range(T):
        print(' Round: {}   of  {} \n'.format(iterat+1, T))
        # Theoretical is 8*log(n). On the paper they do state they use log(n)/2
        p=np.floor(JLcoef*np.log(n)).astype(np.int)
        if(d<=5):
            p=d;
            G_projection=np.eye(d);
        else:
            G_projection=np.random.normal(0,1,size = (p,d))/np.sqrt(p);
        print('   Projecting in dim {}'. format(p))
        # From JL lemma, normal projection preserves distances with high probability
        y_projected=np.matmul(G_projection,x_data-mu_mean)
        print(' PROJECTED  SHAPE: {}'.format(y_projected.shape))
        RANGE_P = np.sqrt(np.amax(np.sum(y_projected**2, 0)))
        SIDE_P = np.amax(np.abs(y_projected))
        print('range projected:  {}, side_length proj {}'.format(RANGE_P, SIDE_P))
        
        
        #print('      T_can:  {} '.format(T_cand))
        #########################################
        ###### THIS IS TO MAKE EQUAL TO MATLAB: REMOVE LATER
        eps_cand = 2*epsilon/3
        eps_cand = epsilon/(6*T)
        T_cand = int(27*k*np.log(n/delta))
        if fast:
        	T_cand = 1
        #T_cand = 2

        c_candidates=candidate(y_projected,n,p,eps_cand, side_length=SIDE_P , T=T_cand);
        print('   Candidate set finished. Produced {} candidates\n'.format(c_candidates.shape))
        #print([np.amin(c_candidates[i]) for i in range(len(c_candidates))])
        #########################################
        ###### THIS IS TO MAKE EQUAL TO MATLAB: REMOVE LATER
        eps_search = epsilon/12
        #eps_search = epsilon/(6*T)
        u_centers=localsearch(data = y_projected, candidate = c_candidates, \
                                n = n,p = p,k = k,epsilon = eps_search, \
                                delta=delta, RANGE=RANGE_P, fast = fast);
        print('   Local search finished.\n');
        clusters = [[] for i in range(k)]
        for i in range(n):
            minval=1e100;
            minindex=0;
            assign = np.argmin(np.linalg.norm(u_centers - y_projected[:,i].reshape((p,1)), axis=0))
            clusters[assign].append(i)

        z_centers=np.zeros((d,k));
        totalloss=0;
        
        for j in range(k):
            #########################################
            ###### THIS IS TO MAKE EQUAL TO MATLAB: REMOVE LATER
            # They use epsilon/6... I came back to paper... I am using MORE NOISE HERE
            #eps_rec = epsilon/(24*T)
            eps_rec = epsilon/6

            #z_centers[:,j], _=recover(x_data[:,clusters[j]],d,eps_rec, RANGE);
            if len(clusters[j]) ==0:
            	mu_mean_j = np.zeros(d)
            else:
            	mu_mean_j = np.mean(x_data[:,clusters[j]], 1)

            z_centers[:,j] = recover_sparse(x_data[:,clusters[j]], mu_mean_j, side_length=side_length, s=sparsity, epsilon=epsilon, delta=delta, eta = 1, noise='')
            totalloss=totalloss+sum(sum((x_data[:,clusters[j]]-z_centers[:,j].reshape((d,1)))**2));

        print('Starting Lloyd.\n');
        nLloyd=3;
        #For Lloyd iteration
        for lloyditer in range(nLloyd):
            rands = 0
            clusters=[[] for i in range(k)]
            for i in range(n):
                minval=1e100;
                minindex=0;
                assign = np.argmin(np.linalg.norm(z_centers - x_data[:,i].reshape((d,1)), axis=0))
                clusters[assign].append(i)
            z_centers=np.zeros((d,k));

            totalloss=0;
            for j in range(k):
                #z_centers[:,j], randi=recover(x_data[:,clusters[j]],d,epsilon/(2*nLloyd), RANGE);
                #if randi==0:
                #    print('        center {} in lloyd {}  is not random'.format(j, lloyditer))
                #rands+=randi
                z_centers[:,j] = recover_sparse(x_data[:,clusters[j]], mu_mean, side_length=side_length, s=sparsity, epsilon=epsilon, delta=delta, eta = 1, noise='')
                totalloss=totalloss+sum(sum((x_data[:,clusters[j]]-z_centers[:,j].reshape((d,1)))**2));
        print('        \n There are {} random centers     \n'.format(rands))
        results[iterat]['z_centers']=z_centers;
        results[iterat]['clusters']=clusters;
        results[iterat]['c_candidates']=c_candidates;
        results[iterat]['u_centers']=u_centers;
        loss_iter[iterat]=totalloss;
        print('Lloyd finished\n');

        #return loss_iter, results
        
    prob = np.maximum(0.0001,np.exp(-epsilon*loss_iter/12))
    iteration,_=sample_discrete(prob);
    
    # look here for recentering...
    #z_centers=results[iteration[0]]['z_centers']+mu_mean
    z_centers=results[iteration[0]]['z_centers']

    clusters=results[iteration[0]]['clusters']
    c_candidates=results[iteration[0]]['c_candidates']
    u_centers=results[iteration[0]]['u_centers']
    L_loss=loss_iter[iteration[0]]
        
    return z_centers, clusters,c_candidates,u_centers,L_loss
def eval_pred(data, X, xhat):
    data['prediction'] = xhat[data.patient_int.values, data.item_int.values]
    data['pred_rank']=np.nan
    counter = 0
    u_test = np.unique(data.patient_int[data.rating_test])
    #print('\n Total number of users: {}'.format(len(u_test)))
    rmse_all = (data.prediction-data['count'])**2
    r_all = np.sqrt(np.mean((X-xhat)**2))
    r_train = np.sqrt(np.mean(rmse_all[~data.rating_test]))
    r_test = np.sqrt(np.mean(rmse_all[data.rating_test]))
    print( '\n RMSE:  {}  --  RMSE train: {}  --  RMSE test: {}  '.format(r_all, r_train,r_test))
    for i in u_test:
        data.pred_rank[np.logical_and(data.patient_int==i, data.rating_test)] = give_percentiles_all(i, data, xhat)
        counter+=1
        if counter%1000==0:
            temp = avg_rank(data['count'][~np.isnan(data.pred_rank)] , data.pred_rank[~np.isnan(data.pred_rank)] )
            print('users  {} --  Average ranking:  {}'.format(counter,temp ))
    #rmse_all = (data.prediction-data['count'])**2
    res = {'n_pat': X.shape[0],\
        'rmse_train_all': r_all, \
            'rmse_train': r_train,\
            'rmse_test': r_test, \
            'avg_rank': avg_rank(data['count'][~np.isnan(data.pred_rank)] , data.pred_rank[~np.isnan(data.pred_rank)] ) 
            }
    return res

def eval_pred_movies(data, X, xhat):
    data['prediction'] = xhat[data.uid.values, data.iid.values]
    data['pred_rank']=np.nan
    counter = 0
    u_test = np.unique(data.uid[data.rating_test])
    #print('\n Total number of users: {}'.format(len(u_test)))
    rmse_all = (data.prediction-data['count'])**2
    r_all = np.sqrt(np.mean((X-xhat)**2))
    r_train = np.sqrt(np.mean(rmse_all[~data.rating_test]))
    r_test = np.sqrt(np.mean(rmse_all[data.rating_test]))
    print( '\n RMSE:  {}  --  RMSE train: {}  --  RMSE test: {}  '.format(r_all, r_train,r_test))
    for i in u_test:
        data.pred_rank[np.logical_and(data.uid==i, data.rating_test)] = give_percentiles_all_m(i, data, xhat)
        counter+=1
        if counter%1000==0:
            temp = avg_rank(data['count'][~np.isnan(data.pred_rank)] , data.pred_rank[~np.isnan(data.pred_rank)] )
            print('users  {} --  Average ranking:  {}'.format(counter,temp ))
    #rmse_all = (data.prediction-data['count'])**2
    res = {'n_pat': X.shape[0],\
        'rmse_train_all': r_all, \
            'rmse_train': r_train,\
            'rmse_test': r_test, \
            'avg_rank': avg_rank(data['count'][~np.isnan(data.pred_rank)] , data.pred_rank[~np.isnan(data.pred_rank)] ) 
            }
    return res
