from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import numpy as np
def give_percentiles_m( scores):
    """ Give percentiles for each number on list of scores.
      Arguments:
        scores: 
      Returns:
        list of percentile of each score (range 0-1)
      Raises:
      """
    res = []
    for i in scores:
        res.append( stats.percentileofscore(scores,i))
    return np.array(res)/100

def avg_rank(real, percentiles):
    """ Estimate how well is a ranking
    Arguments:
    real: real values (number of times an exam/lab was performed)
    percentiles:  predicted percentile
    Returns:
    laverage ranking (float). If bad (random ranking) is 0.5. The closer to 0 the better
    Raises:
    """
    return np.sum(real*percentiles)/np.sum(real)

def process_test_u(data, u):
    """ select test indices for user u in dataset data
    Arguments:
    data: dataset with labs and meds for all users
    u: int_id of user
    Returns:
    5 indices to use as test
    Raises:
    """
    indexes = data.loc[data.uid==u].index.tolist()
    n = np.random.choice(indexes, size = 5, replace = False).tolist()
    return n
    




def give_percentiles_all_m(i, data,X):
    """ Give percentile of TEST samples for a user. Takes all labs/meds into account
      Arguments:
        i: int id of the patient (corresponding row in X)
        data: whole dataset
      Returns:
        list with percentiles for test samples. (close to 0: highly recommended - close to 1: below the rest)
      Raises:
    """
    sc = -X[i,:]
    it_idx = data.iid[np.logical_and(data.uid==i,data.rating_test)]
    res = give_percentiles_m(sc)
    return res[it_idx]

def process_data(data):
	#  """ Give index 0-n for subset size m users, n items, 
	#  	select 20% users for test, and 5 random entries for those users
	#   Arguments:
	#     data: subset of sparse matrix
	#   Returns:
	#     data processed
	#   Raises:
	# """
	patients_ids = data.uid.unique()
	patients_dict = {patients_ids[i]:i for i in range(patients_ids.shape[0])}
	items_ids = data.iid.unique()
	items_dict = {items_ids[i]:i for i in range(items_ids.shape[0])}

	data['uid']  =[patients_dict[i] for i in data.uid]
	data['iid'] = [items_dict[i] for i in data.iid]
	n_pat = len(patients_ids)
	u_train = np.random.choice(n_pat, size=int(0.8*n_pat), replace=False)
	u_test = np.delete(np.arange(n_pat), u_train)
	u_train.sort()
	data['user_train'] = np.in1d(data.uid.values, u_train)
	data['rating_test'] = False
	test_ratings = []
	for i in u_test:
	    test_ratings+=process_test_u(data,i)
	data.rating_test[test_ratings]=True
	return data 

def process_idx_users_items(data, users = True, items = True):
    if users: 
        patients_ids = data.uid.unique()
        patients_dict = {patients_ids[i]:i for i in range(patients_ids.shape[0])}
        data['uid']  =[patients_dict[i] for i in data.uid]
    if items:
        items_ids = data.iid.unique()
        items_dict = {items_ids[i]:i for i in range(items_ids.shape[0])}
        data['iid'] = [items_dict[i] for i in data.iid]    
    return data 





def process_data_synth(data):
    patients_ids = data.uid.unique()
    patients_dict = {patients_ids[i]:i for i in range(patients_ids.shape[0])}

    items_ids = data.iid.unique()
    items_dict = {items_ids[i]:i for i in range(items_ids.shape[0])}

    data['uid']  =[patients_dict[i] for i in data.uid]
    #data['iid'] = [items_dict[i] for i in data.iid]
    n_pat = len(patients_ids)
    u_train = np.random.choice(n_pat, size=int(0.8*n_pat), replace=False)
    u_test = np.delete(np.arange(n_pat), u_train)
    u_train.sort()
    data['user_train'] = np.in1d(data.uid.values, u_train)
    data['rating_test'] = False
    test_ratings = []
    for i in u_test:
        test_ratings+=process_test_u(data,i)
    data.rating_test[test_ratings]=True
    return data 


def create_uid(data,bool_indi = []):
    if len(bool_indi)==0:
        bool_indi = np.repeat(True, data.shape[0]).tolist()
    patients_ids = data.uid[bool_indi].unique()
    patients_dict = {patients_ids[i]:i for i in range(patients_ids.shape[0])}
    data.new_uid[bool_indi]=[patients_dict[i] for i in data.uid[bool_indi]]
    return data

def train_eval(data, n_comp, reg, total_items):
    #  """ use NMF on data and return metrics, 
    #   Arguments:
    #     data: pd dataframe with matrix indices, values and train/test indicator
    #     n_comp: dimension we are factorizing - number of latent factors
    #     reg: level of regularization
    #     total_items: dimension space of signals (number of items reviewed)
    #   Returns:
    #     data processed
    #   Raises:
    # """
    n_train = np.sum(~data.rating_test)
    sparse_patients_labs = csr_matrix((data['count'][~data.rating_test], (data.uid[~data.rating_test], data.iid[~data.rating_test])), shape=(data.uid.unique().shape[0], total_items))
    model_public = NMF(n_components=n_comp, alpha=reg, init='nndsvd')
    u_public= model_public.fit_transform(sparse_patients_labs)
    print('\n Finished factorizing')
    n_test = np.sum(data.rating_test)
    print('\n Finished factorizing')
    X = np.matmul(u_public, model_public.components_)

    data['prediction'] = X[data.uid.values, data.iid.values]
    data['pred_rank']=np.nan
    counter = 0
    u_test = np.unique(data.uid[data.rating_test])
    print('\n Total number of test users: {}'.format(len(u_test)))
    rmse_all = (data.prediction-data['count'])**2
    r_all = np.sqrt(model_public.reconstruction_err_/n_train)
    r_train = np.sqrt(np.mean(rmse_all[~data.rating_test]))
    r_test = np.sqrt(np.mean(rmse_all[data.rating_test]))
    print( '\n RMSE:  {}  --  RMSE train: {}  --  RMSE test: {}  '.format(r_all, r_train,r_test))
    for i in u_test:
        data.pred_rank[np.logical_and(data.uid==i, data.rating_test)] = give_percentiles_all(i, data, X)
        counter+=1
        if counter%1000==0:
            temp = avg_rank(data['count'][~np.isnan(data.pred_rank)] , data.pred_rank[~np.isnan(data.pred_rank)] )
            print('users  {} --  Average ranking:  {}'.format(counter,temp ))
    res = {'rmse_train_all': r_all, \
    		'rmse_train': r_train,\
            'rmse_test': r_test, \
            'avg_rank': avg_rank(data['count'][~np.isnan(data.pred_rank)] , data.pred_rank[~np.isnan(data.pred_rank)] ) 
            }
    return res
