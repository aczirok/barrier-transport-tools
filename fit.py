import numpy as np
from model_v2 import sampling, membrane, cell, fit

###########################
#     data
import pandas as pd
def read_data(filename, corr1=None, corr2=None, **kwargs):

    df = pd.read_csv(filename, header=None)   #comma-delimited, corrected spectroscopy data
    df = df.rename(columns={0:'sample', 1:'cc'})    
    if corr1: df['cc'] += corr1
    if corr2: df['cc'] *= corr2
   
    
    try:
        #apical volume is 0.25 mL
        apical_amount = (df.loc[df['sample'] == 'AP', 'cc'].iloc[0]) * 0.25
    except:
        apical_amount = 0

    # Keep only rows where `samples` can be converted to integers
    df = df[pd.to_numeric(df['sample'], errors='coerce').notna()]
    df['sample'] = df['sample'].astype(int)

    #drop first two samples
    df = df.loc[df['sample'] > 2, ['sample','cc']].reset_index(drop=True)

    #convert cc [uM] to amount [nmol], volume is 1.2 mL
    df['amount'] = df['cc']*1.2
    return(df, apical_amount)

###########################
#     fits
def flatlog(x, x0=1e-3):
    x = np.asarray(x)
    return np.log(np.maximum(x, x0))

def score(output, target):
    N=min(len(output), len(target))
    return np.sum( (flatlog(output[:N]) - flatlog(target[:]))**2 )

def score2(output, target):
    return output

def constrain(p):
    return any(i < 1e-6 for i in p)

def constrain2(p):
    return any(i < 1e-5 or i > .9 for i in p)

###########################
#     specific fitting procedures
def apply_sampling_fit(df, kon, koff, q):
    """Apply sampling fit and populate shared DataFrame fields."""
    samples = df['amount'].to_numpy()
    r = fit(sampling, samples, score, list(samples), constrain=constrain,
            pfix={"kon": kon, "koff": koff, "q": q})
    df['I'] = pd.Series(r["params"])
    df['sim'] = pd.Series(r["output"])
    df['drop'] = pd.Series([x[0] for x in r["hidden"]])
    df['adsorb'] = pd.Series([x[1] for x in r["hidden"]])
    return r, df

def apply_membrane_fit(df):
    r=fit(membrane, df['I'].to_numpy(), score, {"c_A0":0.1, "P0":.05}, constrain=constrain)
    df['model'] = pd.Series(r["output"])
    return r, df

def apply_calib_fit(kon, koff, q, df):
    samples = df['amount'].to_numpy()
    
    def sampling2(I0,x0=(0,0),kon=0.1, koff=0.1,q=0.1):
        input = np.concatenate([I0, np.zeros(len(samples)-1)])
        return( 
            sampling(input, kon=kon*0.6, koff=koff*0.6, q=q*0.6) )
            
        
    r = fit(sampling2, samples, score, [samples[0]/(1-q)], constrain=constrain,
            pfix={"kon": kon, "koff": koff, "q": q})
    df['I'] = pd.Series(
                np.concatenate([r["params"], np.zeros(len(samples)-1)]))
    df['sim'] = pd.Series(r["output"])
    df['drop'] = pd.Series([x[0] for x in r["hidden"]])
    df['adsorb'] = pd.Series([x[1] for x in r["hidden"]])
    df['model'] = df['I']
    return r, df

def make_result(df, r, extra_keys=[]):
    result = {
        "df": df,
        "output": r["error"],
        "model_params": r["params"]
    }
    for key in extra_keys:
        if key in r:
            result[key] = r[key]
    return result

def single(kon, koff, q, df):
    _, df = apply_sampling_fit(df, kon, koff, q)
    r, df = apply_membrane_fit(df)
    return make_result(df, r)

def calib(kon,koff,q,df):
    r,df = apply_calib_fit(kon,koff,q,df)
    return make_result(df, r)

def sampling_corr(kon,koff,q,df):
    r,df = apply_sampling_fit(df,kon,koff,q)
    return make_result(df, r)

def multi(kon, koff, q, dfs):
    ret = {"error":0, "errors":[], "params":[]}
    ret_dfs=[]
    
    for i,df in enumerate(dfs):
        if i > 0:
            r, df = apply_sampling_fit(df, kon, koff, q)         
            r, df = apply_membrane_fit(df)
        else:
            r, df = apply_calib_fit(kon, koff, q, df)
        ret["error"] += r["error"]
        ret["errors"].append(r["error"])
        ret_dfs.append(df)
        ret["params"].append(r["params"])
        
    return make_result(ret_dfs, ret, extra_keys=["errors"])


def cellfit(p, df):
    r=fit(cell, 
            target=df['I'].to_numpy(), 
            score=score, 
            pvar=p, 
            constrain=constrain,
            pfix={"P0":0.061}, 
            logfile="/tmp/fitlog", 
            method='Nelder-Mead')
    df['model'] = pd.Series(r["output"])
    return make_result(df, r)

def cellfit2(p, df):
    r=fit(cell, 
            target=df['I'].to_numpy(), 
            score=score, 
            pvar={"c_A0":1.4}, 
            constrain=constrain,
            pfix={"P0":0.061, **p}, 
            method='Nelder-Mead')
    df['model'] = pd.Series(r["output"])
    return make_result(df, r)
