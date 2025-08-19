import numpy as np
import pickle
from fit import read_data, single, calib, sampling_corr, cellfit

points = [
    [0.228,      0.274,      0.237],
    [0.26579574, 0.31781294, 0.21495152],
    [0.25251492, 0.32230577, 0.22397445],
    [0.22601699, 0.2919778,  0.19083297],
    [0.2529353,  0.30630943, 0.22869999],
    [0.23387863, 0.27399787, 0.2287344],
    [0.21882232, 0.28886096, 0.19759439],
]

import matplotlib.pyplot as plt
def plot(df_list, save=None, ymin=5e-4, ymax=.3, omit_model=False, **kwargs):
    plt.figure()
    
    columns = ['amount', 'I', 'sim', 'drop', 'adsorb', 'model']
    styles = ['o', 'o', '-', '-', '-', '-']
    if "pcell" in kwargs:
        labels = ['measured', 'corrected', 'sampling model', 'droplets', 'adsorbed', 'cell model']
    else:
        labels = ['measured', 'corrected', 'sampling model', 'droplets', 'adsorbed', 'membrane model']
#   colors = ['blue', 'orange', 'green', 'red', 'magenta', 'brown']
    colors = ['#1f77b4', '#d62728', '#17becf', '#2ca02c', '#bcbd22', '#ff7f0e']

    if omit_model:
        z = zip(columns[:-1], styles[:-1], labels[:-1], colors[:-1])
    else:
        z = zip(columns, styles, labels, colors)
    for col, style, label, color in reversed(list(z)):
        for i, df in enumerate(df_list):
            lw = 3 if i == 0 else 1.0
            alpha = 1.0 if i == 0 else 0.6
            size = 6.0 if i == 0 else 3.0
            label = f"{label}" if i == 0 else None
            if style == 'o':
                plt.plot((df['sample'] - 3)*0.5, df[col], style, alpha=alpha, 
                        label=label, color=color, markersize=size)
            else:
                plt.plot((df['sample'] - 3)*0.5, df[col], style, linewidth=lw, 
                        label=label, color=color, alpha=alpha)

#    plt.legend(['measured','deconv','sim','droplets','immobil','fit'])
    plt.yscale('log')
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.xlabel('time [h]', fontsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel('amount [mg]', fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], fontsize=12)
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def proc(basename, target, csv=False, **kwargs):
    df,ap=read_data(basename+'.csv', **kwargs)
    results=[]
    for i, (kon, koff, q) in enumerate(points):
        p = {"kon": kon, "koff": koff, "q": q}
        r=target(**p, df=df)
        if "pcell" in kwargs:
            df = r["df"]
            r=cellfit(kwargs["pcell"], df=df)
        if i==0 and csv:
            r["df"].to_csv(basename+'-fit.csv', index=False, float_format="%.3e" )
        results.append(r["df"].copy(deep=True))
        print(r)
    plot(results,save=basename+'-fit.png', **kwargs)



#proc('../chlq-240705-tw', single)
#proc('../chlq-240919-tw', single)
#proc('../chlq-240918-tw', single, ymin=1e-3)
#proc('../chlq-240924-calib', calib, ymax=2, omit_model=True)
proc('../chlq-240724-caco2', sampling_corr, 
        pcell={"c_A0": 1.4, "P1":.08, "h":2, "m":0.01}, 
        ymax=1e-1, ymin=1e-3, csv=True)


