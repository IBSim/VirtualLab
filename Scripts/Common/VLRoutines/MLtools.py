
import numpy as np
import matplotlib.pyplot as plt

from Scripts.Common.ML import ML


def PCA_Sensitivity(VL,DataDict):

    Parameters = DataDict['Parameters']

    TrainIn, TrainOut = ML.VLGetDataML(VL,Parameters.TrainData)
    TestIn, TestOut = ML.VLGetDataML(VL,Parameters.TestData)

    # centre data to mean 0, stdev 1
    ScalePCA = ML.ScaleValues(TrainOut,'centre')
    TrainOut = ML.DataScale(TrainOut,*ScalePCA)
    TestOut = ML.DataScale(TestOut,*ScalePCA)

    U,s,VT = ML.PCA(TrainOut,centre=False)

    thresholds = [0.99,0.999]
    threshold_ix = ML.PCA_threshold(s,thresholds)
    # convergence_ix = ML.PCA_recon_convergence(train_recon_score)
    # print(convergence_ix)

    train_recon_score, test_recon_score = ML.PCA_sensitivity(VT,TrainOut,TestOut)

    x = list(range(1,len(s)+1))
    fig,ax = plt.subplots()
    ax.plot(x,train_recon_score,linestyle='-.',c='k',label='Train data')
    ax.plot(x,test_recon_score,linestyle='-',c='k',label='Test data')
    ax.plot([x[0],x[-1]],[test_recon_score[-1]]*2,c='r',linestyle=':')

    ylim = ax.get_ylim()
    ceil = np.ceil(max(train_recon_score[0],test_recon_score[0])/10)*10

    ylim = [1e-8,ceil]
    for _threshold_ix,_threshold in zip(threshold_ix,thresholds):
        plt.plot([_threshold_ix]*2,ylim,linestyle='--',c='0.5')
        v = '{}\n({})'.format(_threshold,_threshold_ix)
        ax.annotate(v,(_threshold_ix,ceil/10))

    ax.set_xlabel('No. PC')
    ax.set_ylabel('Reconstruction error')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(ylim)
    ax.legend()
    fig.savefig("{}/PCA_Sensitivity.png".format(DataDict['CALC_DIR']))
    plt.close()

