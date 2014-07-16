from mvpa2.measures import rsa
from mvpa2.mappers.fx import mean_group_sample
import mypymvpa.utilities.visualization as viz
import mypymvpa.utilities.stats as mus
import scipy.stats as sst
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mypymvpa.utilities.hardcodedexpparams import FGEcondmapping
from copy import deepcopy
from mypymvpa.analysisobjs.dataset_funcs import preprocess, prepconfigV2
import mypymvpa.utilities.misc as mum
import itertools

global abb
abb=FGEcondmapping

class RSAresult():
    '''defines single neural RDM. methods for comparing to models and saving result output'''
    def __init__(self, neuralRDM, roi, subjid, ftcRDM=None):
        self.rdm=neuralRDM
        self.fulltimecourserdm=ftcRDM
        self.roi=roi
        self.subjid=subjid
        self.modelcorrs={}
    def comparemodels(self, modelRDMs):
        models=modelRDMs.keys()
        for m in models:
            mrdm=modelRDMs[m]
            tau,p=mus.kendallstau(self.rdm, mrdm, symmetrical=True)
            self.modelcorrs[m]=tau
    def save(self, subjroot, subdir, filename):
        filename=subjroot+subdir+filename
        if not os.path.exists(subjroot):
            os.mkdir(subjroot)
        if not os.path.exists(subjroot+subdir):
            os.mkdir(subjroot+subdir)
        with open(filename, 'wb') as output:
            pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)

class ROIsummary():
    '''defines summary of results for single ROI'''
    def __init__(self, roi, modelRDMs):
        self.roi=roi
        self.grn=None
        self.ftcgrn=None
        self.modelRDMs=modelRDMs
        self.grn2models={modelname:{'tau':None, 'permutation_pval':None, 'bootstrap_CI':None, 'bootstrap_SEM':None} for modelname in modelRDMs}
        self.grnmodelcomparisons={}
        self.grn2modelsRFX={modelname:{'tau':None,'RFX_pval':None, 'RFX_SEM':None, 'RFX_WSSEM':None} for modelname in modelRDMs}
        self.grnmodelcomparisonsRFX={}
    def add_grn2models(self, modelname, tau, permutation_pval=None, bootstrap_CI=None, bootstrap_SEM=None):
        self.grn2models[modelname]={'tau':tau, 'permutation_pval':permutation_pval, 'bootstrap_CI':bootstrap_CI, 'bootstrap_SEM':bootstrap_SEM}
    def add_grnmodelcomparisons(self, comp, tau1,tau2, bootstrap_diff, bootstrap_pval, bootstrap_CI):
        self.grnmodelcomparisons[comp]={'tau1':tau1, 'tau2':tau2, 'bootstrap_pval':bootstrap_pval, 'bootstrap_CI': bootstrap_CI}
    def add_grn2modelsRFX(self, modelname, tau, zval, RFX_pval, RFX_SEM, RFX_WSSEM):
        self.grn2modelsRFX[modelname]={'tau':tau, 'zval':zval,'RFX_pval':RFX_pval, 'RFX_SEM':RFX_SEM, 'RFX_WSSEM':RFX_WSSEM}
    def add_grnmodelcomparisonsRFX(self, comp, tau1,tau2, zval, RFX_pval):
        self.grnmodelcomparisonsRFX[comp]={'tau1':tau1, 'tau2':tau2, 'zval': zval, 'RFX_pval':RFX_pval}
    def save(self, analdir, filename):
        filename=analdir+filename
        if not os.path.exists(analdir):
            os.mkdir(analdir)
        with open(filename, 'wb') as output:
            pickler=pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)

def prepforrsa(ds):
    '''creates separate RDMs in each CV fold'''
    mtcgs = mean_group_sample(['targets', 'chunks'])
    mtcds = mtcgs(ds)
    return mtcds

def prepforsinglerdm(ds):
    '''creates single RDM for whole dataset'''
    mtgs = mean_group_sample(['targets'])
    mtds = mtgs(ds)
    return mtds

def singlesubjanalysis(e, disc, roilist, subjects, runthemnowlist, runindsubjects, configspec, modelRDMs, conditions, savetag, singlesubjplot=True):
    sel=e.selectors.keys()[0]
    grouprdms={}
    groupftcrdms={}
    for roi in roilist:
        print roi
        grouprdms[roi]=[]
        groupftcrdms[roi]=[]
        for subjectn, subject in enumerate(subjects):
            if subject.subjid in runthemnowlist:
                if runindsubjects:
                    print "working on subject " +subject.subjid
                if roi in subject.rois.keys():
                    cfg=prepconfigV2(e,detrend=configspec['detrend'],zscore=configspec['zscore'], averaging=configspec['averaging'],removearts=configspec['removearts'], hpfilter=configspec['hpfilter'], clfname=configspec['clfname'], featureselect=configspec['featureselect'])
                    if runindsubjects:
                        dataset=subject.makedataset(disc, sel, roi)
                        dataset.cfg = cfg
                        dataset.a['cfg'] = cfg
                    subdir='subjresults_%s_%s_%s%s/' % (cfg.clfstring, e.task, e.boldsorbetas, savetag)
                    subjfilename=disc+'_'+sel+'_'+roi+'RSA.pkl'
                    if runindsubjects:
                        preppeddata=preprocess(dataset)
                        singletimecoursedata=prepforsinglerdm(preppeddata)
                        preppeddata=prepforrsa(preppeddata)
                        singleRDM=singletimecoursesimilarities(singletimecoursedata,conditions, subject.subjid, roi, distance=configspec['similarity'], transformed=configspec['transformed'], plotit=singlesubjplot)
                        rdm=crossrunsimilarities(preppeddata, conditions, subject.subjid, roi, distance=configspec['similarity'], transformed=configspec['transformed'], plotit=singlesubjplot)
                        rdmresult=RSAresult(rdm, roi, subject.subjid, ftcRDM=singleRDM)
                        rdmresult.comparemodels(modelRDMs)
                        rdmresult.save(subject.subjanalysisdir,subdir,subjfilename)
                        print "saved to %s%s%s" %(subject.subjanalysisdir,subdir,subjfilename)
                    else:
                        result=mum.loadpickledobject(subject.subjanalysisdir+subdir+subjfilename)
                        rdm=result.rdm
                        singleRDM=result.fulltimecourserdm
                    grouprdms[roi].append(rdm)
                    groupftcrdms[roi].append(singleRDM)
        print "finished %s, %s, %s" %(disc,sel,roi)
    return grouprdms, groupftcrdms, subdir

def euclideandistancematrix(matrix):
    '''take a matrix of items in rows and features in columns and return a similarity space of items'''
    dim=len(matrix)
    distancematrix=np.zeros((dim,dim))
    transformeddistancematrix=np.zeros((dim,dim))
    for vector1n,vector1 in enumerate(matrix):
        for vector2n,vector2 in enumerate(matrix):
            #ed=np.linalg.norm(np.array(vector1)-np.array(vector2))
            ed=similarity(vector1, vector2, metric='euclidean')
            distancematrix[vector1n][vector2n]=ed
            #transformeddistancematrix[vector1n][vector2n]=np.arctanh(rvalue)
    meandist=np.mean(distancematrix)
    stddist=np.std(distancematrix, ddof=1)
    for vector1n,vector1 in enumerate(matrix):
        for vector2n,vector2 in enumerate(matrix):
            ed=distancematrix[vector1n][vector2n]
            zscored_ed=(ed-meandist)/stddist
            transformeddistancematrix[vector1n][vector2n]=zscored_ed
    return distancematrix, transformeddistancematrix


def similarity(pattern1,pattern2, metric='pearsonr'):
    '''takes two arrays (e.g. neural patterns) and computes similarity between them'''
    if metric=='pearsonr':
        sim,p=sst.pearsonr(pattern1,pattern2)
    if metric=='euclidean':
        try:
            sqdiffs=[(p1-pattern2[pn])**2 for pn,p1 in enumerate(pattern1)]
            sim=sum(sqdiffs)
        except:
            sim=(pattern1-pattern2)**2
    return sim

def makemodelmatrices(conditions, rsamatrixfile, matrixkey, similarity='euclidean', itemsoremos='emos', visualizeinput=False, itemindices=None):
    '''make RDMs for each of the models. option to also visualize initial input matrix.'''
    with open(rsamatrixfile, 'r') as inputfile:
        rsaobj=pickle.load(inputfile)
    inputmatrix=rsaobj[matrixkey]['itemavgs']
    itemlabels=rsaobj[matrixkey]['itemlabels']
    item2emomapping=rsaobj['item2emomapping']
    #re order to align with conditions
    newdatamatrix=[]
    newaxis=[]
    for c in conditions:
        condlines=[line for linen,line in enumerate(inputmatrix) if abb[item2emomapping[itemlabels[linen]]]==c]
        condlabels=[itemlabels[linen] for linen,line in enumerate(inputmatrix) if abb[item2emomapping[itemlabels[linen]]]==c]
        newdatamatrix.extend(condlines)
        newaxis.extend(condlabels)
    datamatrix=newdatamatrix
    axis=newaxis
    #make plot for RDM
    if itemsoremos=='ind':
        if visualizeinput:
            if type(datamatrix[0]) in (np.float64, float):
                viz.simplebar(datamatrix, xticklabels=conditions, xtickrotation=90, figsize=[6,4], colors='orange', xlabel=matrixkey+'_inputmatrix')
            else:
                viz.simplematrix(datamatrix, yticklabels=axis, xlabel='%s_inputmatrix' %(matrixkey))
        if similarity=='correlation':
            RSAmat=1-np.corrcoef(datamatrix)
            plotmin=-1
            plotmax=1
        elif similarity=='euclidean':
            RSAmat_untransformed, RSAmat=euclideandistancematrix(datamatrix)
            RSAmat=-1*RSAmat #because zscored, just flip the sign for RSM instead of RDM
            plotmin=np.min(RSAmat)
            plotmax=np.max(RSAmat)
        else:
            print "%s is not a recognized similarity metric" %(similarity)
        viz.simplematrix(RSAmat, minspec=plotmin, maxspec=plotmax, xticklabels=axis, yticklabels=axis, colorspec= 'RdYlBu_r', xtickrotation=90, xlabel='%s_indRSA' %(matrixkey))
    elif itemsoremos=='emos':
        #make emo matrix
        emomatrix=[]
        for emo in conditions:
            lines=[line for linen,line in enumerate(datamatrix) if abb[item2emomapping[axis[linen]]]==emo]
            lines=np.mean(lines,0)
            try:
                lines=[el for el in lines]
            except:
                pass
            emomatrix.append(lines)
        if visualizeinput:
            if type(emomatrix[0]) in (np.float64, float):
                viz.simplebar(emomatrix, xticklabels=conditions, xtickrotation=90, figsize=[6,4], colors='orange', xlabel=matrixkey+'_inputmatrix')
            else:
                viz.simplematrix(emomatrix, yticklabels=conditions, xlabel='%s_inputmatrix' %(matrixkey))
        if similarity=='correlation':
            RSAmat=1-np.corrcoef(emomatrix)
        elif similarity=='euclidean':
            RSAmat_untransformed, RSAmat=euclideandistancematrix(emomatrix)
            RSAmat=-1*RSAmat #because zscored, just flip the sign for RSM instead of RDM
            plotmin=np.min(RSAmat)
            plotmax=np.max(RSAmat)
        else:
            print "%s is not a recognized similarity metric" %(similarity)
        viz.simplematrix(RSAmat, minspec=plotmin, maxspec=plotmax, xticklabels=conditions, yticklabels=conditions, colorspec= 'RdYlBu_r', xtickrotation=90, xlabel='%s_emoRSA' %(matrixkey))
    return RSAmat

def singletimecoursesimilarities(ds,conditions, subjectid, roi, distance='pearsonr', transformed=False, plotit=True):
    '''computes neural RDM within a single timecourse (diagonal necessarily 0)'''
    rdm=np.zeros([len(conditions), len(conditions)])
    for cn,c in enumerate(conditions):
        pattern1=[sample for samplen,sample in enumerate(ds.samples) if ds.sa.targets[samplen]==c][0]
        for c2n,c2 in enumerate(conditions):
            pattern2=[sample for samplen,sample in enumerate(ds.samples) if ds.sa.targets[samplen]==c2][0]
            sim=similarity(pattern1,pattern2, metric=distance)
            rdm[cn][c2n]=1-sim
    if distance=='pearsonr':
        if transformed:
            rdm=transformsimilarities(rdm, distance)
            clim=[np.min(rdm), np.max(rdm)]
        else:
            clim=[-1,1]
    elif distance=='euclidean':
        if transformed:
            rdm=transformsimilarities(rdm, distance)
            clim=[np.min(rdm), np.max(rdm)]
        else:
            clim=[np.min(rdm), np.max(rdm)]
    if plotit:
        viz.plot_simmtx(rdm, conditions, '%s: %s single timecourse pattern distances (%s)' %(subjectid,roi,distance), clim=clim, axislabels=['conditions', 'conditions'])
    return rdm

def crossrunsimilarities(ds,conditions, subjectid, roi, distance='pearsonr', transformed=False, plotit=True):
    '''compute neural RDM across runs (diagonal is interpretable)'''
    folds=list(set(ds.sa.chunks))
    if len(folds)>2:
        print "warning: this code assumes only two folds. you have more than two"
        print breakit
    rdm=np.zeros([len(conditions), len(conditions)])
    for cn,c in enumerate(conditions):
        pattern1=[sample for samplen,sample in enumerate(ds.samples) if ds.sa.chunks[samplen]==folds[0] and ds.sa.targets[samplen]==c]
        for c2n,c2 in enumerate(conditions):
            pattern2=[sample for samplen,sample in enumerate(ds.samples) if ds.sa.chunks[samplen]==folds[1] and ds.sa.targets[samplen]==c2]
            if pattern1!=[] and pattern2!=[]:
                pattern1array=pattern1[0]
                pattern2array=pattern2[0]
                try:
                    sim=similarity(pattern1array,pattern2array, metric=distance)
                except:
                    sim=np.nan
                    print "warning: unexpectedly failed to compute kendal tau for the following 2 patterns"
            else:
                sim=np.nan
                #print "warning: you have nan values in your similarity matrix because some conditions were not found"
            dissim=1-sim
            rdm[cn][c2n]=dissim
    if distance=='pearsonr':
        if transformed:
            rdm=transformsimilarities(rdm, distance)
            clim=[np.min(rdm), np.max(rdm)]
        else:
            clim=[-1,1]
    elif distance=='euclidean':
        if transformed:
            rdm=transformsimilarities(rdm, distance)
            clim=[np.min(rdm), np.max(rdm)]
        else:
            clim=[np.min(rdm), np.max(rdm)]
    if plotit:
        viz.plot_simmtx(rdm, conditions, '%s: %s crossrun pattern distances (%s)' %(subjectid,roi,distance), clim=clim, axislabels=folds)
    return rdm

def relateRDMsgrn(roi_summary, modelRDMs, alphas=[.05, .01, .001], printit=True, plotpermutationfig=True, num_samples=None):
    '''computes relationship between single group neural RDM and each model. significance assessed using bootstrap and condition-permuting.'''
    models=modelRDMs.keys()
    taus=[]
    pvals=[]
    neuralRDM=roi_summary.grn
    for m in models:
        rdm=modelRDMs[m]
        tau,p=mus.kendallstau(neuralRDM, rdm, symmetrical=True)
        permMean, upperbound, lowerbound, nullrejected, realpval, exactp=singleRDMrelation_permutationtest(neuralRDM, rdm, tau, alphas, num_samples=num_samples, plotit=plotpermutationfig)
        string="%s-%s: tau=%.3f, p=%.3f (p%s) (p value from randomized permutation test)" %(roi_summary.roi,m,tau,exactp, realpval)
        if printit:
            print string
        taus.append(tau)
        pvals.append(exactp)
    bm=np.where(taus==np.max(taus))[0]
    if len(bm)>1:
        if printit:
            print "warning: multiple best models. taking just the first"
    bm=bm[0]
    bestmodel=models[bm]
    if printit:
        print "%s, bestmodel: %s" %(roi_summary.roi,bestmodel)
    roi_summary.bestmodel=bestmodel
    return taus, pvals, models, bestmodel, roi_summary

def sampledist(samplestatistics, num_samples, alpha, plotit=True, observed=None, ax=None):
    '''takes set of sample statistics and returns CI and SEM'''
    lowerbound=np.sort(samplestatistics)[int((alpha/2.0)*num_samples)]
    upperbound=np.sort(samplestatistics)[int((1-alpha/2.0)*num_samples)]
    if plotit:
        try:
            ax.hist(samplestatistics,20, color='blue')
        except:
            ax.hist(samplestatistics,len(samplestatistics), color='blue')
        ylim=plt.ylim()
        #print ylim
        ax.plot([lowerbound,lowerbound], ylim, 'k-', lw=2, alpha=.1)
        ax.plot([upperbound,upperbound], ylim, 'k-', lw=2, alpha=.1)
        if observed:
            xlim=list(plt.xlim())
            if xlim[0]>observed:
                xlim[0]=observed-.1
            if xlim[1]<observed:
                xlim[1]=observed+.1
            plt.xlim(xlim)
            ax.plot([observed,observed], ylim, lw=2, color='red')
    SEM=np.std(samplestatistics,ddof=1)
    return lowerbound, upperbound, SEM,ax

def testdist(observedmean, samplemeans, num_samples, alpha, tail='both',ax=None, plotit=True):
    '''takes set of samples, an observation, and an alpha, and returns whether null hypothesis is rejected at that alpha, as well as CI and SEM of distribution'''
    lowerbound, upperbound, SEM, ax = sampledist(samplemeans, num_samples, alpha, plotit=plotit,observed=observedmean, ax=ax)
    unders=np.array(samplemeans)<=observedmean
    overs=np.array(samplemeans)>=observedmean
    exactps=[float(np.sum(unders))/len(samplemeans),float(np.sum(overs))/len(samplemeans)]
    exactp=np.min(exactps)
    pdict={0:'>%s'%(alpha),1:'<%s'%(alpha)}
    if tail=='both':
        h=observedmean<lowerbound or observedmean>upperbound
    elif tail=='right':
        h=observedmean>upperbound
    elif tail=='left':
        h=observedmean<lowerbound
    pstr=pdict[h]
    return h,pstr,exactp, lowerbound, upperbound, SEM, ax

def singleRDMrelation_permutationtest(neuralRDM, modelRDM, observedtau, alphas, num_samples=None, plotit=False):
    neuralRDM=np.array(neuralRDM)
    modelRDM=np.array(modelRDM)
    rdmsize=np.shape(neuralRDM)
    if rdmsize != np.shape(modelRDM):
        print "warning: RDMs differ in size"
        print breakit
    samplecorrs=[]
    for b in range(num_samples):
        colidx=np.random.permutation(rdmsize[0])
        rowidx=np.random.permutation(rdmsize[0])
        colsshuffled=neuralRDM[:,colidx]
        shuffledrdm=colsshuffled[rowidx,:]
        rdmcorr,throwawayp=mus.kendallstau(shuffledrdm, modelRDM, symmetrical=True)
        samplecorrs.append(rdmcorr)
    permMean=np.mean(samplecorrs)
    #print "mean of permuted null distribution=%.8f" %(permMean)
    nullrejections=[]
    pvalstr='>%s' %(alphas[0])
    if plotit:
        f,ax=plt.subplots(figsize=[4,2])
    for alpha in alphas:
        if plotit:
            nullrejected, pval, exactp, lowerbound, upperbound, throwawaypermSEM,ax =testdist(observedtau, samplecorrs, num_samples, alpha,ax=ax, plotit=plotit)
        else:
            nullrejected, pval, exactp, lowerbound, upperbound, throwawaypermSEM,ax =testdist(observedtau, samplecorrs, num_samples, alpha, plotit=plotit)
        nullrejections.append(nullrejected)
        if nullrejected:
            pvalstr=pval
    if plotit:
        plt.show()
    nullrejected=any(nullrejections)
    return permMean, upperbound, lowerbound, nullrejected, pvalstr, exactp


def bootstrapinner(e,subject, disc, sel, roi, configspec, conditions,num_samples, modelRDMs):
    '''this resamples at the level of individual stimuli'''
    cfg=prepconfigV2(e,detrend=configspec['detrend'],zscore=configspec['zscore'], averaging=configspec['averaging'],removearts=configspec['removearts'], hpfilter=configspec['hpfilter'], clfname=configspec['clfname'], featureselect=configspec['featureselect'])
    dataset=subject.makedataset(disc, sel, roi)
    print sel
    print dataset.sa.chunks()
    print breakit
    dataset.cfg = cfg
    dataset.a['cfg'] = cfg
    preppeddata=preprocess(dataset)
    subjrdms=[]
    print "working on bootstrap (%s samples, resampling stimuli)" %(num_samples)
    for b in range(num_samples):
        idx=[eln for eln,el in enumerate(preppeddata.sa.targets) if el in conditions]
        rsampleidx=np.random.choice(idx, size=len(idx), replace=True)
        bssampledata=preppeddata[rsampleidx]
        bssampledata=prepforrsa(bssampledata)
        neuralrdm=crossrunsimilarities(bssampledata, conditions, subject.subjid, roi, distance='euclidean', transformed=True, plotit=False)
        subjrdms.append(neuralrdm)
    return subjrdms

def bootstrapfromconditions(disc, modelRDMs, grouprdmmean, num_samples=None, printit=True):
    if printit:
        print 'performing bootstrapping for errorbars (conditions)'
    shape=np.shape(grouprdmmean)
    idx=range(shape[0])
    bsmodelcorrs={m:[] for m in modelRDMs.keys()}
    for m in modelRDMs.keys():
        for b in range(num_samples):
            resampledgrouprdm=np.empty(shape)
            resampledgrouprdm[:]=np.nan
            rsampleidx=np.random.choice(idx, size=len(idx), replace=True)
            for x in rsampleidx:
                for y in rsampleidx:
                    resampledgrouprdm[x,y]=grouprdmmean[x,y]
            corr,throwawayp=mus.kendallstau(resampledgrouprdm, modelRDMs[m], symmetrical=True)
            bsmodelcorrs[m].append(corr)
    return bsmodelcorrs


def bootstrapfromstimuli(e, disc, roi, subjects, configspec, modelRDMs, conditions, num_samples=None):
    '''bootstraps at the level of stimuli in individual subjects'''
    print 'performing bootstrapping for errorbars (stimuli)'
    print "warning: this probably isn't implemented the way you want it. don't use without reading and thinking about it."
    print "starting bootstrap for %s, %s" %(disc, roi)
    sel=e.selectors.keys()[0]
    neuralmtxs=[]
    grouprdms=[]
    for subjectn, subject in enumerate(subjects):
        print "working on subject " +subject.subjid
        if roi in subject.rois.keys():
            subjrdms=bootstrapinner(e,subject, disc, sel, roi, configspec, conditions,num_samples, modelRDMs)
            grouprdms.append(subjrdms)
    grouprdms=np.mean(grouprdms,0)
    bsmodelcorrs={m:[] for m in modelRDMs.keys()}
    #optimize this so that we don't reloop through everything this way
    for b in range(num_samples):
        for m in modelRDMs.keys():
            corr,throwawayp=mus.kendallstau(grouprdms[b], modelRDMs[m], symmetrical=True)
            bsmodelcorrs[m].append(corr)
    print "finished bootstrapping %s, %s, %s" %(disc,sel,roi)
    return bsmodelcorrs

def comparemodelRDMfits_bootstraptest(model1, model2, bsmodelcorrs, alpha=0.05):
    corr1=np.array(bsmodelcorrs[model1])
    corr2=np.array(bsmodelcorrs[model2])
    diffs=corr1-corr2
    bsMeanDiff=np.mean(diffs)
    bsSEMDiff=np.std(diffs,ddof=1)
    lowerbound, upperbound, SEM, throwawayax=sampledist(diffs, len(diffs), alpha, plotit=False)
    unders=np.array(diffs)<0
    overs=np.array(diffs)>=0
    exactps=[2*float(np.sum(unders))/len(diffs),2*float(np.sum(overs))/len(diffs)]
    pval=min(exactps)
    nullrejected=pval<alpha
    return bsMeanDiff, bsSEMDiff, upperbound, lowerbound, nullrejected, pval

def singlemodelRDM_bootstraperror(roi,modelname,bsmodelcorrs, alpha=0.05, plotit=False, observed=None, printit=True):
    corrs=np.array(bsmodelcorrs[modelname])
    bsMean=np.mean(corrs)
    bsSEM=np.std(corrs,ddof=1)
    if plotit:
        f,ax=plt.subplots(figsize=[4,2])
    else:
        ax=None
    lowerbound, upperbound, SEM, throwawayax=sampledist(corrs, len(corrs), alpha, plotit=plotit, ax=ax, observed=observed)
    if printit:
        print "%s-- %s: actualmean=%.3f, bsmean=%.3f, bsSEM=%.3f, CI=%s,-%s" %(roi,modelname,observed, bsMean, bsSEM, lowerbound, upperbound)
    return bsMean, bsSEM, upperbound, lowerbound

def transformsimilarities(repdismat, distance):
    '''return transformed similarity matrices'''
    rdm=deepcopy(repdismat)
    if distance=='euclidean':
        trdm=rdm*0
        meandist=np.nanmean(rdm)
        stddist=np.nanstd(rdm, ddof=1)
        dims=np.array(repdismat).shape
        for cn,c in enumerate(range(dims[0])):
            for c2n,c2 in enumerate(range(dims[1])):
                ed=rdm[cn][c2n]
                zscored_ed=(ed-meandist)/stddist
                trdm[cn][c2n]=zscored_ed
        rdm=trdm
    if distance=='pearsonr':
        rdm=np.arctanh(rdm)
    return rdm

def singlemodelRFX(e,roi_summary,models, subjects, subdir, disc, errorbars='withinsubj', printit=True):
    '''assess relation between single RDM and single model using RFX across subjects'''
    sel=e.selectors.keys()[0]
    modelsummaries={m:[] for m in models}
    for subject in subjects:
        if roi_summary.roi in subject.rois.keys():
            subjfilename=disc+'_'+sel+'_'+roi_summary.roi+'RSA.pkl'
            result=None
            with open(subject.subjanalysisdir+subdir+subjfilename, 'r') as resultfile:
                result=pickle.load(resultfile)
            for m in models:
                tau=result.modelcorrs[m]
                modelsummaries[m].append(tau)
    sems,withinsubjsems= plotRFXresults(roi_summary.roi,subjects, models, modelsummaries, errorbars='withinsubj', plotit=True)
    if printit:
        print "single model: 1-sided signed rank tests on tau correlations between neural and model RDMs (null hypothesis: tau=0)"
    for mn,m in enumerate(models):
        df=len(modelsummaries[m])-1
        mean=np.mean(modelsummaries[m])
        T,p,z=sst.wilcoxonAES(modelsummaries[m])
        if z>=0:
            p=p/2
        else:
            p=1-p/2
        if printit:
            string='%s: %s(%.2f): z(%.0f)=%.3f, p=%.3f.' % (roi_summary.roi,m,mean,df,z,p)
            print string
        roi_summary.add_grn2modelsRFX(m,mean,z,p,sems[mn], withinsubjsems[mn])
    return modelsummaries, roi_summary

def comparemodels(comparisontype, e, disc, roi_summary, subjects, modelsummaries, configspec, modelRDMs, conditions, num_samples=None, printit=True):
    bsmodelcorrs={}
    if comparisontype=='stimbootstrap':
        bsmodelcorrs=bootstrapfromstimuli(e, disc, roi_summary.roi, subjects, configspec, modelRDMs, conditions, num_samples=num_samples)
        if printit:
            print "model comparison bootstrap tests comparing tau correlations between neural and model RDMs (%s)" %(comparisontype)
    elif comparisontype=='condbootstrap':
        bsmodelcorrs=bootstrapfromconditions(disc, modelRDMs, roi_summary.grn, num_samples=num_samples)
        if printit:
            print "model comparison bootstrap tests comparing tau correlations between neural and model RDMs (%s)" %(comparisontype)
    elif comparisontype=='RFXsubjects':
        if printit:
            print "model comparison 2-sided signed rank tests comparing tau correlations between neural and model RDMs"
    models=modelRDMs.keys()
    comparisons=[el for el in itertools.combinations(range(len(models)),2)]
    for comp in comparisons:
        m1=models[comp[0]]
        m2=models[comp[1]]
        compname=m1+'_VS_'+m2
        print "working on %s (%s)" %(compname, comparisontype)
        array1=modelsummaries[m1]
        array2=modelsummaries[m2]
        if comparisontype in ('RFXsubjects','stimbootstrap'):
            mean1=np.mean(array1)
            mean2=np.mean(array2)
            meandiff=mean1-mean2
        elif comparisontype=='condbootstrap':
            mean1=roi_summary.grn2models[m1]['tau']
            mean2=roi_summary.grn2models[m2]['tau']
            meandiff=mean1-mean2
        if comparisontype=='RFXsubjects':
            df=len(array1)-1
            T,pval,z=sst.wilcoxonAES(array1,array2)
            string='%s: %s(%.2f)-%s(%.2f): z(%.0f)=%.3f, p=%.3f.' % (roi_summary.roi,m1,mean1,m2,mean2,df,z,pval)
            roi_summary.add_grnmodelcomparisonsRFX(compname,mean1,mean2,z,pval)
        elif comparisontype in ('stimbootstrap','condbootstrap'):
            bsMeanDiff, bsSEMDiff, upperbound, lowerbound, nullrejected, pval=comparemodelRDMfits_bootstraptest(m1, m2, bsmodelcorrs, alpha=0.05)
            string = "%s-- %s(%s)-%s(%s): observedMeandiff=%.3f, BSMdiff=%.3f, BSSEMdiff=%.3f, p=%.3f" %(roi_summary.roi,m1,mean1,m2,mean2,meandiff,bsMeanDiff, bsSEMDiff, pval)
            roi_summary.add_grnmodelcomparisons(compname, mean1,mean2, bsMeanDiff, pval, [lowerbound, upperbound])
        if printit:
            print string
    return roi_summary

def plotRFXresults(roi,subjs, models, modelsummaries,errorbars='withinsubj', plotit=True):
    '''plot results of RFX on single subject RDMs'''
    subjs=[s for s in subjs if roi in s.rois.keys()]
    plotmeans=[]
    plotsems=[]
    plotwssems=[]
    subjmeans=[]
    # for each subject (for this roi), get subject's mean tau across models
    for subj in range(len(subjs)):
        subjms=[modelsummaries[modeln][subj] for modeln in models]
        subjmeans.append(np.mean(subjms))
    normalizedmeans=[]
    for subj in range(len(subjs)):
        normalizedsubj=[modelsummaries[modeln][subj]-subjmeans[subj] for modeln in models]
        normalizedmeans.append(normalizedsubj)
    for modeln,m in enumerate(models):
        plotmeans.append(np.mean(modelsummaries[m]))
        plotsems.append(np.std(modelsummaries[m], ddof=1)/np.sqrt(len(modelsummaries[m])))
        normalized=[subjrow[modeln] for subjrow in normalizedmeans]
        plotwssems.append(np.std(normalized, ddof=1)/np.sqrt(len(normalized)))
    if errorbars=='withinsubj':
        if plotit:
            viz.simplebar(plotmeans, yerr=plotwssems, xticklabels=models, title='%s model-neural correlations /n (avg of single subj corrs)' %(roi), ylabel='kendall tau-a \n(SEM within subjs)')
    else:
        if plotit:
            viz.simplebar(plotmeans, yerr=plotsems, xticklabels=models, title='%s model-neural correlations /n (avg of single subj corrs)' %(roi), ylabel='kendall tau-a \n(SEM across subjs)')
    return plotsems, plotwssems