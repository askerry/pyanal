from mvpa2.measures import rsa
from mvpa2.mappers.fx import mean_group_sample
import mypymvpa.utilities.visualization as viz
import mypymvpa.utilities.stats as mus
import scipy.stats as sst
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt
from mypymvpa.utilities.hardcodedexpparams import FGEcondmapping
from copy import deepcopy
from mypymvpa.analysisobjs.dataset_funcs import preprocess, prepconfigV2
import mypymvpa.utilities.misc as mum
import itertools
import warnings
import seaborn as sns

global abb
abb = FGEcondmapping


class RSAresult():
    '''defines single neural RDM. methods for comparing to models and saving result output'''

    def __init__(self, neuralRDM, roi, subjid, ftcRDM=None, corrtype='pearsonr', symmetrical='True', neuraltype='rawsim'):
        self.rdm = neuralRDM
        self.fulltimecourserdm = ftcRDM
        self.roi = roi
        self.corrtype=corrtype
        self.neuraltype=neuraltype
        self.symmetrical=symmetrical
        self.subjid = subjid
        self.modelcorrs = {}

    def comparemodels(self, modelRDMs, whichmodels='all', fullorcrossfoldsRDMs='crossfolds'):
        if whichmodels=='all':
            models = modelRDMs.keys()
        else:
            models=[model+'_'+fullorcrossfoldsRDMs for model in whichmodels]
        for m in models:
            mrdm = modelRDMs[m]
            if self.corrtype=='kendallstau':
                corr, p = mus.kendallstau(self.rdm, mrdm, symmetrical=self.symmetrical)
            elif self.corrtype=='pearsonr':
                corr, p, throwaraylength=mus.pearsonr(self.rdm, mrdm, symmetrical=self.symmetrical)
            elif self.corrtype=='spearman':
                pass
            self.modelcorrs[m] = corr

    def save(self, subjroot, subdir, filename):
        filename = subjroot + subdir + filename
        if not os.path.exists(subjroot):
            os.mkdir(subjroot)
        if not os.path.exists(subjroot + subdir):
            os.mkdir(subjroot + subdir)
        with open(filename, 'wb') as output:
            pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)


class ROIsummary():
    '''defines summary of results for single ROI'''

    def __init__(self, roi, modelRDMs, fullorcrossfoldsRDMs='crossfolds', corrtype='pearsonr'):
        self.roi = roi
        self.grn = None
        self.ftcgrn = None
        self.modelRDMs = [m+'_'+fullorcrossfoldsRDMs for m in modelRDMs]
        self.grn2models = {
        modelname: {'corr': None, 'permutation_pval': None, 'bootstrap_CI': None, 'bootstrap_SEM': None} for modelname in
        modelRDMs}
        self.corrtype=corrtype
        self.grnmodelcomparisons = {}
        self.grn2modelsRFX = {modelname: {'corr': None, 'RFX_pval': None, 'RFX_SEM': None, 'RFX_WSSEM': None} for
                              modelname in modelRDMs}
        self.grnmodelcomparisonsRFX = {}

    def add_grn2models(self, modelname, corr, permutation_pval=None, bootstrap_CI=None, bootstrap_SEM=None):
        self.grn2models[modelname] = {'corr': corr, 'permutation_pval': permutation_pval, 'bootstrap_CI': bootstrap_CI,
                                      'bootstrap_SEM': bootstrap_SEM}

    def add_grnmodelcomparisons(self, comp, corr1, corr2, bootstrap_diff, bootstrap_pval, bootstrap_CI):
        self.grnmodelcomparisons[comp] = {'corr1': corr1, 'corr2': corr2, 'bootstrap_pval': bootstrap_pval,
                                          'bootstrap_CI': bootstrap_CI}

    def add_grn2modelsRFX(self, modelname, corr, stat, RFX_pval, df, RFX_SEM, RFX_WSSEM):
        self.grn2modelsRFX[modelname] = {'corr': corr, 'stat': stat, 'RFX_pval': RFX_pval, 'df': df, 'RFX_SEM': RFX_SEM,
                                         'RFX_WSSEM': RFX_WSSEM}

    def add_grnmodelcomparisonsRFX(self, comp, corr1, corr2, stat, RFX_pval, df):
        self.grnmodelcomparisonsRFX[comp] = {'corr1': corr1, 'corr2': corr2, 'stat': stat, 'RFX_pval': RFX_pval, 'df': df}

    def save(self, analdir, filename):
        filename = analdir + filename
        if not os.path.exists(analdir):
            os.mkdir(analdir)
        with open(filename, 'wb') as output:
            pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)

    def summarize_grn2models(self, models2show='all', modelcolors=None, ylim=None):
        if models2show=='all':
            models=self.modelRDMs
        else:
            models = [m for m in models2show if m in self.modelRDMs]
        if modelcolors:
            colors = [modelcolors[m] for m in models2show]
        else:
            colors=None
        modelcorrs = [self.grn2models[m]['corr'] for m in models]
        pvals = [self.grn2models[m]['permutation_pval'] for m in models]
        sems = [self.grn2models[m]['bootstrap_SEM'] for m in models]
        viz.simplebar(modelcorrs, yerr=sems, title='%s-- %s of group data' % (self.roi, self.corrtype), xlabel='models',
                      xticklabels=models, ylabel='%s\n(SEM from bootstrap test)' %(self.corrtype), colors=colors, ylim=ylim)
        strings=["%s: corr(SEM)=%.2f(%.2f), p(permutation test)=%.3f" %(m, modelcorrs[mn],sems[mn],pvals[mn]) for mn,m in enumerate(models)]
        for string in strings:
            print string

    def summarize_grn2modelsRFX(self, models2show='all',errorbars='ws',modelcolors=None, ylim=None, noiseceiling=None):
        if models2show=='all':
            models=self.modelRDMs
        else:
            models = [m for m in models2show if m in self.modelRDMs]
        if modelcolors:
            colors = [modelcolors[m] for m in models2show]
        else:
            colors=None
        modelcorrs = [self.grn2modelsRFX[m]['corr'] for m in models]
        pvals = [self.grn2modelsRFX[m]['RFX_pval'] for m in models]
        if errorbars == 'ws':
            sems = [self.grn2modelsRFX[m]['RFX_WSSEM'] for m in models]
            eblabel='+/- 1 SEM (within subjects)'
        else:
            sems = [self.grn2modelsRFX[m]['RFX_SEM'] for m in models]
            eblabel='+/- 1 SEM (between subjects)'
        ax=viz.simplebar(modelcorrs, yerr=sems, title='%s-- avg %s of ind data' % (self.roi, self.corrtype), xlabel='models',
                      xticklabels=models, ylabel='%s\n(%s)' %(self.corrtype, eblabel), colors=colors, ylim=ylim, show=False)
        if noiseceiling:
            ax.axhspan(noiseceiling[0]-.01, noiseceiling[1], facecolor='#8888AA', alpha=0.15)
            plt.show()
    def summarize_grnmodelcomparisonsRFX(self, models2show='all'):
        if models2show=='all':
            comparisons=self.grnmodelcomparisons.keys()
        else:
            comparisons = [c for c in self.grnmodelcomparisons.keys() if sum([int(m in c) for m in models2show])>=2]
        for comp in comparisons:
            r = self.grnmodelcomparisonsRFX[comp]
            if r['RFX_pval'] < .05:
                tag = '***'
            else:
                tag = ''
            resultsstring = '%s: M1=%.2f, M2=%.2f, z(%s)=%.2f, p=%.3f. %s' % (
            comp, r['corr1'], r['corr2'], r['df'], r['stat'], r['RFX_pval'], tag)
            print resultsstring

    def summarize_grnmodelcomparisons(self, models2show='all'):
        if models2show=='all':
            comparisons=self.grnmodelcomparisons.keys()
        else:
            comparisons = [c for c in self.grnmodelcomparisons.keys() if sum([int(m in c) for m in models2show])>=2]
        for comp in comparisons:
            r = self.grnmodelcomparisons[comp]
            if r['bootstrap_pval'] < .05:
                tag = '***'
            else:
                tag = ''
            resultsstring = '%s: M1=%.2f, M2=%.2f, p=%.3f. %s' % (comp, r['corr1'], r['corr2'], r['bootstrap_pval'], tag)
            print resultsstring

class TimecourseRSA():
    def __init__(self, windowdur, tcrange, roi=None, disc=None, models=[], analdir=None):
        self.windowdur=windowdur
        self.roi=roi
        self.disc=disc
        self.steps=range(tcrange[0], tcrange[1])
        self.models=models
        self.analdir=analdir
        self.modeltcs_err={model:[0 for tp in self.steps] for model in models}
        self.modeltcs_corr={model:[0 for tp in self.steps] for model in models}
        self.modeltcs_ncl={model:[0 for tp in self.steps] for model in models}
        self.modeltcs_ncu={model:[0 for tp in self.steps] for model in models}
        colors=sns.color_palette('husl',len(self.models))
        self.colordict={model:colors[modeln] for modeln,model in enumerate(self.models)}
    def updatetimecourse(self,model, tpindex, value, error, lower=0, upper=0):
        self.modeltcs_corr[model][tpindex]=value
        self.modeltcs_err[model][tpindex]=error
        self.modeltcs_ncl[model][tpindex]=lower
        self.modeltcs_ncl[model][tpindex]=upper
    def plottimecourse(self, model, ax=None, plotnc=False, ylim=[-.04,.1], color=None):
        y=self.modeltcs_corr[model]
        l=self.modeltcs_ncl[model]
        u=self.modeltcs_ncu[model]
        err=self.modeltcs_err[model]
        if any(np.array(y)!=0):
            if color:
                ax.errorbar(range(len(y)),y, yerr=err, label=model, color=color)
            else:
                ax.errorbar(range(len(y)),y, yerr=err, label=model, color=self.colordict[model])
            if plotnc:
                ax.errorbar(range(len(y)),l, label='lower bound')
                ax.errorbar(range(len(y)),u, label='upper bound')
            ax.set_xticks([el for el in range(len(self.steps))])
            ax.set_xticklabels([str(el) for el in self.steps])
            ax.set_xlabel('RSA timecourse (TRs)')
            ax.set_ylabel('neural-model correlation')
            ax.set_title('neural-model correlations over time: %s' %(self.roi))
            ax.legend(loc=[1.05,.3])
            if plotnc==False:
                ax.set_ylim(ylim)
    def save(self):
        savedir=self.analdir+'timecourse/'
        if not os.path.exists(savedir): #if the path doesn't exist, make the folder
            os.mkdir(savedir)
        filename=savedir+'timecourse_%s_%s.pkl' %(self.roi, self.disc)
        mum.picklethisobject(filename, self)

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

def relatesingle2group(grouprdms, indRDM, configspec):
    corrtype=configspec['corrtype']
    avgRDM=np.mean(grouprdms,axis=0)
    if corrtype=='kendallstau':
        corr, p = mus.kendallstau(avgRDM, indRDM, symmetrical=configspec['symmetrical'])
    elif corrtype=='pearsonr':
        corr, p, throwaraylength=mus.pearsonr(avgRDM, indRDM,  symmetrical=configspec['symmetrical'])
    elif corrtype=='spearman':
        pass
    return corr

def computenoiseceiling(grouprdms, roilist, configspec):
    noiseceilings={}
    for roi in roilist:
        rdms=np.array(grouprdms[roi])
        uppercorrs=[]
        lowercorrs=[]
        for ind in rdms:
            uppercorrs.append(relatesingle2group(rdms, ind, configspec))
        for indn, ind in enumerate(rdms):
            nminus1rdms=[rdms[i] for i in range(len(rdms)) if i != indn]
            lowercorrs.append(relatesingle2group(nminus1rdms, ind, configspec))
        upper=np.mean(uppercorrs)
        lower=np.mean(lowercorrs)
        noiseceilings[roi]=(lower,upper)
    return noiseceilings


def singlesubjanalysis(e, disc, roilist, subjects, runthemnowlist, runindsubjects, configspec, modelRDMs, conditions,
                       savetag, whichmodels='all', svmerrors=None, timecourse=False):
    sel = e.selectors.keys()[0]
    grouprdms = {}
    groupftcrdms = {}
    for roi in roilist:
        print roi
        grouprdms[roi], groupftcrdms[roi] = [], []
        if runindsubjects:
            print 'running individual subjects...'
        for subjectn, subject in enumerate(subjects):
            if subject.subjid in runthemnowlist:
                if roi in subject.rois.keys():
                    subjfilename = disc + '_' + sel + '_' + roi + 'RSA.pkl'
                    if configspec['neuraltype']=='rawsim':
                        cfg = prepconfigV2(e, detrend=configspec['detrend'], zscore=configspec['zscore'],
                                           averaging=configspec['averaging'], removearts=configspec['removearts'],
                                           hpfilter=configspec['hpfilter'], clfname=configspec['clfname'],
                                           featureselect=configspec['featureselect'])
                        if runindsubjects:
                            dataset = subject.makedataset(disc, sel, roi)
                            dataset.cfg, dataset.a['cfg'] = cfg, cfg
                        subdir = 'subjresults_%s_%s_%s%s/' % (cfg.clfstring, e.task, e.boldsorbetas, savetag)
                        if runindsubjects:
                            preppeddata = preprocess(dataset)
                            singletimecoursedata = prepforsinglerdm(preppeddata)
                            preppeddata = prepforrsa(preppeddata)
                            singleRDM = singletimecoursesimilarities(singletimecoursedata, conditions, subject.subjid, roi,
                                                                     distance=configspec['similarity'],
                                                                     transformed=configspec['transformed'],
                                                                     plotit=configspec['plotindsubjs'])
                            rdm = crossrunsimilarities(preppeddata, conditions, subject.subjid, roi,
                                                       distance=configspec['similarity'],
                                                       transformed=configspec['transformed'], plotit=configspec['plotindsubjs'])
                        else:
                            result = mum.loadpickledobject(subject.subjanalysisdir + subdir + subjfilename)
                            rdm, singleRDM = result.rdm, result.fulltimecourserdm
                        rdmresult = RSAresult(rdm, roi, subject.subjid, ftcRDM=singleRDM, symmetrical=configspec['symmetrical'], corrtype=configspec['corrtype'], neuraltype=configspec['neuraltype'])
                    elif configspec['neuraltype']=='svmerrors':
                        errorfile=svmerrors.replace('<subject>', subject.subjid)
                        errorfile=errorfile.replace('<roi>',roi)
                        err=mum.loadpickledobject(errorfile)
                        subdir = 'subjresults_SVMerrors_%s_%s%s/' % (e.task, e.boldsorbetas, savetag)
                        rdm=-1*np.array(err.confusions['confmatrix'])
                        rdm=transformsimilarities(rdm, configspec['similarity'])
                        singleRDM=None
                        rdmresult = RSAresult(rdm, roi, subject.subjid, ftcRDM=None, symmetrical=configspec['symmetrical'], corrtype=configspec['corrtype'],neuraltype=configspec['neuraltype'])
                    rdmresult.comparemodels(modelRDMs, whichmodels='all', fullorcrossfoldsRDMs=configspec['fullorcrossfoldsRDMs'])
                    if timecourse:
                        subjfilename='%s_%s' %(e.trshift, subjfilename)
                        subdir=subdir+'timecourse/'
                        rdmresult.save(subject.subjanalysisdir, subdir, subjfilename)
                    else:
                        rdmresult.save(subject.subjanalysisdir, subdir, subjfilename)
                        print "saved to %s%s%s" % (subject.subjanalysisdir, subdir, subjfilename)
                    grouprdms[roi].append(rdm)
                    groupftcrdms[roi].append(singleRDM)
        print "finished %s, %s, %s" % (disc, sel, roi)
    return grouprdms, groupftcrdms, subdir


def euclideandistancematrix(matrix):
    '''take a matrix of items in rows and features in columns and return a similarity space of items'''
    dim = len(matrix)
    distancematrix= np.zeros((dim, dim))
    for vector1n, vector1 in enumerate(matrix):
        for vector2n, vector2 in enumerate(matrix):
            #ed=np.linalg.norm(np.array(vector1)-np.array(vector2))
            ed = similarity(vector1, vector2, metric='euclidean')
            distancematrix[vector1n][vector2n] = ed
    distancematrix=-1*distancematrix
    transformeddistancematrix=transformeuclideandistancematrix(distancematrix, dim)
    return distancematrix, transformeddistancematrix

def transformeuclideandistancematrix_OLD(distancematrix, dim):
    transformeddistancematrix = np.zeros((dim, dim))
    meandist, stddist = np.mean(distancematrix), np.std(distancematrix, ddof=1)
    for vector1n in range(dim):
        for vector2n in range(dim):
            ed = distancematrix[vector1n][vector2n]
            zscored_ed = (ed - meandist) / stddist
            transformeddistancematrix[vector1n][vector2n] = zscored_ed
    return transformeddistancematrix

def transformeuclideandistancematrix(distancematrix, dim):
    transformeddistancematrix = np.zeros((dim, dim))
    maxdist, mindist = np.max(distancematrix), np.min(distancematrix)
    for vector1n in range(dim):
        for vector2n in range(dim):
            ed = distancematrix[vector1n][vector2n]
            normalized = float(ed - mindist) / (maxdist-mindist)
            transformeddistancematrix[vector1n][vector2n] = normalized
    return transformeddistancematrix

def similarity(pattern1, pattern2, metric='pearsonr'):
    '''takes two arrays (e.g. neural patterns) and computes similarity between them'''
    if metric == 'pearsonr':
        sim, p = sst.pearsonr(pattern1, pattern2)
    if metric == 'euclidean':
        try:
            sqdiffs = [(p1 - pattern2[pn]) ** 2 for pn, p1 in enumerate(pattern1)]
            sim = sum(sqdiffs)
        except:
            #if patterns are single digit
            sim = ((pattern1 - pattern2) ** 2)
    return sim # higher is more similar

def makesimmatrixcrossruns(conditions, item2emomapping, inputmatrix, itemlabels, distance, itemsoremos):
    idx1=[]
    idx2=[]
    for c in conditions:
        items=[i[0] for i in item2emomapping.items() if abb[i[1]]==c]
        random.shuffle(items)
        idx1.extend(items[0:5])
        idx2.extend(items[5:10])
    if itemsoremos=='items':
        elements=itemlabels
    else:
        elements=conditions
    rdm = np.zeros([len(elements), len(elements)])
    for cn, c in enumerate(elements):
        if itemsoremos=='items':
            pattern1 = [sample for samplen, sample in enumerate(inputmatrix) if
                        itemlabels[samplen] in idx1 and itemlabels[samplen] == c]
        else:
            pattern1 = [sample for samplen, sample in enumerate(inputmatrix) if
                        itemlabels[samplen] in idx1 and abb[item2emomapping[itemlabels[samplen]]] == c]
        for c2n, c2 in enumerate(elements):
            if itemsoremos=='items':
                pattern2 = [sample for samplen, sample in enumerate(inputmatrix) if
                        itemlabels[samplen] in idx2 and itemlabels[samplen] == c]
            else:
                pattern2 = [sample for samplen, sample in enumerate(inputmatrix) if
                        itemlabels[samplen] in idx2 and abb[item2emomapping[itemlabels[samplen]]] == c2]
            pattern1array, pattern2array = np.mean(pattern1,axis=0), np.mean(pattern2, axis=0)
            try:
                sim = similarity(pattern1array, pattern2array, metric=distance)
            except:
                sim = np.nan
                warnings.warn("You have nan values in your similarity matrix because some conditions were not found")
            rdm[cn][c2n] = sim
    if distance=='correlation':
        rdm=1-rdm
    if distance=='euclidean':
        rdm=-1*rdm
    rdm=transformsimilarities(rdm, distance)
    return rdm

def makemodelmatrices(conditions, rsamatrixfile, matrixkey, similarity='euclidean', itemsoremos='emos',
                      visualizeinput=False, itemindices=None, iterations=10, rankcolors=False):
    '''make RDMs for each of the models. option to also visualize initial input matrix.'''
    with open(rsamatrixfile, 'r') as inputfile:
        rsaobj = pickle.load(inputfile)
    inputmatrix, itemlabels, item2emomapping = rsaobj[matrixkey]['itemavgs'], rsaobj[matrixkey]['itemlabels'], rsaobj[
        'item2emomapping']
    #re order to align with conditions
    datamatrix, axis = [], []
    for c in conditions:
        condlines = [line for linen, line in enumerate(inputmatrix) if abb[item2emomapping[itemlabels[linen]]] == c]
        condlabels = [itemlabels[linen] for linen, line in enumerate(inputmatrix) if
                      abb[item2emomapping[itemlabels[linen]]] == c]
        datamatrix.extend(condlines)
        axis.extend(condlabels)
    if itemsoremos == 'ind':
        #get single rdm
        if visualizeinput:
            if type(datamatrix[0]) in (np.float64, float):
                viz.simplebar(datamatrix, xticklabels=conditions, xtickrotation=90, figsize=[6, 4], colors='orange',
                              xlabel=matrixkey + '_inputmatrix')
            else:
                viz.simplematrix(datamatrix, yticklabels=axis, xlabel='%s_inputmatrix' % (matrixkey))
        if similarity == 'correlation':
            RSAmat = 1 - np.corrcoef(datamatrix)
            plotmin, plotmax = -1, 1
        elif similarity == 'euclidean':
            RSAmat_untransformed, RSAmat = euclideandistancematrix(datamatrix)
            RSAmat=1-RSAmat
            #plotmin, plotmax = np.min(RSAmat), np.max(RSAmat)
            plotmin, plotmax = 0,1
        else:
            raise RuntimeError("%s is not a recognized similarity metric" % (similarity))
        #viz.simplematrix(RSAmat, minspec=plotmin, maxspec=plotmax, xticklabels=axis, yticklabels=axis,
        #                 colorspec='RdYlBu_r', xtickrotation=90, xlabel='%s_indRSA_full' % (matrixkey))
        #get cross run rdm
        rdms=[]
        for i in range(iterations):
            rdms.append(makesimmatrixcrossruns(conditions, item2emomapping, inputmatrix, itemlabels, similarity, itemsoremos))
        crossrun_rdm=np.mean(rdms, axis=0)
        crossrun_rdm = 1 - crossrun_rdm
        #plotmin, plotmax = np.min(crossrun_rdm), np.max(crossrun_rdm)
        if rankcolors:
            plotmin,plotmax=None,None
        else:
            plotmin, plotmax = 0,1
        viz.simplematrix(crossrun_rdm, minspec=plotmin, maxspec=plotmax, xticklabels=axis, yticklabels=axis,
                         colorspec='RdYlBu_r', xtickrotation=90, xlabel='%s_indRSA_crossrun' % (matrixkey), rankcolors=rankcolors)
    elif itemsoremos == 'emos':
        #make emo matrix
        emomatrix = []
        for emo in conditions:
            lines = [line for linen, line in enumerate(datamatrix) if abb[item2emomapping[axis[linen]]] == emo]
            lines = np.mean(lines, 0)
            try:
                lines = [el for el in lines]
            except:
                pass
            emomatrix.append(lines)
        # get single RDM
        if visualizeinput:
            if type(emomatrix[0]) in (np.float64, float):
                viz.simplebar(emomatrix, xticklabels=conditions, xtickrotation=90, figsize=[6, 4], colors='orange',
                              xlabel=matrixkey + '_inputmatrix')
            else:
                viz.simplematrix(emomatrix, yticklabels=conditions, xlabel='%s_inputmatrix' % (matrixkey))
        if similarity == 'correlation':
            RSAmat = 1 - np.corrcoef(emomatrix)
        elif similarity == 'euclidean':
            RSAmat_untransformed, RSAmat = euclideandistancematrix(emomatrix)
            RSAmat=1-RSAmat
        else:
            raise RuntimeError("%s is not a recognized similarity metric" % (similarity))
        #if rankcolors:
        #    plotmin,plotmax=None,None
        #else:
        #    plotmin, plotmax = 0,1
        #viz.simplematrix(RSAmat, minspec=plotmin, maxspec=plotmax, xticklabels=conditions, yticklabels=conditions,
        #                 colorspec='RdYlBu_r', xtickrotation=90, xlabel='%s_emoRSA_full' % (matrixkey))
        #get cross run rdm
        rdms=[]
        for i in range(iterations):
            rdm=(makesimmatrixcrossruns(conditions, item2emomapping, inputmatrix, itemlabels, similarity, itemsoremos))
            rdms.append(rdm)
        crossrun_rdm=np.mean(rdms, axis=0)
        crossrun_rdm = 1 - crossrun_rdm
        #plotmin, plotmax = np.min(crossrun_rdm), np.max(crossrun_rdm)
        if rankcolors:
            plotmin,plotmax=None,None
        else:
            plotmin, plotmax = 0,1
        viz.simplematrix(crossrun_rdm, minspec=plotmin, maxspec=plotmax, xticklabels=conditions, yticklabels=conditions,
                         colorspec='RdYlBu_r', xtickrotation=90, xlabel='%s_emoRSA_crossrun' % (matrixkey), rankcolors=rankcolors)
    return RSAmat, crossrun_rdm, item2emomapping

def gettpresults(roi_summary, model, error='ws'):
    result=roi_summary.grn2modelsRFX[model]
    tpcorr=result['corr']
    if error=='ws':
        tperr=result['RFX_WSSEM']
    elif error=='bs':
        tperr=result['RFX_SEM']
    return tpcorr, tperr

def addconfmatrix(filename, item2emomapping, conditions, rankcolors=False, similarity='euclidean'):
    with open(filename, 'r') as inputfile:
        errors = pickle.load(inputfile)
    emos=[item2emomapping[item] for item in errors['itemlabels']]
    confmat=[list(line) for line in errors['confmat']]
    sortederror=[]
    emoorders=[abb[emo] for emo in errors['emos']]
    for line in confmat:
        sortederror.append([line[emoorders.index(emo)] for emo in conditions])
    finalmat=[]
    for emo in conditions:
        vector=[line for linen,line in enumerate(sortederror) if abb[emos[linen]]==emo]
        finalmat.append(np.mean(vector, axis=0))
    finalmat=1-np.array(finalmat)
    if rankcolors:
        plotmin,plotmax=None,None
    else:
        plotmin, plotmax = 0,1
    finalmat = transformsimilarities(finalmat, similarity)
    viz.simplematrix(finalmat, minspec=plotmin, maxspec=plotmax, xticklabels=conditions, yticklabels=conditions,
                         colorspec='RdYlBu_r', xtickrotation=90, ylabel='intended emotion', xlabel='raw error', rankcolors=rankcolors)
    return finalmat

def addcosinematrix(filename, item2emomapping, conditions, iterations=2, rankcolors=False, similarity='euclidean'):
    with open(filename, 'r') as inputfile:
        cosine = pickle.load(inputfile)
    itememos=[item2emomapping[item] for item in cosine['itemlabels']]
    mat=[list(line) for line in cosine['matrix']]
    matrices=[]
    for i in range(iterations):
        idx1=[]
        idx2=[]
        for c in conditions:
            items=[inum for inum,i in enumerate(itememos) if abb[i]==c]
            random.shuffle(items)
            idx1.extend(items[0:5])
            idx2.extend(items[5:10])
        sortedmat=[]
        newemos=[]
        for linen,line in enumerate(mat):
            if linen in idx1:
                newline=[]
                for emo in conditions:
                    relevantitems=[eln for eln,el in enumerate(itememos) if abb[el]==emo and eln in idx2]
                    emoavg=np.nanmean([line[x] for x in relevantitems])
                    newline.append(emoavg)
                sortedmat.append(newline)
                newemos.append(abb[itememos[linen]])
        finalmat=[]
        for emo in conditions:
            submatrix=[line for linen,line in enumerate(sortedmat) if newemos[linen]==emo]
            finalmat.append(np.mean(submatrix,axis=0))
        #normedmat=transformeuclideandistancematrix(finalmat, len(finalmat)) #transform as though it's euclidean
        matrices.append(np.array(finalmat))
    simmat=np.mean(matrices,axis=0)
    if rankcolors:
        plotmin,plotmax=None,None
    else:
        plotmin, plotmax = 0,1
    simmat=-1*simmat
    simmat = transformsimilarities(simmat, similarity)
    viz.simplematrix(simmat, minspec=plotmin, maxspec=plotmax, xticklabels=conditions, yticklabels=conditions,
                             colorspec='RdYlBu_r', xtickrotation=90, xlabel='cosinesimilarity', rankcolors=rankcolors)
    return simmat

def addsvmerrors(filename, item2emomapping, conditions, rankcolors=False):
    with open(filename, 'r') as inputfile:
        errors = pickle.load(inputfile)
    confmat=[list(line) for line in errors['confmat']]
    sortederror=[]
    emoorders=[abb[emo] for emo in errors['emolabels']]
    for line in confmat:
        sortederror.append([line[emoorders.index(emo)] for emo in conditions])
    finalmat=[]
    for emo in conditions:
        vector=[line for linen,line in enumerate(sortederror) if abb[emos[linen]]==emo]
        finalmat.append(np.mean(vector, axis=0))
    finalmat=1-np.array(finalmat)
    if rankcolors:
        plotmin,plotmax=None,None
    else:
        plotmin, plotmax = 0,1
    viz.simplematrix(finalmat, minspec=plotmin, maxspec=plotmax, xticklabels=conditions, yticklabels=conditions,
                         colorspec='RdYlBu_r', xtickrotation=90, ylabel='intended emotion', xlabel='raw error', rankcolors=rankcolors)
    return -1*finalmat


def singletimecoursesimilarities(ds, conditions, subjectid, roi, distance='pearsonr', transformed=False, plotit=True, rankcolors=False):
    '''computes neural RDM within a single timecourse (diagonal necessarily 0)'''
    rdm = np.zeros([len(conditions), len(conditions)])
    for cn, c in enumerate(conditions):
        pattern1 = [sample for samplen, sample in enumerate(ds.samples) if ds.sa.targets[samplen] == c][0]
        for c2n, c2 in enumerate(conditions):
            pattern2 = [sample for samplen, sample in enumerate(ds.samples) if ds.sa.targets[samplen] == c2][0]
            sim = similarity(pattern1, pattern2, metric=distance)
            rdm[cn][c2n] = sim
    if distance == 'pearsonr':
        if transformed:
            rdm = transformsimilarities(rdm, distance)
            clim = [np.min(rdm), np.max(rdm)]
        else:
            clim = [-1, 1]
    elif distance == 'euclidean':
        if transformed:
            rdm = transformsimilarities(rdm, distance)
            clim = [0,1]
        else:
            clim = [np.min(rdm), np.max(rdm)]
    if plotit:
        if rankcolors:
            clim = [None,None]
        viz.plot_simmtx(rdm, conditions, '%s: %s single timecourse pattern distances (%s)' % (subjectid, roi, distance),
                        clim=clim, axislabels=['conditions', 'conditions'], rankcolors=rankcolors)
    return rdm


def crossrunsimilarities(ds, conditions, subjectid, roi, distance='pearsonr', transformed=False, plotit=True, rankcolors=False):
    '''compute neural RDM across runs (diagonal is interpretable)'''
    folds = list(set(ds.sa.chunks))
    if len(folds) > 2:
        raise RuntimeError('This code assumes only two folds, but you appear to have more than two.')
    rdm = np.zeros([len(conditions), len(conditions)])
    for cn, c in enumerate(conditions):
        pattern1 = [sample for samplen, sample in enumerate(ds.samples) if
                    ds.sa.chunks[samplen] == folds[0] and ds.sa.targets[samplen] == c]
        for c2n, c2 in enumerate(conditions):
            pattern2 = [sample for samplen, sample in enumerate(ds.samples) if
                        ds.sa.chunks[samplen] == folds[1] and ds.sa.targets[samplen] == c2]
            if pattern1 != [] and pattern2 != []:
                pattern1array, pattern2array = pattern1[0], pattern2[0]
                try:
                    sim = similarity(pattern1array, pattern2array, metric=distance)
                except:
                    sim = np.nan
                    warnings.warn("Unexpectedly failed to compute distance for the following 2 patterns")
            else:
                sim = np.nan
                warnings.warn("You have nan values in your similarity matrix because some conditions were not found")
            rdm[cn][c2n] = sim
    if distance == 'pearsonr':
        if transformed:
            rdm = transformsimilarities(rdm, distance)
            clim = [np.min(rdm), np.max(rdm)]
        else:
            clim = [-1, 1]
    elif distance == 'euclidean':
        if transformed:
            rdm = transformsimilarities(rdm, distance)
            clim = [0,1]
        else:
            clim = [np.min(rdm), np.max(rdm)]
    if plotit:
        if rankcolors:
            clim = [None,None]
        viz.plot_simmtx(rdm, conditions, '%s: %s crossrun pattern distances (%s)' % (subjectid, roi, distance),
                        clim=clim, axislabels=folds, rankcolors=rankcolors)
    return rdm


def relateRDMsgrn(roi_summary, modelRDMs, configspec, alphas=[.05, .01, .001], printit=True, plotpermutationfig=True,
                  whichmodels='all'):
    '''computes relationship between single group neural RDM and each model. significance assessed using bootstrap and condition-permuting.'''
    corrtype=configspec['corrtype']
    num_samples=configspec['num_samples']
    if whichmodels=='all':
        models = modelRDMs.keys()
    else:
        models=[model+'_'+configspec['fullorcrossfoldsRDMs'] for model in whichmodels]
    corrs, pvals = [], []
    neuralRDM = roi_summary.grn
    for m in models:
        rdm = modelRDMs[m]
        if corrtype=='kendallstau':
            corr, p = mus.kendallstau(neuralRDM, rdm, symmetrical=configspec['symmetrical'])
        elif corrtype=='pearsonr':
            corr, p, throwaraylength=mus.pearsonr(neuralRDM, rdm,  symmetrical=configspec['symmetrical'])
        elif corrtype=='spearman':
            pass
        permMean, upperbound, lowerbound, nullrejected, realpval, exactp = singleRDMrelation_permutationtest(neuralRDM,
                                                                                                             rdm, configspec, corr,
                                                                                                             alphas,
                                                                                                             plotit=plotpermutationfig)
        string = "%s-%s: %s=%.3f, p=%.3f (p%s) (p value from randomized permutation test)" % (roi_summary.roi, m, corrtype, corr, exactp, realpval)
        if printit:
            print string
        corrs.append(corr)
        pvals.append(exactp)
    bm = np.where(corrs == np.max(corrs))[0]
    if len(bm) > 1:
        if printit:
            warnings.warn("multiple best models found. taking just the first")
    bm = bm[0]
    bestmodel = models[bm]
    if printit:
        print "%s, bestmodel: %s" % (roi_summary.roi, bestmodel)
    roi_summary.bestmodel = bestmodel
    return corrs, pvals, models, bestmodel, roi_summary


def sampledist(samplestatistics, num_samples, alpha, plotit=True, observed=None, ax=None):
    '''takes set of sample statistics and returns CI and SEM'''
    lowerbound = np.sort(samplestatistics)[int((alpha / 2.0) * num_samples)]
    upperbound = np.sort(samplestatistics)[int((1 - alpha / 2.0) * num_samples)]
    if plotit:
        try:
            ax.hist(samplestatistics, 20, color='blue')
        except:
            ax.hist(samplestatistics, len(samplestatistics), color='blue')
        ylim = plt.ylim()
        #print ylim
        ax.plot([lowerbound, lowerbound], ylim, 'k-', lw=2, alpha=.1)
        ax.plot([upperbound, upperbound], ylim, 'k-', lw=2, alpha=.1)
        if observed:
            xlim = list(plt.xlim())
            if xlim[0] > observed:
                xlim[0] = observed - .1
            if xlim[1] < observed:
                xlim[1] = observed + .1
            plt.xlim(xlim)
            ax.plot([observed, observed], ylim, lw=2, color='red')
    SEM = np.std(samplestatistics, ddof=1)
    return lowerbound, upperbound, SEM, ax


def testdist(observedmean, samplemeans, num_samples, alpha, tail='both', ax=None, plotit=True):
    '''takes set of samples, an observation, and an alpha, and returns whether null hypothesis is rejected at that alpha, as well as CI and SEM of distribution'''
    lowerbound, upperbound, SEM, ax = sampledist(samplemeans, num_samples, alpha, plotit=plotit, observed=observedmean,
                                                 ax=ax)
    unders = np.array(samplemeans) <= observedmean
    overs = np.array(samplemeans) >= observedmean
    exactps = [float(np.sum(unders)) / len(samplemeans), float(np.sum(overs)) / len(samplemeans)]
    exactp = np.min(exactps)
    pdict = {0: '>%s' % (alpha), 1: '<%s' % (alpha)}
    if tail == 'both':
        h = observedmean < lowerbound or observedmean > upperbound
    elif tail == 'right':
        h = observedmean > upperbound
    elif tail == 'left':
        h = observedmean < lowerbound
    pstr = pdict[h]
    return h, pstr, exactp, lowerbound, upperbound, SEM, ax


def singleRDMrelation_permutationtest(neuralRDM, modelRDM, configspec, observedcorr, alphas, plotit=False):
    corrtype=configspec['corrtype']
    num_samples=configspec['num_samples']
    neuralRDM = np.array(neuralRDM)
    modelRDM = np.array(modelRDM)
    rdmsize = np.shape(neuralRDM)
    if rdmsize != np.shape(modelRDM):
        raise RuntimeError('Error: your RDMs differ in size')
    samplecorrs = []
    for b in range(num_samples):
        colidx = np.random.permutation(rdmsize[0])
        rowidx = np.random.permutation(rdmsize[0])
        colsshuffled = neuralRDM[:, colidx]
        shuffledrdm = colsshuffled[rowidx, :]
        if corrtype=='kendallstau':
            rdmcorr, throwawayp = mus.kendallstau(shuffledrdm, modelRDM, symmetrical=configspec['symmetrical'])
        elif corrtype=='pearsonr':
            rdmcorr, throwawayp, throwaraylength=mus.pearsonr(shuffledrdm, modelRDM,  symmetrical=configspec['symmetrical'])
        elif corrtype=='spearman':
            pass
        samplecorrs.append(rdmcorr)
    permMean = np.mean(samplecorrs)
    #print "mean of permuted null distribution=%.8f" %(permMean)
    nullrejections = []
    pvalstr = '>%s' % (alphas[0])
    if plotit:
        f, ax = plt.subplots(figsize=[4, 2])
    for alpha in alphas:
        if plotit:
            nullrejected, pval, exactp, lowerbound, upperbound, throwawaypermSEM, ax = testdist(observedcorr,
                                                                                                samplecorrs,
                                                                                                num_samples, alpha,
                                                                                                ax=ax, plotit=plotit)
        else:
            nullrejected, pval, exactp, lowerbound, upperbound, throwawaypermSEM, ax = testdist(observedcorr,
                                                                                                samplecorrs,
                                                                                                num_samples, alpha,
                                                                                                plotit=plotit)
        nullrejections.append(nullrejected)
        if nullrejected:
            pvalstr = pval
    if plotit:
        plt.show()
    nullrejected = any(nullrejections)
    return permMean, upperbound, lowerbound, nullrejected, pvalstr, exactp


def bootstrapinner(e, subject, disc, sel, roi, configspec, conditions, modelRDMs):
    '''this resamples at the level of individual stimuli'''
    raise RuntimeError(
        'This feature (bootstrapping at the level of individual stimuli) is not yet implemented correctly')
    #the following may not be implemented correctly
    # cfg=prepconfigV2(e,detrend=configspec['detrend'],zscore=configspec['zscore'], averaging=configspec['averaging'],removearts=configspec['removearts'], hpfilter=configspec['hpfilter'], clfname=configspec['clfname'], featureselect=configspec['featureselect'])
    # dataset=subject.makedataset(disc, sel, roi)
    # dataset.cfg = cfg
    # dataset.a['cfg'] = cfg
    # preppeddata=preprocess(dataset)
    # subjrdms=[]
    # print "working on bootstrap (%s samples, resampling stimuli)" %(configspec['num_samples'])
    # for b in range(configspec['num_samples']):
    #     idx=[eln for eln,el in enumerate(preppeddata.sa.targets) if el in conditions]
    #     rsampleidx=np.random.choice(idx, size=len(idx), replace=True)
    #     bssampledata=preppeddata[rsampleidx]
    #     bssampledata=prepforrsa(bssampledata)
    #     neuralrdm=crossrunsimilarities(bssampledata, conditions, subject.subjid, roi, distance='euclidean', transformed=True, plotit=False)
    #     subjrdms.append(neuralrdm)
    return subjrdms


def bootstrapfromconditions(disc, modelRDMs, grouprdmmean, configspec, printit=True, whichmodels='all', fullorcrossfoldsRDMs='crossfolds'):
    corrtype=configspec['corrtype']
    num_samples=configspec['num_samples']
    if printit:
        print 'performing bootstrapping for errorbars (conditions)'
    shape = np.shape(grouprdmmean)
    idx = range(shape[0])
    if whichmodels=='all':
        models = modelRDMs.keys()
    else:
        models=[model+'_'+fullorcrossfoldsRDMs for model in whichmodels]
    bsmodelcorrs = {m: [] for m in models}
    for m in models:
        for b in range(num_samples):
            resampledgrouprdm = np.empty(shape)
            resampledgrouprdm[:] = np.nan
            rsampleidx = np.random.choice(idx, size=len(idx), replace=True)
            for x in rsampleidx:
                for y in rsampleidx:
                    resampledgrouprdm[x, y] = grouprdmmean[x, y]
            if corrtype=='kendallstau':
                corr, throwawayp = mus.kendallstau(resampledgrouprdm, modelRDMs[m], symmetrical=configspec['symmetrical'])
            elif corrtype=='pearsonr':
                corr, throwawayp, throwaraylength=mus.pearsonr(resampledgrouprdm, modelRDMs[m], symmetrical=configspec['symmetrical'])
            elif corrtype=='spearman':
                pass
            bsmodelcorrs[m].append(corr)
    return bsmodelcorrs


def bootstrapfromstimuli(e, disc, roi, subjects, configspec, modelRDMs, conditions, whichmodels='all', fullorcrossfoldsRDMs='crossfolds'):
    '''bootstraps at the level of stimuli in individual subjects'''
    print 'performing bootstrapping for errorbars (stimuli)'
    warnings.warn(
        "This probably isn't implemented the way you want it. don't use without reading and thinking about it.")
    print "starting bootstrap for %s, %s" % (disc, roi)
    corrtype=configspec['corrtype']
    num_samples=configspec['num_samples']
    sel = e.selectors.keys()[0]
    neuralmtxs = []
    grouprdms = []
    for subjectn, subject in enumerate(subjects):
        print "working on subject " + subject.subjid
        if roi in subject.rois.keys():
            subjrdms = bootstrapinner(e, subject, disc, sel, roi, configspec, conditions, num_samples, modelRDMs)
            grouprdms.append(subjrdms)
    grouprdms = np.mean(grouprdms, 0)
    if whichmodels=='all':
        models = modelRDMs.keys()
    else:
        models=[model+'_'+fullorcrossfoldsRDMs for model in whichmodels]
    bsmodelcorrs = {m: [] for m in models}
    #optimize this so that we don't reloop through everything this way
    for b in range(num_samples):
        for m in models:
            if corrtype=='kendallstau':
                corr, throwawayp = mus.kendallstau(grouprdms[b], modelRDMs[m], symmetrical=configspec['symmetrical'])
            elif corrtype=='pearsonr':
                corr, throwawayp, throwaraylength=mus.pearsonr(grouprdms[b], modelRDMs[m],  symmetrical=configspec['symmetrical'])
            elif corrtype=='spearman':
                pass
            bsmodelcorrs[m].append(corr)
    print "finished bootstrapping %s, %s, %s" % (disc, sel, roi)
    return bsmodelcorrs


def comparemodelRDMfits_bootstraptest(model1, model2, bsmodelcorrs, alpha=0.05):
    corr1, corr2 = np.array(bsmodelcorrs[model1]), np.array(bsmodelcorrs[model2])
    diffs = corr1 - corr2
    bsMeanDiff = np.mean(diffs)
    bsSEMDiff = np.std(diffs, ddof=1)
    lowerbound, upperbound, SEM, throwawayax = sampledist(diffs, len(diffs), alpha, plotit=False)
    unders = np.array(diffs) < 0
    overs = np.array(diffs) >= 0
    exactps = [2 * float(np.sum(unders)) / len(diffs), 2 * float(np.sum(overs)) / len(diffs)]
    pval = min(exactps)
    nullrejected = pval < alpha
    return bsMeanDiff, bsSEMDiff, upperbound, lowerbound, nullrejected, pval


def singlemodelRDM_bootstraperror(roi, modelname, bsmodelcorrs, alpha=0.05, plotit=False, observed=None, printit=True):
    corrs = np.array(bsmodelcorrs[modelname])
    bsMean, bsSEM = np.mean(corrs), np.std(corrs, ddof=1)
    if plotit:
        f, ax = plt.subplots(figsize=[4, 2])
    else:
        ax = None
    lowerbound, upperbound, SEM, throwawayax = sampledist(corrs, len(corrs), alpha, plotit=plotit, ax=ax,
                                                          observed=observed)
    if printit:
        print "%s-- %s: actualmean=%.3f, bsmean=%.3f, bsSEM=%.3f, CI=%s,-%s" % (
        roi, modelname, observed, bsMean, bsSEM, lowerbound, upperbound)
    return bsMean, bsSEM, upperbound, lowerbound


def transformsimilarities(repdismat, distance):
    '''return transformed similarity matrices'''
    rdm = deepcopy(repdismat)
    if distance == 'euclidean':
        rdm=transformeuclideandistancematrix(rdm, len(rdm))
        # trdm = rdm * 0
        # meandist = np.nanmean(rdm)
        # stddist = np.nanstd(rdm, ddof=1)
        # dims = np.array(repdismat).shape
        # for cn, c in enumerate(range(dims[0])):
        #     for c2n, c2 in enumerate(range(dims[1])):
        #         ed = rdm[cn][c2n]
        #         zscored_ed = (ed - meandist) / stddist
        #         trdm[cn][c2n] = zscored_ed
        # rdm = trdm
    if distance == 'pearsonr':
        rdm = np.arctanh(rdm)
    return rdm


def singlemodelRFX(e, roi_summary, models, subjects, subdir, disc, errorbars='withinsubj', printit=True, corrtype='pearsonr', testtype='wilcoxin', timecourse=False, plotit=True):
    '''assess relation between single RDM and single model using RFX across subjects'''
    sel = e.selectors.keys()[0]
    modelsummaries = {m: [] for m in models}
    for subject in subjects:
        if roi_summary.roi in subject.rois.keys():
            if timecourse:
                subjfilename = '%s_%s_%s_%sRSA.pkl' %(e.trshift, disc,sel, roi_summary.roi)
            else:
                subjfilename = '%s_%s_%sRSA.pkl' %(disc,sel, roi_summary.roi)
            #result = None
            with open(subject.subjanalysisdir + subdir + subjfilename, 'r') as resultfile:
                result = pickle.load(resultfile)
            for m in models:
                corr = result.modelcorrs[m]
                modelsummaries[m].append(corr)
    sems, withinsubjsems = plotRFXresults(roi_summary.roi, subjects, models, modelsummaries, errorbars='withinsubj',
                                          plotit=plotit, corrtype=corrtype)
    if printit:
        print "single model: 1-sided %s on %s correlations between neural and model RDMs (null hypothesis: %s=0)" %(testtype,corrtype, corrtype)
    for mn, m in enumerate(models):
        df = len(modelsummaries[m]) - 1
        mean = np.mean(modelsummaries[m])
        if testtype=='wilcoxin':
            T, p, stat = sst.wilcoxonAES(modelsummaries[m])
            stattype='z'
        elif testtype=='ttest':
            transformedrs=[np.arctanh(el) for el in modelsummaries[m]]
            stat,p = sst.ttest_1samp(transformedrs, 0)
            stattype='t'
        else:
            raise RuntimeError("%s is not a recognized test" % (testtype))
        if stat >= 0:
            p = p / 2
        else:
            p = 1 - p / 2
        if printit:
            string = '%s: %s(%.2f): %s(%.0f)=%.3f, p=%.3f.' % (roi_summary.roi, m, mean, stattype, df, stat, p)
            print string
        roi_summary.add_grn2modelsRFX(m, mean, stat, p, df, sems[mn], withinsubjsems[mn])
    return modelsummaries, roi_summary


def comparemodels(comparisontype, e, disc, roi_summary, subjects, modelsummaries, configspec, modelRDMs, conditions,
                printit=True, whichmodels='all', fullorcrossfoldsRDMs='crossfolds'):
    bsmodelcorrs = {}
    if comparisontype == 'stimbootstrap':
        bsmodelcorrs = bootstrapfromstimuli(e, disc, roi_summary.roi, subjects, configspec, modelRDMs, conditions,
                                            fullorcrossfoldsRDMs=fullorcrossfoldsRDMs, corrtype=configspec['corrtype'])
        if printit:
            print "model comparison bootstrap tests comparing %s correlations between neural and model RDMs (%s)" % (configspec['corrtype'],
            comparisontype)
    elif comparisontype == 'condbootstrap':
        bsmodelcorrs = bootstrapfromconditions(disc, modelRDMs, roi_summary.grn, configspec, fullorcrossfoldsRDMs=fullorcrossfoldsRDMs)
        if printit:
            print "model comparison bootstrap tests comparing %s correlations between neural and model RDMs (%s)" % (configspec['corrtype'],comparisontype)
    elif comparisontype == 'RFXsubjects':
        if printit:
            print "model comparison 2-sided signed rank tests comparing %s correlations between neural and model RDMs" % (configspec['corrtype'])
    if whichmodels=='all':
        models = [model for model in modelRDMs.keys() if fullorcrossfoldsRDMs in model]
    else:
        models=[model+'_'+fullorcrossfoldsRDMs for model in whichmodels]
    comparisons = [el for el in itertools.combinations(range(len(models)), 2)]
    for comp in comparisons:
        m1, m2 = models[comp[0]], models[comp[1]]
        compname = m1 + '_VS_' + m2
        array1, array2 = modelsummaries[m1], modelsummaries[m2]
        if comparisontype in ('RFXsubjects', 'stimbootstrap'):
            mean1, mean2 = np.mean(array1), np.mean(array2)
            meandiff = mean1 - mean2
        elif comparisontype == 'condbootstrap':
            mean1, mean2 = roi_summary.grn2models[m1]['corr'], roi_summary.grn2models[m2]['corr']
            meandiff = mean1 - mean2
        if comparisontype == 'RFXsubjects':
            df = len(array1) - 1
            if configspec['testtype']=='wilcoxin':
                T, pval, stat = sst.wilcoxonAES(array1, array2)
                stattype='z'
            elif configspec['testtype']=='ttest':
                transformedarray1=[np.arctanh(el) for el in array1]
                transformedarray1=[np.arctanh(el) for el in array2]
                stat,p = sst.ttest_ind(array1, array2)
                stattype='t'
            else:
                raise RuntimeError("%s is not a recognized test" % (configspec['testtype']))
            string = '%s: %s(%.2f)-%s(%.2f): %s(%.0f)=%.3f, p=%.3f.' % (
            roi_summary.roi, m1, mean1, m2, mean2, stattype,df, stat, pval)
            roi_summary.add_grnmodelcomparisonsRFX(compname, mean1, mean2, stat, pval, df)
        elif comparisontype in ('stimbootstrap', 'condbootstrap'):
            bsMeanDiff, bsSEMDiff, upperbound, lowerbound, nullrejected, pval = comparemodelRDMfits_bootstraptest(m1,
                                                                                                                  m2,
                                                                                                                  bsmodelcorrs,
                                                                                                                  alpha=0.05)
            string = "%s-- %s(%s)-%s(%s): observedMeandiff=%.3f, BSMdiff=%.3f, BSSEMdiff=%.3f, p=%.3f" % (
            roi_summary.roi, m1, mean1, m2, mean2, meandiff, bsMeanDiff, bsSEMDiff, pval)
            roi_summary.add_grnmodelcomparisons(compname, mean1, mean2, bsMeanDiff, pval, [lowerbound, upperbound])
        if printit:
            print string
    return roi_summary


def plotRFXresults(roi, subjs, models, modelsummaries, errorbars='withinsubj', plotit=True, corrtype='pearsonr'):
    '''plot results of RFX on single subject RDMs'''
    subjs = [s for s in subjs if roi in s.rois.keys()]
    plotmeans, plotsems, plotwssems, subjmeans = [], [], [], []
    # for each subject (for this roi), get subject's mean corr across models
    for subj in range(len(subjs)):
        subjms = [modelsummaries[modeln][subj] for modeln in models]
        subjmeans.append(np.mean(subjms))
    normalizedmeans = []
    for subj in range(len(subjs)):
        normalizedsubj = [modelsummaries[modeln][subj] - subjmeans[subj] for modeln in models]
        normalizedmeans.append(normalizedsubj)
    for modeln, m in enumerate(models):
        plotmeans.append(np.mean(modelsummaries[m]))
        plotsems.append(np.std(modelsummaries[m], ddof=1) / np.sqrt(len(modelsummaries[m])))
        normalized = [subjrow[modeln] for subjrow in normalizedmeans]
        plotwssems.append(np.std(normalized, ddof=1) / np.sqrt(len(normalized)))
    if errorbars == 'withinsubj':
        if plotit:
            viz.simplebar(plotmeans, yerr=plotwssems, xticklabels=models,
                          title='%s model-neural correlations /n (avg of single subj corrs)' % (roi),
                          ylabel='%s \n(SEM within subjs)' %(corrtype))
    else:
        if plotit:
            viz.simplebar(plotmeans, yerr=plotsems, xticklabels=models,
                          title='%s model-neural correlations /n (avg of single subj corrs)' % (roi),
                          ylabel='%s \n(SEM across subjs)' % (corrtype))
    return plotsems, plotwssems

class ItemResult():
    def __init__(self,roi,disc,conditions, items, e, subject=None, avgaccuracy=None, avgintensity=None, confusions=None, emoaccuracy=None, emointensity=None):
        self.roi=roi
        self.subject=subject
        self.conditions=conditions
        self.items=items
        self.disc=disc
        self.avgaccuracy=avgaccuracy
        self.avgintensity=avgintensity
        self.confusions=confusions
        self.emoaccuracy=emoaccuracy
        self.emointensity=emointensity
        self.e=e
    def itemsave(self,):
        mypath=os.path.join(self.e.maindir, 'itemwise/')
        if self.subject:
            filename='%s_%s_itemresults_%s.pkl' %(self.roi, self.disc, self.subject)
        else:
            filename='%s_%s_itemresults.pkl' %(self.roi, self.disc)
        if not os.path.exists(mypath): #if the path doesn't exist, make the folder
            os.mkdir(mypath)
        mum.picklethisobject(mypath + filename, self)
        return mypath + filename
    def plotattr(self, attribute, axislabels='items'):
        array=getattr(self,attribute)
        if axislabels=='items':
            xticklabels=self.items
        elif axislabels=='conditions':
            xticklabels=self.conditions
        viz.simplebar(array, xlabel=axislabels, xticklabels=xticklabels, xtickrotation=90, ylabel=attribute)

#perform RSA analysis over time
def timecourseanalysis(e, conditions, windowdur, tcrange, roilist, subjlist, configspec=None, analdir=None, modelkeys=[], modelRDMs=[], disc=None, savetag=None):
    runindsubjects=True
    specmodelkeys=[m+'_'+configspec['fullorcrossfoldsRDMs'] for m in modelkeys]
    timepoints=range(tcrange[0], tcrange[1])
    tcobjs={}
    for roi in roilist:
        tcobj=TimecourseRSA(windowdur, tcrange, analdir=analdir, roi=roi, disc=disc, models=specmodelkeys)
        tcobjs[roi]=tcobj
    for sn,s in enumerate(timepoints):
        print 'working on timepoint %s' % (s)
        ewindow=deepcopy(e)
        ewindow.trshift=s
        ewindow.duration=windowdur
        ewindow.subjects=subjlist
        subjects=ewindow.makesubjects()
        #define individual subjects
        grouprdms, groupftcrdms, subdir=singlesubjanalysis(ewindow, disc, roilist, subjects, subjlist, runindsubjects, configspec, modelRDMs, conditions, savetag, whichmodels='all', svmerrors=None, timecourse=True)
        #noise ceilings
        noiseceilings=computenoiseceiling(grouprdms, roilist, configspec)
        #group average RDMs % single model statistics
        printsinglemodelstats=False
        roi_summaries={}
        modelsummaries={}
        for roi in roilist:
            tcobj=tcobjs[roi]
            roi_summary=ROIsummary(roi, modelkeys, fullorcrossfoldsRDMs=configspec['fullorcrossfoldsRDMs'], corrtype=configspec['similarity'])
            if configspec['neuraltype']=='rawsim':
                roi_summary.grn=np.mean(grouprdms[roi],0) #compute mean of RDMs across subjects
            else:
                print "can't perform timecourse analysis with svmerrors"
                print breakit
            corrs, pvals, models, bestmodel, roi_summary=relateRDMsgrn(roi_summary, modelRDMs, configspec, plotpermutationfig=False, printit=printsinglemodelstats) #single model, group level (permutation for significance)
            modelsummaries[roi], roi_summaries[roi]=singlemodelRFX(ewindow,roi_summary,specmodelkeys, subjects, subdir, disc, errorbars='withinsubj', printit=printsinglemodelstats, corrtype=configspec['corrtype'], testtype=configspec['testtype'], timecourse=True, plotit=False) #single model, RFX across participants
            for modelname in specmodelkeys:
                tpcorr, tperr=gettpresults(roi_summary, modelname, error='ws')
                tpindex=timepoints.index(ewindow.trshift)
                lower=noiseceilings[roi][0]
                upper=noiseceilings[roi][1]
                tcobj.updatetimecourse(modelname, tpindex, tpcorr, tperr, lower=lower, upper=upper)
            tcobjs[roi]=tcobj
    return tcobjs