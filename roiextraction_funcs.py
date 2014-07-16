# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:07:49 2014

@author: amyskerry
"""
import sys
sys.path.append('/mindhive/saxelab/scripts/aesscripts/') #for mypymvpa
import mypymvpa.utilities.visualization as viz
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as sst
import csv
import itertools

def findextractedfiles(studydir,roidir,taskname, key1='', key2=''):
    pscfiles='%s%s/PSC*%s*%s*%s' %(studydir,roidir,key1,taskname,key2)
    foundpscfiles=glob.glob(pscfiles)
    betafiles='%s%s/BETA*%s*%s*%s' %(studydir,roidir,key1,taskname,key2)
    foundbetafiles=glob.glob(betafiles)
    return foundpscfiles, foundbetafiles

def analyzepsc(pscfiles,conditions,colors,subjlist):
    condlabels=[]
    for f in pscfiles:
        with open(f, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            colnames=reader.next()
            subjindex=colnames.index('subject')
            condindex=colnames.index('condition')
            roiindex=colnames.index('roi')
            data=[row for row in reader]
            dataindices=[eln for eln,el in enumerate(colnames) if 'TR' in el]
            labels=[el for el in colnames if 'TR' in el]
            if not conditions:
                conditions=list(set([d[condindex] for d in data]))
            subjlist=[s for s in subjlist if s in set([d[subjindex] for d in data])]
            f,ax=plt.subplots(figsize=[8,6])
            for cn,c in enumerate(conditions):
                hrfs=np.array([d[dataindices[0]:dataindices[-1]] for d in data if d[condindex] == c and d[subjindex] in subjlist])
                hrfs=hrfs.astype('float64')
                meanhrf=np.mean(hrfs,0)
                sqdeviations=[]
                for h in hrfs:
                    sqdeviations.append((h-meanhrf)**2)
                stdhrf=sum(sqdeviations,0)/len(hrfs-1)
                semhrf=stdhrf/np.sqrt(len(hrfs))
                if condlabels:
                    condname=condlabels[c]
                else:
                    condname=c
                #ax.plot(range(len(meanhrf)),meanhrf,color=colors[cn], label=condname)
                ax.errorbar(range(len(meanhrf)),meanhrf,color=colors[cn], yerr=semhrf, label=condname)
            ax.set_title(data[0][roiindex])
            ax.set_xticks(range(len(meanhrf)))
            ax.set_xticklabels(labels, rotation=90)
            legend = plt.legend(loc='upper right',bbox_to_anchor=(1.5, 1.00),ncol=3, shadow=False)
            plt.show()

def analyzebetas(betafiles,conditions,colors,subjlist):
    condlabels=[]
    roilist=[]
    for f in betafiles:
        with open(f, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            colnames=reader.next()
            subjindex=colnames.index('subject')
            condindex=colnames.index('condition')
            roiindex=colnames.index('roi')
            data=[row for row in reader]
            dataindex=colnames.index('mean')
            if not conditions:
                conditions=list(set([d[condindex] for d in data]))
            subjlist=[s for s in subjlist if s in set([d[subjindex] for d in data])]
            f,ax=plt.subplots(figsize=[8,6])
            for cn,c in enumerate(conditions):
                betas=np.array([d[dataindex] for d in data if d[condindex] == c and d[subjindex] in subjlist])
                betas=betas.astype('float64')
                meanbetas=np.zeros([len(conditions),1])
                meanbetas[cn]=np.mean(betas)
                stdbeta=np.std(betas,ddof=1)
                sembeta=stdbeta/np.sqrt(len(betas))
                sembetas=np.zeros([len(conditions),1])
                sembetas[cn]=sembeta
                meanbetas=[m[0] for m in meanbetas]
                sembetas=[sem[0] for sem in sembetas]
                ax.bar(range(len(meanbetas)),meanbetas,yerr=sembetas,color=colors[cn], error_kw={'ecolor':colors[cn]})
            ax.set_title(data[0][roiindex])
            ax.set_xlim([0,len(meanbetas)])
            ax.set_xticks(np.arange(len(meanbetas))+.5)
            ax.set_xticklabels(conditions, rotation=90)
            plt.show()
            roilist.append(data[0][roiindex])
    return roilist

def mushconditions2compare(roi,conditiondict, data, dataindex, subjindex, roiindex, condindex, subjlist):
    subjlist=[s for s in subjlist if s in set([d[subjindex] for d in data])]
    mushes=[]
    means=[]
    sems=[]
    condlabels=[]
    f,ax=plt.subplots(figsize=[3,6])
    for mushedcondn, mushedcond in enumerate(conditiondict.keys()):
        condlabels.append(mushedcond)
        conds=conditiondict[mushedcond]['conds']
        color=conditiondict[mushedcond]['color']
        subjmushes=[]
        for subj in subjlist:
            subjbetas=np.array([d[dataindex] for d in data if d[condindex] in conds and d[subjindex] == subj])
            subjbetas=subjbetas.astype('float64')
            subjmushes.append(np.mean(subjbetas))
        mushes.append(subjmushes)
        condstd=np.std(subjmushes, ddof=1)
        condsem=condstd/np.sqrt(len(subjmushes))
        sems.append(condsem)
        plotmeans=[0,0]
        plotsems=[0,0]
        plotmeans[mushedcondn]=np.mean(subjmushes)
        plotsems[mushedcondn]=condsem
        ax.bar(range(len(plotmeans)),plotmeans,yerr=plotsems,color=color, error_kw={'ecolor':color})
    ax.set_title(data[0][roiindex])
    ax.set_xlim([0,len(plotmeans)])
    ax.set_xticks(np.arange(len(plotmeans))+.5)
    ax.set_xticklabels(condlabels, rotation=90)
    array1=mushes[0]
    array2=mushes[1]
    df=len(array1)-1
    t,p =  sst.ttest_rel(array1, array2)
    string=roi+' :'+condlabels[0]+'-'+condlabels[1]+': t(%.0f)=%.3f, p=%.3f.' % (df,t,p)
    print string

def mushconditions2compare_groups(roi,conditiondict, data, dataindex, subjindex, roiindex, condindex,groups):
    mushes=[]
    means=[]
    sems=[]
    condlabels=[]
    groupmushes=[[] for group in groups]
    for mushedcondn, mushedcond in enumerate(conditiondict.keys()):
        condlabels.append(mushedcond)
        conds=conditiondict[mushedcond]['conds']
        color=conditiondict[mushedcond]['color']
        for groupn,group in enumerate(groups):
            groupname=group['groupname']
            subjlist=group['subjects']
            subjlist=[s for s in subjlist if s in set([d[subjindex] for d in data])]
            subjmushes=[]
            for subj in subjlist:
                subjbetas=np.array([d[dataindex] for d in data if d[condindex] in conds and d[subjindex] == subj])
                subjbetas=subjbetas.astype('float64')
                subjmushes.append(np.mean(subjbetas))
            groupmushes[groupn].append(subjmushes)
    groupcombs=itertools.combinations(groups,2)
    comps=[gc for gc in groupcombs]
    for c in comps:
        g1=c[0]
        g2=c[1]
        group1n=groups.index(g1)
        group2n=groups.index(g2)
        group1name=g1['groupname']
        group2name=g2['groupname']
        group1diffs=np.array(groupmushes[group1n][0])-np.array(groupmushes[group1n][1])
        group2diffs=np.array(groupmushes[group2n][0])-np.array(groupmushes[group2n][1])
        group1mean=np.mean(group1diffs)
        group1sem=np.std(group1diffs,ddof=1)/np.sqrt(len(group1diffs))
        group2mean=np.mean(group2diffs)
        group2sem=np.std(group2diffs,ddof=1)/np.sqrt(len(group2diffs))
        clusteredmeans=[(np.mean(groupmushes[group1n][0]),np.mean(groupmushes[group1n][1])),(np.mean(groupmushes[group2n][0]),np.mean(groupmushes[group2n][1]))]
        clusteredsems=[(np.std(groupmushes[group1n][0], ddof=1)/np.sqrt(len(groupmushes[group1n][0])),np.std(groupmushes[group1n][1], ddof=1)/np.sqrt(len(groupmushes[group1n][1]))),(np.std(groupmushes[group2n][0], ddof=1)/np.sqrt(len(groupmushes[group2n][0])),np.std(groupmushes[group2n][1], ddof=1)/np.sqrt(len(groupmushes[group2n][1])))]
        t,p =  sst.ttest_ind(group1diffs, group2diffs)
        df=len(group1diffs)+len(group2diffs)-2
        string=roi+' :'+group1name+'-'+group2name+': t(%.0f)=%.3f, p=%.3f.' % (df,t,p)
        viz.simpleclusterbar(clusteredmeans, yerr=clusteredsems, figsize=[3,3], xlabel=roi, bar_labels=condlabels, group_labels=[group1name,group2name], bar_colors=['blue','purple'])
    print string

def compareconds(studydir,roidir,taskname,roilist,comparisons, subjlist):
    for comp in comparisons:
        print comp
        for roi in roilist:
            roipscfiles,roibetafiles=findextractedfiles(studydir,roidir,taskname, key1=roi)
            if len(roibetafiles)>1:
                print "warning: found multiple beta files"
            f=roibetafiles[0]
            with open(f, 'rU') as csvfile:
                reader = csv.reader(csvfile)
                colnames=reader.next()
                subjindex=colnames.index('subject')
                condindex=colnames.index('condition')
                roiindex=colnames.index('roi')
                dataindex=colnames.index('mean')
                data=[row for row in reader]
            mushconditions2compare(roi,comp, data, dataindex, subjindex, roiindex, condindex, subjlist)

def comparegroupbetas(studydir,roidir,taskname,roilist,comparisons,groups):
    '''compares 2 groups. groups should be list dicts with keys "groupname" and "subjects"'''
    for comp in comparisons:
        print comp
        for roi in roilist:
            roipscfiles,roibetafiles=findextractedfiles(studydir,roidir,taskname, key1=roi)
            if len(roibetafiles)>1:
                print "warning: found multiple beta files"
            f=roibetafiles[0]
            with open(f, 'rU') as csvfile:
                reader = csv.reader(csvfile)
                colnames=reader.next()
                subjindex=colnames.index('subject')
                condindex=colnames.index('condition')
                roiindex=colnames.index('roi')
                dataindex=colnames.index('mean')
                data=[row for row in reader]
            mushconditions2compare_groups(roi,comp, data, dataindex, subjindex, roiindex, condindex, groups)

