__author__ = 'askerry'
import os
import pandas as pd
import sys
sys.path.append('/mindhive/saxelab/scripts/aesscripts/') #for mypymvpa
sys.path.append('/software/python/anaconda/lib/python2.7/site-packages/') #for h5py
from mypymvpa.utilities.misc import makeIDs
import glob
import glob
import scipy.io
import h5py
import numpy as np
import warnings

def extractbadrunconds(studydir,task, subjtaskfile, taskkey):
    subjectinfo=loadsubjmat(os.path.join(studydir, subjtaskfile), taskkey)
    arts, behaves=loadsubjartandbehave(studydir, task, subjectinfo)
    timecourses, mapping=gettimecourses(behaves)
    dropconds=findbadconds(timecourses, arts, mapping)
    return dropconds

def loadsubjartandbehave(studydir, task, subjectinfo):
    behaves,arts={},{}
    for subj in subjectinfo.keys():
        subjdir=os.path.join(studydir, subj)
        rundirs=subjectinfo[subj][task]
        arts[subj], behaves[subj]=[],[]
        for rn,r in enumerate(rundirs):
            bfile=glob.glob(os.path.join(studydir,'behavioural', subj+'.'+task+'.'+str(rn+1)+'.mat'))[0]
            afile=glob.glob(os.path.join(studydir,subj, 'bold', '*'+str(r), '*art*outliers_s*'))[0]
            behave=extractbehave(bfile)
            behaves[subj].append(behave)
            arts[subj].append(getarts(afile, behave['ips']))
    return arts, behaves

def extractbehave(bf):
    '''loads a behavioral file, returns behavioral dict. so many hacks because interfacing with .mat files is hell!'''
    #some .mats require the h5py, whereas others require scipy. lord knows why.
    try:
        b = scipy.io.loadmat(bf)
    except:
        try:
            b = hdfbehaveextract(bf)
        except:
            b = hdfbehaveextract2(bf)
    for key in b.keys():
        try:
            b[key] = list(b[key].squeeze())
        except:
            pass
    return b


def hdfbehaveextract(bf):
    '''function to extract .mats using h5py. output should be a dict with items corresponding to .mat structure fields, just as would be returned by scipy.io.loadmat'''
    bhdf = h5py.File(bf, 'r')
    bips = int(bhdf['ips'][0][0])
    brt = [float(el) for el in bhdf['RT'][0]]
    brun = [int(el) for el in bhdf['run'][0]]
    bkey = [int(el) for el in bhdf['key'][0]]
    stims_unicode = [bhdf[el] for el in bhdf['item_orders'][0]]
    bitem_orders = []
    bspm_i = bhdf['spm_inputs']
    numconditions = len(bspm_i['name'])
    bspm_inputs = []
    for c in range(numconditions):
        onsets = bhdf[bspm_i['ons'][c][0]]
        onsets = np.array([[float(el)] for el in onsets[0]])
        durations = bhdf[bspm_i['dur'][c][0]]
        durations = np.array([[el] for el in durations[0]])
        condition = ''.join(unichr(i[0]) for i in bhdf[bspm_i['name'][c][0]])
        row = ([condition], onsets, durations)
        bspm_inputs.append(row)
    bspm_inputs = bspm_inputs
    for stim in stims_unicode:
        bitem_orders.append([u''.join(unichr(i[0]) for i in stim)])
    b = {'ips': bips, 'spm_inputs': bspm_inputs, 'RT': brt, 'item_orders': bitem_orders, 'run': brun, 'key': bkey}
    return b


def hdfbehaveextract2(bf):
    '''function to extract .mats using h5py. output should be a dict with items corresponding to .mat structure fields, just as would be returned by scipy.io.loadmat'''
    bhdf = h5py.File(bf, 'r')
    bips = int(bhdf['ips'][0][0])
    brt = [float(el) for el in bhdf['RT'][0]]
    brun = [int(el) for el in bhdf['run'][0]]
    bkey = [int(el) for el in bhdf['key'][0]]
    stims_unicode = bhdf['item_orders']
    numtrials = stims_unicode.shape[1]
    bitem_orders = []
    for trial in range(numtrials):
        bitem_orders.append([u''.join(unichr(i[trial]) for i in stims_unicode.value)])
    bspm_i = bhdf['spm_inputs']
    numconditions = len(bspm_i['name'])
    bspm_inputs = []
    for c in range(numconditions):
        onsets = bhdf[bspm_i['ons'][c][0]]
        onsets = np.array([[float(el)] for el in onsets[0]])
        durations = bhdf[bspm_i['dur'][c][0]]
        durations = np.array([[el] for el in durations[0]])
        condition = ''.join(unichr(i[0]) for i in bhdf[bspm_i['name'][c][0]])
        row = ([condition], onsets, durations)
        bspm_inputs.append(row)
    bspm_inputs = bspm_inputs
    b = {'ips': bips, 'spm_inputs': bspm_inputs, 'RT': brt, 'item_orders': bitem_orders, 'run': brun, 'key': bkey}
    return b

def getarts(afile, ips, printit=False):
    '''marks arts with a 1'''
    artsloaded = h5py.File(afile, 'r')
    mat = np.array(artsloaded['R'])
    if type(mat[0]) is not np.ndarray:
        if printit:
            warnings.warn("found strange art file... ignoring arts found in %s" % (artfiles[0]))
        artvector = list(np.zeros(int(ips)))
    else:
        artvector = list(sum(mat, 0))
    return artvector

def loadsubjmat(matfile, globalkey):
    '''expects s.ID, s.ASD, s.EmoBioLoc, s.tomloc, s.EIB_main, but should be flexible-ish'''
    print "loading file: " + matfile + " as subjinfo"
    try:
        f = scipy.io.loadmat(matfile)
    except:
        f = h5py.File(matfile, 'r')
        #globalkey = f.keys()[1]
    data = f.get(globalkey)
    keys = data.keys()
    subjects = {}
    for a in range(len(data[keys[0]])):
        subject = {}
        for key in keys:
            try:
                entry = f[data[key][a][0]]
            except:
                entry = data[key]
            if entry.dtype == 'u2':
                entry = u''.join(unichr(i[0]) for i in entry.value)
            else:
                try:
                    #if subj mat contains entries that should be floats (like age) add them here
                    if key in ('age'):
                        entry = [float(i[0]) for i in entry.value]
                    else:
                        entry = [int(i[0]) for i in entry.value]
                except:
                    entry = [(i) for i in entry.value]
            subject[key] = entry
        subjects[subject['ID']]=subject
    return subjects

def gettimecourses(behaves):
    timecourses={}
    for subj in behaves.keys():
        timecourses[subj]=[]
        behavedicts=behaves[subj]
        for b in behavedicts:
            timecourse=np.zeros(int(b['ips']))
            mapping={}
            for condn,cond in enumerate(b['spm_inputs']):
                name=cond[0][0]
                mapping[name]=condn+1
                ons=int(cond[1][0][0])
                dur=int(cond[2][0][0])
                for TR in range(dur):
                    timecourse[ons-1+TR]=condn+1
            timecourses[subj].append(timecourse)
    return timecourses, mapping

def findbadconds(timecourses, arts, mapping):
    inversemapping={item[1]:item[0] for item in mapping.items()}
    dropconds={}
    for subj in timecourses.keys():
        dropconds[subj]={}
        tc=timecourses[subj]
        atc=arts[subj]
        for runn,rtc in enumerate(tc):
            dropconds[subj][runn]=[]
            rtc=np.array(rtc)
            ract=np.array(atc[runn])
            for cond in [el for el in np.unique(rtc) if el!=0]:
                relarts=ract[np.where(rtc==cond)]
                if len(relarts)-sum(relarts)<3:
                    dropconds[subj][runn].append(inversemapping[cond])
    return dropconds