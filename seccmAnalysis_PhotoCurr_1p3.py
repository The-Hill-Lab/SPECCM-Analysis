#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 1.3 7/30/2019
Caleb M. Hill
Department of Chemistry
University of Wyoming
caleb.hill@uwyo.edu

This program was designed for the purpose of analyzing photoelectrochemical microscopy data acquired in the Hill Lab at the University of Wyoming. This encompasses several specific tasks:
-Fitting of voltammograms to obtain dark current and photocurrent values
-Construction of current "movies" (dark currents and photocurrents)
-Generation of current images (dark currents and photocurrents)
-Generation of onset potential images
-GUI-assisted generation of voltammograms corresponding to specific points

The data is expected in the form of a 2D array, with Nt points in the first dimension (corresponding to different points in time) and Nx*Ny+1 points in the second dimension, corresponding to different spatial pixels. The first column is expected to hold the potentials corresponding to each point in time. The software expects current values in A, and potentials in V.
"""

#Import all the packages you'll need
import tkinter as tk
from tkinter import filedialog 
import scipy as sp
from scipy import optimize 
from scipy import signal 
import matplotlib
#matplotlib.use("TkAgg") #This line seems to impart Mac functionality
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import matplotlib.colorbar as cb 
import matplotlib as mpl
from matplotlib import gridspec
import pandas as pd
from tifffile import imsave
import os
from matplotlib.colors import LinearSegmentedColormap

		
#Get file path via GUI and make new paths for output
root = tk.Tk()
root.withdraw()
filePath = filedialog.askopenfilename()
basePath, baseName = os.path.split(os.path.normpath(filePath))
imgsPath = os.path.join(basePath, baseName + '_output', baseName + '_PhotoCurrImgs','')
cvsPath = os.path.join(basePath, baseName + '_output', baseName + '_CVs','')
if not os.path.isdir(imgsPath):
    os.makedirs(imgsPath)
if not os.path.isdir(cvsPath):
    os.makedirs(cvsPath)

#Import data
print('Importing data...')
rawData = pd.read_csv(filePath,dtype=sp.float64,sep='\t').as_matrix()
Nt = int(rawData.shape[0])
N = rawData.shape[1] - 1
Nx = int(input("What's the image width (in pixels)?")) #Note that this has to be imported manually. We always specify dimensions in the data filename.
Ny = int(N/Nx)
print('Image dimensions are x:', Nx, ', y:', Ny)

#Create array with potential values				
E = rawData[:,0]

#Format data cube. Note that this takes into account the alternating raster pattern of the Hill Lab SECCM instruments.                
dataCube = sp.zeros((Ny,Nx,Nt),dtype=sp.float64)
for nx in range(Nx):
	for ny in range(Ny):
		print('Reformatting data:', round((nx + ny/Ny)/Nx*100,1), 'percent complete.',end='\r')
		for nz in range(Nt):
			datapoint = rawData[nz,nx+ny*Nx+1]
			if ny % 2 == 0:
				dataCube[ny,nx,nz] = datapoint
			else:
				dataCube[ny,-(nx+1),nz] = datapoint
print('Reformatting data: 100.0 percent complete.')


Asc = 1e-12 #Scales to desired units of pA
y0 = -38.4  #Background correction (in pA)

#This defines the primary fitting function for extracting photocurrents. The "dutyCycle" and "freq" values must be changed to values appropriate to the source employed. The fitting model is just the sum of a second order polynomial (dark currents) and another second order polynomial multiplied by a square waveform.
dutyCycle = 0.53
freq = 25
def fitFun2(x,ad,bd,cd,ah,bh,ch,th):
    return (ad*x**2 + bd*x + cd) + (ah*x**2 + bh*x + ch)*(0.5*signal.square(2*sp.pi*x*freq+th,dutyCycle)+0.5)

#Generate photocurrent data from raw data
trim1 = 0 #These "trim" values can be used to select a particular part of the voltammograms for analysis (e.g., only the first sweep)
trim2 = 0
photoCube = sp.zeros((Ny,Nx,Nt-(trim1+trim2)),dtype=sp.float64)
darkCube = sp.zeros((Ny,Nx,Nt-(trim1+trim2)),dtype=sp.float64)
ntot = 0
fails = 0
fig = plt.figure()
ax = fig.add_subplot(111)
xData = E[trim1:Nt-trim2]
line1, = ax.plot(xData,dataCube[0,0,trim1:Nt-trim2], 'b-')
line2, = ax.plot(xData,dataCube[0,0,trim1:Nt-trim2], 'r-')
line3, = ax.plot(xData,dataCube[0,0,trim1:Nt-trim2], 'g-')
line4, = ax.plot(xData,dataCube[0,0,trim1:Nt-trim2], 'k-')
for nx in range(Nx):
    for ny in range(Ny):
        print('Calculating photocurrents:', round((nx + ny/Ny)/Nx*100,1), 'percent complete.',end='\r')
        yData = dataCube[ny,nx,trim1:Nt-trim2]/Asc - y0
        yFit = sp.zeros(sp.shape(yData),dtype=sp.float64)
        yPhoto = sp.zeros(sp.shape(yData),dtype=sp.float64)
        yDark = sp.zeros(sp.shape(yData),dtype=sp.float64)
        pieceSize = 600         #This controls the size of the pieces in the piece-wise fitting algorithm
        overlap = int(0.2*pieceSize)
        nPieces = len(range(0,len(yData),pieceSize))
        yFitPieces = list()
        yPhotoPieces = list()
        yDarkPieces = list()
        startPositions = range(0,len(yData),pieceSize)
        for i in range(nPieces):
                if i==0:
                        startPos = 0
                else:
                        startPos = startPositions[i]-overlap
                stopPos = startPositions[i]+pieceSize
                xPiece = xData[startPos:stopPos]
                yPiece = yData[startPos:stopPos]
                p1 = sp.polyfit(xPiece,yPiece,2)
                testAngles = sp.linspace(0,2*sp.pi,num=50,endpoint=True)
                minError = sp.inf
                for phi in testAngles:
                        err = sp.sum((yPiece-fitFun2(xPiece,0,0,0,*p1,phi))**2)
                        if err < minError:
                                minError = err
                                minPhi = phi
                p2 = minPhi
                f = lambda p: fitFun2(xPiece,p[0],p[1],p[2],p[3],p[4],p[5],p2) - yPiece
                try:
                        optResult = sp.optimize.least_squares(f,x0=[*p1,*p1],verbose=0)
                        p3 = optResult.x                        
                except RuntimeError:
                        p3 = [*p1,0.5,0.0]
                        fails += 1
                yFitPieces.append(fitFun2(xPiece,*p3,p2))
                yPhotoPieces.append(sp.polyval(p3[3:6],xPiece))
                yDarkPieces.append(sp.polyval(p3[0:3],xPiece))
        for i in range(nPieces):
                if i==nPieces-1:
                        yFit[i*pieceSize::] = yFitPieces[i][overlap::]
                else:
                        if i==0:
                                yFit[i*pieceSize:(i+1)*pieceSize-overlap] = yFitPieces[i][0:-overlap]
                        else:
                                yFit[i*pieceSize:(i+1)*pieceSize-overlap] = yFitPieces[i][overlap:-overlap]
                        for j in range(overlap):
                                yFit[(i+1)*pieceSize-(overlap-j)] = j/overlap*yFitPieces[i+1][j] + (overlap-j)/overlap*yFitPieces[i][-(overlap-j)]
                if i==nPieces-1:
                        yPhoto[i*pieceSize::] = yPhotoPieces[i][overlap::]
                        yDark[i*pieceSize::] = yDarkPieces[i][overlap::]
                else:
                        if i==0:
                                yPhoto[i*pieceSize:(i+1)*pieceSize-overlap] = yPhotoPieces[i][0:-overlap]
                                yDark[i*pieceSize:(i+1)*pieceSize-overlap] = yDarkPieces[i][0:-overlap]
                        else:
                                yPhoto[i*pieceSize:(i+1)*pieceSize-overlap] = yPhotoPieces[i][overlap:-overlap]
                                yDark[i*pieceSize:(i+1)*pieceSize-overlap] = yDarkPieces[i][overlap:-overlap]
                        for j in range(overlap):
                                yPhoto[(i+1)*pieceSize-(overlap-j)] = j/overlap*yPhotoPieces[i+1][j] + (overlap-j)/overlap*yPhotoPieces[i][-(overlap-j)]
                                yDark[(i+1)*pieceSize-(overlap-j)] = j/overlap*yDarkPieces[i+1][j] + (overlap-j)/overlap*yDarkPieces[i][-(overlap-j)]
        if ntot%25==0:
                line1.set_ydata(yData)
                ax.set_ylim(sp.amin(yData),sp.amax(yData))
                line2.set_ydata(yFit)
                line3.set_ydata(yPhoto+yDark)
                line4.set_ydata(yDark)
                fig.canvas.draw()
                plt.pause(0.001)
        photoCube[ny,nx,:] = yPhoto
        darkCube[ny,nx,:] = yDark
        ntot += 1        
print('Calculating photocurrents: 100.0 percent complete.')
print(str(fails/ntot*100) + ' percent failure rate')

fig, current_ax = plt.subplots()
totalCurr = sp.average(dataCube,axis=2)  
totCurrPlot = plt.imshow(totalCurr,interpolation='none',origin='lower')
plt.show()

#Query user as to whether sweep is reversed. This can be used if you would like exported movies to play in reverse. It also affects the calculation of the onset potentials described below.
sweepReversed = input("Is the sweep direction reversed? (y/n)")

#Generate photocurrent movie if desired.
choice = input("Would you like a photocurrent movie? (y/n)")
if choice == 'y':
        ytrim1 = int(input("Trim n rows in the y-direction at start of scan? (0 if no)"))
        ytrim2 = int(input("Trim n rows in the y-direction at end of scan? (0 if no)"))
        datamin = float(input("What's the plot minimum? (in pA)"))
        datamax = float(input("What's the plot maximum? (in pA)"))
        print('Movie is being prepared... Be patient...') #It does take a bit.
        fig = plt.figure()
        gs = gridspec.GridSpec(1,2,width_ratios=[10,1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax1.set_xlabel("x / micron")
        ax1.set_ylabel("y / micron")

        colors = [(1, 0, 0), (0, 0, 0), (0, 1, 0)]
        cm = LinearSegmentedColormap.from_list("Yep",colors,N=256)
        norm = mpl.colors.Normalize(vmin=datamin, vmax=datamax)
        imagecb = cb.ColorbarBase(ax2,norm=norm,format='%.0f',cmap=cm)
        imagecb.ax.set_title('i / pA')

        ims = []
        textx = 0.05*Nx
        texty = 0.95*(Ny-(ytrim1+ytrim2))
        for nt in range(0,Nt-(trim1+trim2)):
                if sweepReversed == 'y':
                        n = -1*(nt+1)
                else:
                        n = nt
                frame = photoCube[ytrim1:Ny-ytrim2,:,n]
                im = ax1.imshow(frame,norm=norm,interpolation='none',origin='lower',cmap=cm)
                tx = ax1.text(textx,texty, 'E = ' + '{0:.3f}'.format(xData[n]) + ' V',color='w',family='sans-serif',size=24, va='top',ha='left',weight='bold')#bbox={'facecolor':'black', 'alpha':0.1, 'pad':5}
                ims.append([im,tx])
                
        ani = animation.ArtistAnimation(fig, ims, interval=10, repeat=False)

        path = imgsPath + baseName + '_PhotoCurr_Movie.mp4'
        ani.save(path,bitrate=20000)

        plt.show()

#Generate dark current movie if desired.
choice = input("Would you like a dark current movie? (y/n)")
if choice == 'y':
        ytrim1 = int(input("Trim n rows in the y-direction at start of scan? (0 if no)"))
        ytrim2 = int(input("Trim n rows in the y-direction at end of scan? (0 if no)"))
        datamin = float(input("What's the plot minimum? (in pA)"))
        datamax = float(input("What's the plot maximum? (in pA)"))
        print('Movie is being prepared... Be patient...') #It does take a bit.
        fig = plt.figure()
        gs = gridspec.GridSpec(1,2,width_ratios=[10,1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax1.set_xlabel("x / micron")
        ax1.set_ylabel("y / micron")

        colors = [(1, 0, 0), (0, 0, 0), (0, 1, 0)]
        cm = LinearSegmentedColormap.from_list("Yep",colors,N=256)
        norm = mpl.colors.Normalize(vmin=datamin, vmax=datamax)
        imagecb = cb.ColorbarBase(ax2,norm=norm,format='%.0f',cmap=cm)
        imagecb.ax.set_title('i / pA')

        ims = []
        textx = 0.05*Nx
        texty = 0.95*(Ny-(ytrim1+ytrim2))
        for nt in range(0,Nt-(trim1+trim2)):
                if sweepReversed == 'y':
                        n = -1*(nt+1)
                else:
                        n = nt
                frame = darkCube[ytrim1:Ny-ytrim2,:,n]
                im = ax1.imshow(frame,norm=norm,interpolation='none',origin='lower',cmap=cm)
                tx = ax1.text(textx,texty, 'E = ' + '{0:.3f}'.format(xData[n]) + ' V',color='w',family='sans-serif',size=24, va='top',ha='left',weight='bold')#bbox={'facecolor':'black', 'alpha':0.1, 'pad':5}
                ims.append([im,tx])
                
        ani = animation.ArtistAnimation(fig, ims, interval=10, repeat=False)

        path = imgsPath + baseName + '_DarkCurr_Movie.mp4'
        ani.save(path,bitrate=20000)

        plt.show()
			
#GUI-triggered functions to output voltammograms for a single region. Each output file contains 4 columns: the potential, raw current signal, dark current fit, and photocurrent fit.
ncv = 1
def exportCV(x,y):
        global ncv
        curr = dataCube[y,x,trim1:Nt-trim2]/Asc - y0
        darkCurr = darkCube[y,x,:]
        photoCurr = photoCube[y,x,:]
        cv = sp.stack((E[trim1:Nt-trim2],curr,darkCurr,photoCurr),axis=-1)
        path = cvsPath + baseName + '_' + str(ncv) + '_x' + str(x) + '_y' + str(y) + '.txt'
        sp.savetxt(path,cv,delimiter='\t')
        ncv += 1
	
def onclick(event):
	x = int(round(event.xdata))
	y = int(round(event.ydata))
	exportCV(x,y)
	print(x, y)

#This creates still current images at a series of different potentials.
Nimgs = 10
Nframes = int(sp.floor((Nt-(trim1+trim2))/Nimgs))
for nimg in range(Nimgs):
    ni = nimg*Nframes
    nf = ni + Nframes
    photoCurrImage = sp.average(photoCube[:,:,ni:nf],axis=2)
    darkCurrImage = sp.average(darkCube[:,:,ni:nf],axis=2)
    photoPath = imgsPath + baseName + '_PhotoCurr_Image_' + str(round(sp.average(E[ni:nf]),3)) + 'V.txt'
    darkPath = imgsPath + baseName + '_DarkCurr_Image_' + str(round(sp.average(E[ni:nf]),3)) + 'V.txt'
    sp.savetxt(photoPath,photoCurrImage,delimiter='\t')
    sp.savetxt(darkPath,darkCurrImage,delimiter='\t')

#This calculates an image of onset potentials. It does this in the simplest fashion of finding the first data point which exceeds a specified value. Note that if no such data point is found, the value is (roughly) pi.
iThreshold = float(input("What's the threshold current? (in pA)"))
EonImagePhoto = sp.zeros((Ny,Nx),dtype=sp.float64) + 3.14       
EonImageDark = sp.zeros((Ny,Nx),dtype=sp.float64) + 3.14
for nx in range(Nx):
        for ny in range(Ny):
                for nt in range(Nt-(trim1+trim2)):
                        if sweepReversed == 'y':
                                n = -1*(nt+1)
                        else:
                                n = nt
                        if photoCube[ny,nx,n]>iThreshold:
                                EonImagePhoto[ny,nx] = xData[n]
                                break
                for nt in range(Nt-(trim1+trim2)):
                        if sweepReversed == 'y':
                                n = -1*(nt+1)
                        else:
                                n = nt
                        if darkCube[ny,nx,n]>iThreshold:
                                EonImageDark[ny,nx] = xData[n]
                                break
photoPath = imgsPath + baseName + '_Eon_ImagePhoto.txt'
sp.savetxt(photoPath,EonImagePhoto,delimiter='\t')
darkPath = imgsPath + baseName + '_Eon_ImageDark.txt'
sp.savetxt(darkPath,EonImageDark,delimiter='\t')

#This just plots the image for a final time and binds the GUI functions to the figure.
fig, current_ax = plt.subplots()
totalCurr = sp.average(dataCube,axis=2)  
totCurrPlot = plt.imshow(totalCurr,interpolation='none',origin='lower')
plt.connect('button_press_event', onclick)
plt.show()