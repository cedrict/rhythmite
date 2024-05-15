#!/usr/bin/env python3

###  L'Heureux Diagenesis modelling ###

import numpy as np
# functions for saving output of python model to file
# ASCII and VTU formats


def export_to_ascii(var,istep,coords,f1,f2,f3,f4,f5,f6,f7):
    filename = 'solution_{}_{:06d}.ascii'.format(var, istep)
    np.savetxt(filename,np.array([coords,f1,f2,f3,f4,f5,f6,f7]).T,fmt='%1.5e',\
               header='x AR CA ca co phi u v')


def export_to_vtu(istep,coords,f1,f2,f3,f4,f5,f6,f7):
    # AR, CA, ca, co, phi

    m=4
    nnx=2
    nny=len(coords)
    nelx=1
    nely=nny-1
    N=nnx*nny
    nel=nelx*nely
    Ly=max(coords)
    Lx=Ly/10
    x = np.empty(N,dtype=np.float64)
    y = np.empty(N,dtype=np.float64)
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            counter += 1
    icon =np.zeros((m, nel),dtype=np.int32)
    counter = 0
    for j in range(0, nely):
        for i in range(0, nelx):
            icon[0, counter] = i + j * (nelx + 1)
            icon[1, counter] = i + 1 + j * (nelx + 1)
            icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
            icon[3, counter] = i + (j + 1) * (nelx + 1)
            counter += 1

    filename = 'solution_{:06d}.vtu'.format(istep)
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1'byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '>\n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3'Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####
    #vtufile.write("<CellData Scalars='scalars'>\n")
    #vtufile.write("</CellData>\n")
    #####
    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='AR' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f1[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='CA' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f2[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='c_ca' Format='ascii'>\n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f3[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='c_co' Format='ascii'>\n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f4[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='phi' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f5[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f6[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='w' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" %f7[nny-1-j])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")

    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity'Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d\n"
%(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets'Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


###############################################################################
def export_to_vtu2(nstep,coords,soln,vel_U,vel_W,xbot,xtop,xscale,dt):
    # AR, CA, ca, co, phi, u ,W

    m=4
    nnx=nstep
    nny=len(coords)
    nelx=nnx-1
    nely=nny-1
    N=nnx*nny
    nel=nelx*nely
    Ly=max(coords)
    Lx=Ly*2
    h = Ly/(nny-1) #???

    x = np.empty(N,dtype=np.float64)
    y = np.empty(N,dtype=np.float64)
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            x[counter]=i*Lx/float(nelx)
            y[counter]=j*Ly/float(nely)
            counter += 1
    icon =np.zeros((m, nel),dtype=np.int32)
    counter = 0
    for j in range(0, nely):
        for i in range(0, nelx):
            icon[0, counter] = i + j * (nelx + 1)
            icon[1, counter] = i + 1 + j * (nelx + 1)
            icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
            icon[3, counter] = i + (j + 1) * (nelx + 1)
            counter += 1

    filename = 'solution.vtu'
    vtufile=open(filename,"w")
    vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
    vtufile.write("<UnstructuredGrid> \n")
    vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '>\n" %(N,nel))
    #####
    vtufile.write("<Points> \n")
    vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
    for i in range(0,N):
        vtufile.write("%10f %10f %10f \n" %(x[i],y[i],0.))
    vtufile.write("</DataArray>\n")
    vtufile.write("</Points> \n")
    #####

    vtufile.write("<PointData Scalars='scalars'>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='AR' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" % soln[nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='CA' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" % soln[nny+nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='c_ca' Format='ascii'>\n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" % soln[2*nny+nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='c_co' Format='ascii'>\n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" % soln[3*nny+nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='phi' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%10f \n" % soln[4*nny+nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='u' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%8f \n" %vel_U[nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")

    #--
    vtufile.write("<DataArray type='Float32' Name='CFL nb (u)' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%8f \n" % (dt*vel_U[nny-1-j,i]/h))
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='w' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%8f \n" %vel_W[nny-1-j,i])
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Float32' Name='CFL nb (w)' Format='ascii'> \n")
    counter = 0
    for j in range(0, nny):
        for i in range(0, nnx):
            vtufile.write("%8f \n" % (dt*vel_W[nny-1-j,i]/h))
            counter += 1
    vtufile.write("</DataArray>\n")
    #--
    #vtufile.write("<DataArray type='Float32' Name='heaviside' Format='ascii'> \n")
    #counter = 0
    #for j in range(0, nny):
    #    for i in range(0, nnx):
    #        vtufile.write("%10f \n" % heaviside(Ly-y[counter],xbot,xtop,xscale))
    #        counter += 1
    #vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</PointData>\n")

    #####
    vtufile.write("<Cells>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d %d %d %d\n" %(icon[0,iel],icon[1,iel],icon[2,iel],icon[3,iel]))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %((iel+1)*4))
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
    for iel in range (0,nel):
        vtufile.write("%d \n" %9)
    vtufile.write("</DataArray>\n")
    #--
    vtufile.write("</Cells>\n")
    #####
    vtufile.write("</Piece>\n")
    vtufile.write("</UnstructuredGrid>\n")
    vtufile.write("</VTKFile>\n")
    vtufile.close()


























Lx=1
nnx=100

x = np.linspace(0,Lx,nnx,dtype=np.float64)

AR = np.zeros(nnx)
CA = np.zeros(nnx)
c1 = np.zeros(nnx)
c2 = np.zeros(nnx)
phi = np.zeros(nnx)
u = np.zeros(nnx)
w = np.zeros(nnx)

AR=np.random.rand(nnx)
CA=np.random.rand(nnx)
c1=np.random.rand(nnx)
c2=np.random.rand(nnx)
phi=np.random.rand(nnx)
u=np.random.rand(nnx)
w=np.random.rand(nnx)

istep=123

export_to_ascii('t',istep,x,AR,CA,c1,c2,phi,u,x)
