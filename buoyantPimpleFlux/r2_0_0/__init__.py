#!/usr/bin/env python

#--------------------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey Petrov, Andrey Simurzin
##

#---------------------------------------------------------------------------
from Foam import ref, man


#---------------------------------------------------------------------------
def createFields( runTime, mesh, g ):
    ref.ext_Info()<< "Reading thermophysical properties\n" << ref.nl
    
    pThermo = man.basicRhoThermo.New( mesh )
    
    rho = man.volScalarField( man.IOobject( ref.word( "rho" ), 
                                            ref.fileName( runTime.timeName() ),
                                            mesh, 
                                            ref.IOobject.NO_READ, 
                                            ref.IOobject.NO_WRITE ),
                              man.volScalarField( pThermo.rho(), man.Deps( pThermo ) ) )

    p =  man.volScalarField( pThermo.p(), man.Deps( pThermo ) ) 
    h =  man.volScalarField( pThermo.h(), man.Deps( pThermo ) ) 
    psi = man.volScalarField( pThermo.psi(), man.Deps( pThermo ) ) 
  

    ref.ext_Info()<< "Reading field U\n" << ref.nl
    U = man.volVectorField( man.IOobject( ref.word( "U" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh ) 


    phi = man.compressibleCreatePhi( runTime, mesh, U, rho )

    ref.ext_Info()<< "Creating turbulence model\n" << ref.nl
    turbulence = man.compressible.turbulenceModel.New( rho, U, phi, pThermo )

    ref.ext_Info()<< "Calculating field g.h\n" << ref.nl
    gh = man.volScalarField( ref.word( "gh" ), man.volScalarField(  g & mesh.C(), man.Deps( mesh ) ) )
    ghf = man.surfaceScalarField( ref.word( "ghf" ), man.surfaceScalarField( g & mesh.Cf(), man.Deps( mesh ) ) )

    ref.ext_Info()<< "Reading field p_rgh\n" << ref.nl
    p_rgh = man.volScalarField( man.IOobject( ref.word( "p_rgh" ),
                                            ref.fileName( runTime.timeName() ),
                                            mesh,
                                            ref.IOobject.MUST_READ,
                                            ref.IOobject.AUTO_WRITE ),
                              mesh )

    # Force p_rgh to be consistent with p
    p_rgh <<= p - rho*gh

    ref.ext_Info()<< "Creating field DpDt\n" << ref.nl
    
    DpDt = man.volScalarField( ref.word( "DpDt" ), 
                               man.fvc.DDt( man.surfaceScalarField( ref.word( "phiU" ), phi / man.fvc.interpolate( rho ) ), p ) )

    return pThermo, p, rho, h, psi, U, phi, turbulence, gh, ghf, p_rgh, DpDt


#---------------------------------------------------------------------------
def fun_Ueqn( pimple, mesh, rho, U, phi, turbulence, ghf, p_rgh ):
    # Solve the Momentum equation

    UEqn = man.fvm.ddt( rho, U ) + man.fvm.div( phi, U ) + man.fvVectorMatrix( turbulence.divDevRhoReff( U() ), man.Deps( turbulence, U ) )
    UEqn.relax()
    
    if pimple.momentumPredictor():
        ref.solve( UEqn == man.fvc.reconstruct( ( - ghf * man.fvc.snGrad( rho ) - man.fvc.snGrad( p_rgh ) )
                                                * man.surfaceScalarField( mesh.magSf(), man.Deps( mesh ) ) ) )
  
    return UEqn


#---------------------------------------------------------------------------
def fun_hEqn( thermo, rho, p, h, phi, turbulence, DpDt ):
     hEqn = ref.fvm.ddt( rho, h ) + ref.fvm.div( phi, h )- ref.fvm.laplacian( turbulence.alphaEff(), h ) == DpDt

     hEqn.relax()
     hEqn.solve()

     thermo.correct()
     pass


#---------------------------------------------------------------------------
def fun_pEqn( mesh, runTime, pimple, thermo, rho, p, h, psi, U, phi, turbulence, gh, ghf, p_rgh, UEqn, DpDt, cumulativeContErr, corr ):
    rho <<= thermo.rho()
  
    # Thermodynamic density needs to be updated by psi*d(p) after the
    # pressure solution - done in 2 parts. Part 1:
    tmp = thermo.rho()
    tmp -= psi() * p_rgh()
  
    rAU = 1.0 / UEqn.A()
    rhorAUf = ref.surfaceScalarField( ref.word( "(rho*(1|A(U)))" ), ref.fvc.interpolate( rho * rAU ) )
  
    U <<= rAU * UEqn.H()

    phi <<= ref.fvc.interpolate( rho ) * ( ( ref.fvc.interpolate( U ) & mesh.Sf() ) + ref.fvc.ddtPhiCorr( rAU, rho, U, phi ) )

    buoyancyPhi = -rhorAUf * ghf * ref.fvc.snGrad( rho ) * mesh.magSf()
  
    phi += buoyancyPhi

    p_rghDDtEqn = ref.fvc.ddt( rho ) + psi * ref.correction( ref.fvm.ddt( p_rgh ) ) + ref.fvc.div( phi )

    for nonOrth in range( pimple.nNonOrthCorr() + 1 ):
        p_rghEqn = p_rghDDtEqn - ref.fvm.laplacian( rhorAUf, p_rgh )
        
        p_rghEqn.solve( mesh.solver( p_rgh.select( pimple.finalInnerIter( corr, nonOrth ) ) ) )
        
        if nonOrth == pimple.nNonOrthCorr():
            # Calculate the conservative fluxes
            phi += p_rghEqn.flux()
            # Explicitly relax pressure for momentum corrector
            p_rgh.relax()
            
            # Correct the momentum source with the pressure gradient flux
            # calculated from the relaxed pressure
            U += rAU * ref.fvc.reconstruct( ( buoyancyPhi + p_rghEqn.flux() ) / rhorAUf )
            U.correctBoundaryConditions()
            pass
        pass
    
    p <<= p_rgh + rho*gh

    # Second part of thermodynamic density update
    tmp = thermo.rho()
    tmp += psi * p_rgh

    DpDt <<= ref.fvc.DDt( ref.surfaceScalarField( ref.word( "phiU" ), phi() / ref.fvc.interpolate( rho ) ), p ) # mixed calculations

    ref.rhoEqn( rho, phi )
    cumulativeContErr = ref.compressibleContinuityErrs( rho(), thermo, cumulativeContErr ) #mixed calculations
    
    return cumulativeContErr
  

#---------------------------------------------------------------------------
def main_standalone( argc, argv ):
    args = ref.setRootCase( argc, argv )

    runTime = man.createTime( args )

    mesh = man.createMesh( runTime )
    
    g = ref.readGravitationalAcceleration( runTime, mesh )
    
    thermo, p, rho, h, psi, U, phi, turbulence, gh, ghf, p_rgh, DpDt = createFields( runTime, mesh, g )

    cumulativeContErr = ref.initContinuityErrs()
  
    adjustTimeStep, maxCo, maxDeltaT = ref.readTimeControls( runTime )
  
    CoNum, meanCoNum = ref.compressibleCourantNo( mesh, phi, rho, runTime )
    
    runTime = ref.setInitialDeltaT( runTime, adjustTimeStep, maxCo, maxDeltaT, CoNum )
  
    pimple = man.pimpleControl( mesh ) 


    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    ref.ext_Info()<< "\nStarting time loop\n" << ref.nl

    while runTime.run():
    
        adjustTimeStep, maxCo, maxDeltaT = ref.readTimeControls( runTime )
        CoNum, meanCoNum = ref.compressibleCourantNo( mesh, phi, rho, runTime )

        runTime = ref.setDeltaT( runTime, adjustTimeStep, maxCo, maxDeltaT, CoNum )
        runTime.increment()

        ref.ext_Info()<< "Time = " << runTime.timeName() << ref.nl << ref.nl

        ref.rhoEqn( rho, phi )

        # --- Pressure-velocity PIMPLE corrector loop
        pimple.start()
        while pimple.loop():
            if pimple.nOuterCorr() != 1:
                p_rgh.storePrevIter()
                pass
      
            UEqn = fun_Ueqn( pimple, mesh, rho, U, phi, turbulence, ghf, p_rgh )
            fun_hEqn( thermo, rho, p, h, phi, turbulence, DpDt )
      
            # --- PISO loop
            for corr in range( pimple.nCorr() ):
                cumulativeContErr = fun_pEqn( mesh, runTime, pimple, thermo, rho, p, h, psi, U, phi, 
                                              turbulence, gh, ghf, p_rgh, UEqn, DpDt, cumulativeContErr, corr )
                pass
            if pimple.turbCorr():
                turbulence.correct()
                pass
           
            pimple.increment()
            pass
        rho <<= thermo.rho()

        runTime.write()

        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" \
                       << "  ClockTime = " << runTime.elapsedClockTime() << " s" \
                       << ref.nl << ref.nl
    ref.ext_Info()<< "End\n" << ref.nl
    
    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
import sys, os
from Foam import FOAM_REF_VERSION
if FOAM_REF_VERSION( ">=", "020000" ):
   if __name__ == "__main__" :
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass
else:
   from Foam.OpenFOAM import ext_Info
   ext_Info()<< "\nTo use this solver, It is necessary to SWIG OpenFoam2.0.0 or higher \n "


#--------------------------------------------------------------------------------------


