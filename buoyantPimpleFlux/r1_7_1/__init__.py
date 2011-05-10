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
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey Petrov
##


#---------------------------------------------------------------------------
def create_fields( runTime, mesh, g ):
    from Foam.OpenFOAM import ext_Info, nl
    ext_Info() << "Reading thermophysical properties\n" << nl
    
    from Foam.thermophysicalModels import basicRhoThermo
    thermo = basicRhoThermo.New( mesh )
    
    from Foam.OpenFOAM import IOobject, word, fileName
    from Foam.finiteVolume import volScalarField
    
    rho = volScalarField( IOobject( word( "rho" ),
                                    fileName( runTime.timeName() ),
                                    mesh,
                                    IOobject.NO_READ,
                                    IOobject.NO_WRITE ),
                          thermo.rho() )

    p = thermo.p()
    h = thermo.h()
    psi = thermo.psi()

    ext_Info() << "Reading field U\n" << nl
    from Foam.finiteVolume import volVectorField
    U = volVectorField( IOobject( word( "U" ),
                                  fileName( runTime.timeName() ),
                                  mesh,
                                  IOobject.MUST_READ,
                                  IOobject.AUTO_WRITE ),
                        mesh )

    from Foam.finiteVolume.cfdTools.compressible import compressibleCreatePhi
    phi = compressibleCreatePhi( runTime, mesh, rho, U )

    ext_Info() << "Creating turbulence model\n" << nl
    from Foam import compressible
    turbulence = compressible.turbulenceModel.New( rho, U, phi, thermo() )

    ext_Info() << "Calculating field g.h\n" << nl
    gh = volScalarField( word( "gh" ), g & mesh.C() )
    
    from Foam.finiteVolume import surfaceScalarField
    ghf = surfaceScalarField( word( "ghf" ), g & mesh.Cf() )

    ext_Info() << "Reading field p_rgh\n" << nl
    p_rgh = volScalarField( IOobject( word( "p_rgh" ),
                                      fileName( runTime.timeName() ),
                                      mesh,
                                      IOobject.MUST_READ,
                                      IOobject.AUTO_WRITE ),
                            mesh )

    # Force p_rgh to be consistent with p
    p_rgh.ext_assign( p - rho * gh )

    ext_Info() << "Creating field DpDt\n" << nl
    from Foam import fvc
    DpDt = volScalarField( word( "DpDt" ), fvc.DDt( surfaceScalarField( word( "phiU" ), phi / fvc.interpolate( rho ) ), p ) )

    return thermo, p, rho, h, psi, U, phi, turbulence, gh, ghf, p_rgh, DpDt


#---------------------------------------------------------------------------
def fun_UEqn( mesh, rho, phi, U, p_rgh, ghf, turbulence, finalIter, momentumPredictor ):
    
    from Foam import fvm, fvc    
    # Solve the Momentum equation
    UEqn = fvm.ddt( rho, U ) + fvm.div( phi, U ) + turbulence.divDevRhoReff( U )

    UEqn.relax()

    if momentumPredictor:
       from Foam.finiteVolume import solve
       solve( UEqn == fvc.reconstruct( ( - ghf * fvc.snGrad( rho ) - fvc.snGrad( p_rgh ) ) * mesh.magSf() ), 
              mesh.solver( U.select( finalIter) ) );

    return UEqn


#---------------------------------------------------------------------------
def fun_hEqn( mesh, rho, h, phi, DpDt, thermo, turbulence, finalIter ):
    from Foam import fvm
    hEqn = fvm.ddt( rho, h ) + fvm.div( phi, h ) - fvm.laplacian( turbulence.alphaEff(), h ) == DpDt

    hEqn.relax()
    hEqn.solve( mesh.solver( h.select( finalIter ) ) )

    thermo.correct()
    
    pass


#---------------------------------------------------------------------------
def fun_pEqn( mesh, p, rho, psi, p_rgh, U, phi, ghf, gh, DpDt, UEqn, thermo, nNonOrthCorr, corr, nCorr, finalIter, cumulativeContErr ):
    
    rho.ext_assign( thermo.rho() )

    # Thermodynamic density needs to be updated by psi*d(p) after the
    # pressure solution - done in 2 parts. Part 1:
    thermo.rho().ext_assign( thermo.rho() - psi * p_rgh )

    rUA = 1.0 / UEqn.A()
    from Foam.finiteVolume import surfaceScalarField
    from Foam.OpenFOAM import word
    from Foam import fvc
    rhorUAf = surfaceScalarField( word( "(rho*(1|A(U)))" ), fvc.interpolate( rho * rUA ) )

    U.ext_assign( rUA*UEqn.H() )

    phi.ext_assign( fvc.interpolate( rho ) * ( ( fvc.interpolate( U ) & mesh.Sf() ) + fvc.ddtPhiCorr( rUA, rho, U, phi ) ) )

    buoyancyPhi = -rhorUAf * ghf * fvc.snGrad( rho ) * mesh.magSf()
    phi.ext_assign( phi + buoyancyPhi )
    
    from Foam import fvm
    from Foam.finiteVolume import correction
    for nonOrth in range( nNonOrthCorr +1 ):
        p_rghEqn = fvc.ddt( rho ) + psi * correction( fvm.ddt( p_rgh ) ) + fvc.div( phi ) - fvm.laplacian( rhorUAf, p_rgh )

        p_rghEqn.solve( mesh.solver( p_rgh.select( ( finalIter and corr == nCorr-1 and nonOrth == nNonOrthCorr ) ) ) )

        if nonOrth == nNonOrthCorr:
            # Calculate the conservative fluxes
            phi.ext_assign( phi + p_rghEqn.flux() )

            # Explicitly relax pressure for momentum corrector
            p_rgh.relax()

            # Correct the momentum source with the pressure gradient flux
            # calculated from the relaxed pressure
            U.ext_assign( U + rUA * fvc.reconstruct( ( buoyancyPhi + p_rghEqn.flux() ) / rhorUAf ) )
            U.correctBoundaryConditions()
            pass

    p.ext_assign( p_rgh + rho * gh )

    # Second part of thermodynamic density update
    thermo.rho().ext_assign( thermo.rho() + psi * p_rgh )

    DpDt.ext_assign( fvc.DDt( surfaceScalarField( word( "phiU" ), phi / fvc.interpolate( rho ) ), p ) )

    from Foam.finiteVolume.cfdTools.compressible import rhoEqn  
    rhoEqn( rho, phi )
    
    from Foam.finiteVolume.cfdTools.compressible import compressibleContinuityErrs
    cumulativeContErr = compressibleContinuityErrs( rho, thermo, cumulativeContErr )

    return cumulativeContErr


#---------------------------------------------------------------------------
def main_standalone( argc, argv ):

    from Foam.OpenFOAM.include import setRootCase
    args = setRootCase( argc, argv )

    from Foam.OpenFOAM.include import createTime
    runTime = createTime( args )

    from Foam.OpenFOAM.include import createMesh
    mesh = createMesh( runTime )
    
    from Foam.finiteVolume.cfdTools.general.include import readGravitationalAcceleration
    g = readGravitationalAcceleration( runTime, mesh)

    thermo, p, rho, h, psi, U, phi, turbulence, gh, ghf, p_rgh, DpDt = create_fields( runTime, mesh, g )
    
    from Foam.finiteVolume.cfdTools.general.include import initContinuityErrs
    cumulativeContErr = initContinuityErrs()
    
    from Foam.finiteVolume.cfdTools.general.include import readTimeControls
    adjustTimeStep, maxCo, maxDeltaT = readTimeControls( runTime )
    
    from Foam.finiteVolume.cfdTools.compressible import compressibleCourantNo
    CoNum, meanCoNum = compressibleCourantNo( mesh, phi, rho, runTime )
    
    from Foam.finiteVolume.cfdTools.general.include import setInitialDeltaT
    runTime = setInitialDeltaT( runTime, adjustTimeStep, maxCo, maxDeltaT, CoNum )
    
    from Foam.OpenFOAM import ext_Info, nl
    ext_Info()<< "\nStarting time loop\n" << nl
    
    while runTime.run() :
        
        from Foam.finiteVolume.cfdTools.general.include import readTimeControls
        adjustTimeStep, maxCo, maxDeltaT = readTimeControls( runTime )
    
        from Foam.finiteVolume.cfdTools.general.include import readPIMPLEControls
        pimple, nOuterCorr, nCorr, nNonOrthCorr, momentumPredictor, transonic = readPIMPLEControls( mesh )
        
        from Foam.finiteVolume.cfdTools.compressible import compressibleCourantNo
        CoNum, meanCoNum = compressibleCourantNo( mesh, phi, rho, runTime )
        
        from Foam.finiteVolume.cfdTools.general.include import setDeltaT
        runTime = setDeltaT( runTime, adjustTimeStep, maxCo, maxDeltaT, CoNum )
        
        runTime.increment()
        ext_Info() << "Time = " << runTime.timeName() << nl << nl
        
        from Foam.finiteVolume.cfdTools.compressible import rhoEqn  
        rhoEqn( rho, phi )
        
        # --- Pressure-velocity PIMPLE corrector loop
        for oCorr in range( nOuterCorr ):
            finalIter = oCorr == ( nOuterCorr-1 )

            if nOuterCorr != 1:
                p_rgh.storePrevIter()
                pass

            UEqn = fun_UEqn( mesh, rho, phi, U, p_rgh, ghf, turbulence, finalIter, momentumPredictor )
            fun_hEqn( mesh, rho, h, phi, DpDt, thermo, turbulence, finalIter )

            # --- PISO loop
            for corr in range( nCorr ):
                cumulativeContErr = fun_pEqn( mesh, p, rho, psi, p_rgh, U, phi, ghf, gh, DpDt, UEqn, \
                                              thermo, nNonOrthCorr, corr, nCorr, finalIter, cumulativeContErr )
                pass

            turbulence.correct()

            rho.ext_assign( thermo.rho() )
            pass

        runTime.write()
        
        ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" << "  ClockTime = " << runTime.elapsedClockTime() << " s" << nl << nl
        
        pass
    
    ext_Info() << "End\n"

    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
import sys, os
from Foam import FOAM_REF_VERSION
if FOAM_REF_VERSION( ">=", "010701" ):
   if __name__ == "__main__" :
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass
else:
   from Foam.OpenFOAM import ext_Info
   ext_Info()<< "\nTo use this solver, It is necessary to SWIG OpenFoam1.7.1 or higher \n "


#--------------------------------------------------------------------------------------
