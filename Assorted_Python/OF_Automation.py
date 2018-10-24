import re
import os
import sys
import shutil
import fileinput

Case = []
a = 0
ii = 0
while (a != 1) :
	Sim_Start = raw_input('Start a new Case (y/n):')
	if Sim_Start == 'y' :
			# Copy shit from Reference Folder
			Case_num = raw_input('Enter Case number (Integers only>0):')
			Dir_path = raw_input('Enter the Full directory Path(with the / or \ at end) :')
			Case.append(Case_num)
			Case_zero = Dir_path+'Case0'
			Case_zero_zero = Dir_path+'Case0/0'
			Case_zero_system = Dir_path+'Case0/system'
			Case_zero_constant = Dir_path+'Case0/constant'
			new_path = Dir_path+'Case'+Case_num
			Case_new_zero = new_path+'/0'
			Case_new_system = new_path+'/system'
			Case_new_constant = new_path+'/constant'

			if not os.path.exists(new_path) :
				os.makedirs(new_path)
			else :
				OvrWrite_path = raw_input('Path already exists. Do you want to overwrite it? (y/n) :')
				if OvrWrite_path == 'y' and Case_num != '0':
					shutil.rmtree(new_path)
					os.makedirs(new_path)
				else :
					sys.exit("Program will exit now")


			shutil.copytree(Case_zero_zero, Case_new_zero)
			shutil.copytree(Case_zero_system, Case_new_system)
			shutil.copytree(Case_zero_constant, Case_new_constant)
			shutil.copy(Case_zero + '/PostPro.pvsm', new_path)


			# Start with controlDict - 0 Folder
			print('Adaptive Time Stepping\n\n')
			maxCo = raw_input('Enter max Courant Number (In Format 5.0, Range[5-50], Reference 5.0):')
			fileToSearch = Case_new_system+'/controlDict'
			textToSearch = 'maxCo 5.0;'
			textToReplace = 'maxCo ' + maxCo + ';'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))

			# Start with fvSchemes - 0 Folder
			print('\n\nSolver Schemes\n\n')
			laplacian = raw_input('Enter Gauss Linear Limited Laplacian Scheme (In Format 0.3333, Range[0.1-0.5], Reference 0.3333):')
			snGradSchmes = raw_input('Enter Limited snGradScheme (In Format 0.3333, Range[0.1-0.5], Reference 0.3333):')
			fileToSearch = Case_new_system+'/fvSchemes'

			textToSearch = '    default         Gauss linear limited 0.333;'
			textToSearch1 = '    laplacian(nu,U) Gauss linear limited 0.333;'
			textToSearch2 = '    laplacian((1|A(U)),p) Gauss linear limited 0.333;'
			textToSearch3 = '    default         limited 0.333;'		
			textToReplace = '    default         Gauss linear limited ' + laplacian + ';'
			textToReplace1 = '    laplacian(nu,U) Gauss linear limited ' + laplacian + ';'
			textToReplace2 = '    laplacian((1|A(U)),p) Gauss linear limited ' + laplacian + ';'
			textToReplace3 = '    default         limited ' + snGradSchmes + ';'

			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch1, textToReplace1))
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch2, textToReplace2))
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch3, textToReplace3))

			# Start with fvSolution - 0 Folder
			print('\n\n')
			solversel = raw_input('Enter solver [PCG for pre-conj Grad or DIRECT for smoothSolver]')
			if solversel == 'PCG' :
				os.remove(Case_new_system+'/fvSolution_Smooth')
				os.rename(Case_new_system+'/fvSolution_PCG',Case_new_system+'/fvSolution2')
			else :
				os.remove(Case_new_system+'/fvSolution_PCG')
				os.rename(Case_new_system+'/fvSolution_Smooth',Case_new_system+'/fvSolution2')
			solvresidual = raw_input('Enter Velocity,k,epsilon,etc Solver Residual (In Format 1e-08, Range[1e-04 - 1e-08], Reference 1e-08):')
			fileToSearch = Case_new_system+'/fvSolution2'
			textToSearch = '        tolerance       1e-08;'
			textToReplace = '        tolerance       ' + solvresidual + ';'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))
			filehead = Case_new_system+'/fvSolution_Head'
			filenames = [filehead, fileToSearch]
			fvSolution = Case_new_system+'/fvSolution'
			with open(fvSolution, 'w') as outfile:
				for fname in filenames:
					with open(fname) as infile:
						for line in infile:
							outfile.write(line)

			os.remove(filehead)
			os.remove(fileToSearch)

			# Start with fvSolution - k,eps,U Folder
			print('\n\n')
			print('Boundary Conditions\n\n')
			inletVel = raw_input('Enter Inlet Velocity (In Format -4.3, Range[-4 - -5], Reference -4.3):')
			TurbLenScale = raw_input('Enter Turbulence Length Scale in % (In Format - 7, Range [3-10], Reference 7)')
			CharLength = (2*0.1*0.01)/(0.1+0.01)
			ReysNo = 1000*abs(float(inletVel))*CharLength/(1.002e-3)
			TurbIntens = 0.16*((ReysNo)**(-0.125))
			TurbLen = float(TurbLenScale)*CharLength
			kTurb = 1.5*(float(inletVel)*TurbIntens)**2
			epsTurb = (0.1643*kTurb**1.5)/TurbLen

			fileToSearch = Case_new_zero+'/U'
			textToSearch = '        value           uniform (0 -4.3 0);'
			textToReplace = '        value           uniform (0 ' + inletVel + ' 0);'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))

			fileToSearch = Case_new_zero+'/k'
			textToSearch = 'internalField   uniform 0.025;'
			textToReplace = 'internalField   uniform ' + str(kTurb) + ';'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))


			fileToSearch = Case_new_zero+'/epsilon'
			textToSearch = 'internalField   uniform 0.065;'
			textToReplace = 'internalField   uniform ' + str(epsTurb) + ';'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))

			fileToSearch = new_path+'/PostPro.pvsm'
			textToSearch = '        <Element index="0" value="/home/aranjan/Documents/System_X/OPENFOAM/Steady_State/kEpsilon/kEpsilon.OpenFOAM"/>'
			textToReplace = '        <Element index="0" value="' + new_path + '/Case' + Case_num + '.OpenFOAM' +'"/>'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))			
			textToSearch = '      <Item id="3530" name="kEpsilon.OpenFOAM"/>'
			textToReplace = '      <Item id="3530" name="' + 'Case' + Case_num + '.OpenFOAM' +'"/>'
			for line in fileinput.input(fileToSearch, inplace=True):
				sys.stdout.write(line.replace(textToSearch, textToReplace))


			# Export All Parameters
			ExportedFile = new_path + 'ExportedFile.txt'
			f = open(ExportedFile, 'w')
			f.write ("Inputs")
			f.write ("\n\nAdaptive Time Stepping\n")
			line = "Max Courant Number =" + maxCo + "\n"
			f.write(line)
			f.write ("\n\nSolver Schemes\n")
			line = "Laplacian Gauss Linear Limited =" + laplacian + "\n"
			f.write(line)
			line = "snGradScheme Linear =" + snGradSchmes + "\n"
			f.write(line)
			line = "Implemented Solver =" + solversel + "\n"
			f.write(line)
			line = "Solver Tolerance =" + solvresidual + "\n"
			f.write(line)
			f.write ("\n\nBoundary Conditions\n")
			line = "Inlet Velocity =" + inletVel + "\n"
			f.write(line)
			line = "Turbulence Length Scale =" + TurbLenScale + "\n"
			f.write(line)
			f.write ("\n\n\nComputed Values\n")
			line = "Reynolds' Number =" + str(ReysNo) + "\n"
			f.write(line)
			line = "Turbulence Energy =" + str(kTurb) + "\n"
			f.write(line)
			line = "Turbulence Dissipation =" + str(epsTurb) + "\n"
			f.write(line)
			line = "################# Checked OK##################"
			f.write(line)
			
			query_queue = raw_input('Do you want to queue the developed deck? (y/n):')
			if query_queue == 'n' :
				a = 1
				ii = ii + 1
			else :
				ii = ii + 1
				

jj=0
if ii > 1 : 
	while (jj < ii) :
		new_path = Dir_path+'Case'+Case[jj]
		os.chdir (new_path)
		os.system ("simpleFoam")
		jj = jj + 1
	

		