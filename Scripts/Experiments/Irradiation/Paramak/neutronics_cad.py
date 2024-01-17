#!/bin/bash
import shutil
import os
import sys
import paramak
sys.dont_write_bytecode=True
import paramak


def cadparamak(CALC_DIR,Name,copper_interlayer_radius,copper_interlayer_thickness,Warmour_thickness,
               Warmour_width,Warmour_height_lower,Warmour_height_upper,pipe_radius,pipe_thickness,
               pipe_length,pipe_protrusion,cad_output='dagmc.h5m',Headless=False,_Name=None):
	
	# Creation of CAD model for neutronics simulation 

	# Create outer circle of copper interlayer with the thickness of tungsten monoblock
	rotated_circle = paramak.ExtrudeCircleShape(points=[(0, 0),],radius=copper_interlayer_radius,
	                                            distance=Warmour_thickness, workplane='XZ', name='part0.stl',)

        # Create tungsten armour block
	tungsten_armour = paramak.ExtrudeStraightShape(points=[(-Warmour_width/2, -Warmour_height_lower),
								 (Warmour_width/2, -Warmour_height_lower),
								 (Warmour_width/2, Warmour_height_upper),
								 (-Warmour_width/2, Warmour_height_upper)],
		      							distance=Warmour_thickness,
									color=(0.5, 0.5, 0.5),
									cut=rotated_circle,
									name="tungsten_armour",)

	# Create copper interlayer block
	copper_interlayer = paramak.RotateStraightShape(points=[(pipe_radius+pipe_thickness, -Warmour_width/2),
							(pipe_radius+pipe_thickness+copper_interlayer_thickness, -Warmour_width/2),
                                                      (pipe_radius+pipe_thickness+copper_interlayer_thickness,  Warmour_width/2),
							(pipe_radius+pipe_thickness, Warmour_width/2)],
									color=(0.5, 0, 0),
									workplane="XY",
									rotation_angle=360,
									name="copper_interlayer",)

	# Create CuCrZr coolant pipe
	CuCrZr_pipe = paramak.RotateStraightShape(points=[(pipe_radius, (-pipe_length/2)-pipe_protrusion),
							    (pipe_radius+pipe_thickness, (-pipe_length/2)-pipe_protrusion),
							    (pipe_radius+pipe_thickness, (pipe_length/2)+pipe_protrusion),
							    (pipe_radius, (pipe_length/2)+pipe_protrusion)],
									color=(0, 0, 0.5),
									workplane="XY",
									rotation_angle=360,
									name="CuCrZr_pipe")

	# Combine tungsten armour, copper interlayer, CuCrZr coolant pipe
	
	fusion_reactor_comp = paramak.Reactor([tungsten_armour, copper_interlayer, CuCrZr_pipe])

       # Create dagmc format of cad model for neutronics simulation
       # Mesh size for dagmc is selected

	fusion_reactor_comp.export_dagmc_h5m(filename=f"{CALC_DIR}/{cad_output}", min_mesh_size=.001, 
	                                     max_mesh_size=.1)

