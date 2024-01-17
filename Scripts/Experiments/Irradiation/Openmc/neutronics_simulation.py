#!/bin/bash\n
import math
import os
import re
import openmc
import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()
import numpy as N
import openmc_dagmc_wrapper as odw
import openmc_data_downloader as odd
import openmc_tally_unit_converter as opp
from openmc_mesh_tally_to_vtk.utils import _get_mesh_from_tally
from openmc_mesh_tally_to_vtk.utils import _find_coords_of_mesh
from openmc_mesh_tally_to_vtk.utils import _write_vtk


def simulation_Openmc(CALC_DIR,Name,Warmour_thickness, Warmour_width,Warmour_height_lower,Warmour_height_upper,
                      pipe_protrusion,width,height,thickness,source_location,heat_output='heating_openmc_mesh.vtk',
                      cad_input='dagmc.h5m',damage_energy_output='damage_energy_openmc_mesh.vtk', Headless=False,_Name=None):
               
	# Assign material properties for tungsten armour, copper interlayer, CuCrZr coolant pipe
	w = openmc.Material(name="w")
	w.add_element("W", 1)
	w.set_density("g/cc", 19.3)
	cucrzr = openmc.Material(name="cu")
	cucrzr.add_element("Cr", 0.012)
	cucrzr.add_element("Zr", 0.0007)
	cucrzr.set_density("g/cc", 8.96)
	copper = openmc.Material(name="copper")
	copper.add_element("copper", 1)
	copper.set_density("g/cc", 8.96)
	# Create a sphere around the geometry to fill vacuum
	vac_surf = openmc.Sphere(r=10000, surface_id=9999, boundary_type="vacuum")
	# specifies the region  inside the vacuum surfaces
	region = -vac_surf
	# Open dagmc model in OpenMC
	dag_univ = openmc.DAGMCUniverse("dagmc.h5m")
	containing_cell = openmc.Cell(cell_id=9999,region=region, fill=dag_univ)
	geometry = openmc.Geometry(root=[containing_cell])
	# Assign neutron source
	my_source = openmc.Source()
	# sets the location of the source x,y,z
	my_source.space = openmc.stats.Point((0, 0, source_location))
	# sets the direction to isotropic
	my_source.angle = openmc.stats.Isotropic()
	# sets the energy distribution to 100% 14MeV neutrons
	my_source.energy = openmc.stats.Discrete([14e6], [1])
	# this links the material tags in the dagmc h5m file with materials
	# these materials are input as strings so they will be looked up in the neutronics material maker package
	materials = odw.Materials(
				  h5m_filename=f"{CALC_DIR}/{cad_input}",
				  correspondence_dict={ "CuCrZr_pipe": cucrzr,
				  	                "tungsten_armour":w,
							 "copper_interlayer": copper,
 							},)
	# Set the number of neutrons emitted from the source in batches
	settings = openmc.Settings()
	settings.batches = 500
	settings.particles = 100000
	settings.inactive = 0
	settings.run_mode = "fixed source"
	settings.source = my_source
	# adds a tally to record the heat deposited and damage energy in entire geometry
	cell_tally = openmc.Tally(name="heating")
	cell_tally.scores = ["heating"]
	cell_tally1 = openmc.Tally(name="damage-energy")
	cell_tally1.scores = ["damage-energy"]
	# creates a mesh that covers the geometry\n'
	mesh = openmc.RegularMesh()
	mesh.dimension = [width, thickness, height]
	mesh.lower_left = [-Warmour_width/2, (-Warmour_thickness/2)-pipe_protrusion, -Warmour_height_lower] 
	mesh.upper_right = [Warmour_width/2, (Warmour_thickness/2)+pipe_protrusion, Warmour_height_upper]
	# makes a mesh tally using the previously created mesh and records heating and damage energy on the mesh
	mesh_tally = openmc.Tally(name="heating_on_mesh")
	mesh_filter = openmc.MeshFilter(mesh)
	mesh_tally.filters = [mesh_filter]
	mesh_tally.scores = ["heating"]
	mesh_tally1 = openmc.Tally(name="damage-energy_on_mesh")
	mesh_filter1 = openmc.MeshFilter(mesh)
	mesh_tally1.filters = [mesh_filter1]
	mesh_tally1.scores = ["damage-energy"]
	# groups the two tallies
	tallies = openmc.Tallies([mesh_tally,mesh_tally1])
	# Choose the nuclear librarary
	odd.just_in_time_library_generator(
					   destination=f"{CALC_DIR}",
					   libraries='ENDFB-7.1-NNDC',
					   materials=materials
					   )
	# builds the openmc model
	my_model = openmc.Model(
	materials=materials, geometry=geometry, settings=settings, tallies=tallies
				)
	# starts the simulation
	my_model.run(cwd=f"{CALC_DIR}")
	from openmc_mesh_tally_to_vtk import write_mesh_tally_to_vtk
	# open the results file using statepoint file (sp)
	sp = openmc.StatePoint(f"{CALC_DIR}/statepoint.500.h5")
	my_tally = sp.get_tally(name="heating_on_mesh")
	damage_mesh_tally = sp.get_tally(name="damage-energy_on_mesh")
	# Results saved in VTK format for damage-energy
	write_mesh_tally_to_vtk(
				tally=damage_mesh_tally,
				filename=f"{CALC_DIR}/{damage_energy_output}")
	# this finds the number of neutrons emitted per second by a 1.5e5W fusion DT plasma
	source_strength = opp.find_source_strength(
	fusion_energy_per_second_or_per_pulse=1.5e5, reactants="DT")
	# Convert the heating tallied to watts/meter**3 
	result,error = opp.process_tally(
					  tally=my_tally,
					  source_strength=source_strength,  # number of neutrons per second emitted by the source
					  required_units = "watts / meter **3"
					)
	tally3 = np.array(result)
	print((result))
	mesh1 = _get_mesh_from_tally(my_tally)
	xs, ys, zs = _find_coords_of_mesh(mesh1)
	# Results saved in VTK format for heating
	output_filename = _write_vtk(
				      xs=xs,
				      ys=ys,
				      zs=zs,
				      tally_data=result,
				      filename=f"{CALC_DIR}/{heat_output}")
