import json
import os
from fireworks import LaunchPad
from atomate2.vasp.flows.core import RelaxMaker,HSEOpticsMaker
from atomate2.vasp.powerups import add_metadata_to_flow
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core import Structure
from jobflow import Flow
from atomate2.vasp.powerups import update_user_incar_settings

def generate_material_workflows(json_file):
    """
    Generate and submit workflows for materials listed in a JSON file.
    
    Parameters:
    json_file (str): Path to the JSON file containing material IDs
    """
    # Load the materials dictionary from JSON
    with open(json_file, 'r') as f:
        materials_dict = json.load(f)
    
    # Initialize LaunchPad
    lpad = LaunchPad.auto_load()
    
    # Process each material
    for material_name, mp_id in materials_dict.items():
        try:
            # Construct CIF filename (from CIFS in directory ./cifs/)
            cif_filename = f"cifs/{material_name}.cif"
            
            # Check if CIF file exists
            if not os.path.exists(cif_filename):
                print(f"Warning: CIF file {cif_filename} not found. Skipping {material_name}")
                continue
            
            # Read structure from CIF
            material_structure = Structure.from_file(cif_filename)
            
            # Create hybrid optics workflow using relaxed structure
            optics_flow = HSEOpticsMaker().make(structure=material_structure)
            
            # Create full workflow - adjust NCORE as req. by hardware
            full_flow =  update_user_incar_settings(optics_flow,{"NCORE":10})            
            
            # Add metadata to the flow
            full_flow = add_metadata_to_flow(
                flow=full_flow,
                additional_fields={"mp_id": mp_id, "material_name": material_name}
            )
            
            # Convert flow to FireWorks workflow
            wf = flow_to_workflow(full_flow)
            
            # Submit workflow to LaunchPad
            lpad.add_wf(wf)
            
            print(f"Workflow submitted successfully for {material_name} (MP ID: {mp_id})")
        
        except Exception as e:
            print(f"Error processing {material_name}: {str(e)}")

if __name__ == "__main__":
    generate_material_workflows("key_materials.json")
