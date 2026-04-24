from mp_api.client import MPRester

# Use your Materials Project API key
with MPRester("ld5HdyuSrVhvlNZKJIPy2iM58UCKuX2V") as mpr:
    # get_structure_by_material_id returns the primitive cell by default
    primitive_struct = mpr.get_structure_by_material_id("mp-66")
    
    # Export to CIF for use in other software
    primitive_struct.to(filename="C_primitive.cif")