from symmstate.abinit import SmodesProcessor, AbinitUnitCell

smodes_object = SmodesProcessor("input_file.abi", "Smodes input", "GM4-", disp_mag=0.01)
smodes_object = AbinitUnitCell("input_file.abi")

smodes_unstable_phonons = smodes_object.symmadapt()
