# tests/integration/test_perturbations.py
def test_perturbation_workflow(tmp_path, sample_unit_cell):
    perts = Perturbations(
        base_cell=sample_unit_cell,
        perturbation=np.random.rand(4,3)*0.01,
        num_datapoints=5
    )
    
    perts.generate_perturbations()
    assert len(perts.perturbed_objects) == 5
    assert len(perts.list_amps) == 5
    
    # Test amplitude stepping
    expected_amps = np.linspace(0, 0.5, 5)
    assert np.allclose(perts.list_amps, expected_amps)