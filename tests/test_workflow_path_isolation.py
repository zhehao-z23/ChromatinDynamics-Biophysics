from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location('physics_workflow', Path(__file__).resolve().parents[1] / 'scripts/run_pipeline.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@pytest.mark.parametrize('cache,output,snapshot', [
    ('data/cache', 'data/cache', None),
    ('data/cache', 'data/cache/results', None),
    ('data/cache', 'data', None),
    ('data/cache', 'results', 'data/cache/snapshot'),
    ('data/cache', 'data/snapshot/results', 'data/snapshot'),
])
def test_frozen_input_cannot_receive_generated_output(cache, output, snapshot):
    with pytest.raises(ValueError, match='non-overlapping'):
        workflow.validate_path_separation(Path(cache), Path(output), Path(snapshot) if snapshot else None)


def test_independent_sibling_inputs_and_results_are_allowed():
    workflow.validate_path_separation(Path('data/cache'), Path('runs/new'), Path('data/snapshot'))
