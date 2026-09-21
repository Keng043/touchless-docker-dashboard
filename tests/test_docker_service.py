import pytest
from docker_service import DockerContainer, MockDockerService, RealDockerService

def test_mock_lists_expected_containers():
    service = MockDockerService()
    containers = service.list_containers()
    assert len(containers) == 5
    assert all(isinstance(c, DockerContainer) for c in containers)
    assert all(c.memory_limit_mb > 0 for c in containers)
    assert all(0 <= c.cpu_percentage <= 100 for c in containers)

def test_lifecycle_uses_container_id():
    service = MockDockerService()
    target = next(c for c in service.list_containers() if c.status == 'exited')
    assert service.start_container(target.id) is True
    assert next(c for c in service.list_containers() if c.id == target.id).status == 'running'
    assert service.stop_container(target.id) is True
    assert next(c for c in service.list_containers() if c.id == target.id).status == 'exited'

def test_unknown_container_is_rejected():
    service = MockDockerService()
    assert service.start_container('not-a-container') is False
    assert service.stop_container('not-a-container') is False
    assert service.restart_container('not-a-container') is False

def test_cpu_percentage_calculation():
    service = object.__new__(RealDockerService)
    stats = {'cpu_stats': {'cpu_usage': {'total_usage': 300}, 'system_cpu_usage': 1300, 'online_cpus': 2}, 'precpu_stats': {'cpu_usage': {'total_usage': 100}, 'system_cpu_usage': 900}}
    assert service._calculate_cpu_percentage(stats) == 100.0

@pytest.mark.parametrize('stats', [{}, {'cpu_stats': None}, {'precpu_stats': None}])
def test_cpu_calculation_handles_malformed_stats(stats):
    service = object.__new__(RealDockerService)
    assert service._calculate_cpu_percentage(stats) == 0.0
