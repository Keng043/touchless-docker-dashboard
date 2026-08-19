"""
Module: docker_service
Project: Touchless Docker Dashboard
Author: Python Software Architect & DevOps Engineer
Description: SOLID-compliant, decoupled Docker management module.
             It employs the Interface Segregation and Dependency Inversion Principles
             to provide a robust API with automatic Mock Fallback when Docker is unavailable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import List, Optional


class DockerServiceError(Exception):
    """Base exception for docker service related issues."""
    pass


@dataclass
class DockerContainer:
    """
    Represents a unified, decoupled representation of a Docker container.
    """
    id: str
    name: str
    status: str            # e.g., "running", "exited", "paused"
    image: str             # e.g., "nginx:alpine"
    cpu_percentage: float  # Current CPU usage percentage [0.0 - 100.0+]
    memory_usage_mb: float # Memory usage in MB
    memory_limit_mb: float # Total memory limit in MB


class IDockerService(ABC):
    """
    Interface (Abstract Base Class) defining the contract for Docker operations.
    All dashboard container services must adhere to this interface (Liskov Substitution Principle).
    """

    @abstractmethod
    def list_containers(self) -> List[DockerContainer]:
        """
        Retrieves a list of all containers with their real-time stats.

        Returns:
            List[DockerContainer]: List of unified DockerContainer models.
        """
        pass

    @abstractmethod
    def start_container(self, container_id: str) -> bool:
        """
        Starts a stopped container.

        Args:
            container_id (str): The ID of the container.

        Returns:
            bool: True if started successfully, False otherwise.
        """
        pass

    @abstractmethod
    def stop_container(self, container_id: str) -> bool:
        """
        Stops a running container.

        Args:
            container_id (str): The ID of the container.

        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        pass

    @abstractmethod
    def restart_container(self, container_id: str) -> bool:
        """
        Restarts a container.

        Args:
            container_id (str): The ID of the container.

        Returns:
            bool: True if restarted successfully, False otherwise.
        """
        pass


class RealDockerService(IDockerService):
    """
    Production implementation that interacts with the real Docker daemon using the official Python SDK.
    """

    def __init__(self) -> None:
        # Import dynamically so that the module can load even if the 'docker' package is not installed.
        import docker
        try:
            self.client = docker.from_env()
            # Test connection immediately
            self.client.ping()
            print("[INFO] RealDockerService initialized. Connected to Docker daemon successfully.")
        except Exception as e:
            raise DockerServiceError(f"Failed to connect to Docker daemon: {str(e)}") from e

    def _calculate_cpu_percentage(self, stats: dict) -> float:
        """
        Helper method to calculate CPU percentage from raw docker stats JSON.
        """
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})

            # Total and system CPU delta
            cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get("system_cpu_usage", 0)

            if system_delta > 0 and cpu_delta > 0:
                # Number of active CPU cores
                online_cpus = cpu_stats.get("online_cpus", len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [1])))
                return (cpu_delta / system_delta) * online_cpus * 100.0
            return 0.0
        except Exception:
            return 0.0

    def list_containers(self) -> List[DockerContainer]:
        import docker
        try:
            raw_containers = self.client.containers.list(all=True)
            containers: List[DockerContainer] = []

            for raw in raw_containers:
                # Default empty stats
                cpu_pct = 0.0
                mem_usage = 0.0
                mem_limit = 1024.0 # Fallback 1GB limit if stats are unreadable
                
                # Fetch stats only for running containers
                if raw.status == "running":
                    try:
                        # stream=False returns a single snapshot dictionary
                        stats = raw.stats(stream=False)
                        
                        # Calculate CPU
                        cpu_pct = self._calculate_cpu_percentage(stats)
                        
                        # Calculate Memory (bytes to MB)
                        mem_usage = stats.get("memory_stats", {}).get("usage", 0) / (1024 * 1024)
                        mem_limit = stats.get("memory_stats", {}).get("limit", 1024 * 1024 * 1024) / (1024 * 1024)
                    except Exception:
                        pass # Keep default values if stats fetch fails

                # Parse raw properties
                # Take first 12 characters of ID for standard short ID format
                short_id = raw.id[:12]
                name = raw.name
                status = raw.status
                image_name = raw.image.tags[0] if raw.image.tags else "unknown:latest"

                containers.append(
                    DockerContainer(
                        id=short_id,
                        name=name,
                        status=status,
                        image=image_name,
                        cpu_percentage=round(cpu_pct, 2),
                        memory_usage_mb=round(mem_usage, 1),
                        memory_limit_mb=round(mem_limit, 1)
                    )
                )
            return containers

        except Exception as e:
            raise DockerServiceError(f"Failed to list containers: {str(e)}") from e

    def start_container(self, container_id: str) -> bool:
        try:
            container = self.client.containers.get(container_id)
            container.start()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start container {container_id}: {e}")
            return False

    def stop_container(self, container_id: str) -> bool:
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=5)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to stop container {container_id}: {e}")
            return False

    def restart_container(self, container_id: str) -> bool:
        try:
            container = self.client.containers.get(container_id)
            container.restart(timeout=5)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to restart container {container_id}: {e}")
            return False


class MockDockerService(IDockerService):
    """
    Fallback mock implementation simulating realistic Docker stats and interactive container control.
    Enables development and testing without a live Docker daemon.
    """

    def __init__(self) -> None:
        print("[WARNING] MockDockerService initialized. Running in Simulator Mode.")
        # Create a set of static mock containers representing a standard microservice architecture
        self.containers = {
            "web_nginx": DockerContainer("a1b2c3d4e5f6", "web-nginx-prod", "running", "nginx:alpine", 1.2, 24.5, 512.0),
            "postgres_db": DockerContainer("b2c3d4e5f6g7", "postgres-db-replica", "running", "postgres:15-alpine", 3.5, 128.4, 2048.0),
            "redis_cache": DockerContainer("c3d4e5f6g7h8", "redis-cache-cluster", "running", "redis:alpine", 0.8, 18.2, 1024.0),
            "node_api": DockerContainer("d4e5f6g7h8i9", "node-api-server", "running", "node:18-alpine", 5.2, 85.0, 1024.0),
            "python_worker": DockerContainer("e5f6g7h8i9j0", "python-worker-service", "exited", "python:3.10-slim", 0.0, 0.0, 512.0)
        }

    def list_containers(self) -> List[DockerContainer]:
        # Simulate slight fluctuations in CPU and memory usage to make the UI look alive
        for key, container in self.containers.items():
            if container.status == "running":
                # CPU fluctuations
                change = random.uniform(-1.5, 1.5)
                container.cpu_percentage = max(0.5, min(95.0, round(container.cpu_percentage + change, 2)))
                
                # Memory fluctuations (slow dynamic drift)
                mem_change = random.uniform(-0.5, 0.5)
                container.memory_usage_mb = max(10.0, min(container.memory_limit_mb - 50, round(container.memory_usage_mb + mem_change, 1)))
            else:
                container.cpu_percentage = 0.0
                container.memory_usage_mb = 0.0

        return list(self.containers.values())

    def start_container(self, container_id: str) -> bool:
        for key, container in self.containers.items():
            if container.id == container_id:
                if container.status != "running":
                    container.status = "running"
                    container.cpu_percentage = random.uniform(5.0, 15.0)
                    container.memory_usage_mb = random.uniform(15.0, 60.0)
                    print(f"[MOCK] Started container: {container.name} ({container_id})")
                    return True
        return False

    def stop_container(self, container_id: str) -> bool:
        for key, container in self.containers.items():
            if container.id == container_id:
                if container.status == "running":
                    container.status = "exited"
                    container.cpu_percentage = 0.0
                    container.memory_usage_mb = 0.0
                    print(f"[MOCK] Stopped container: {container.name} ({container_id})")
                    return True
        return False

    def restart_container(self, container_id: str) -> bool:
        for key, container in self.containers.items():
            if container.id == container_id:
                print(f"[MOCK] Restarting container: {container.name} ({container_id})")
                container.status = "running"
                container.cpu_percentage = random.uniform(10.0, 25.0)  # Initial spike
                container.memory_usage_mb = max(15.0, container.memory_usage_mb * 0.9)
                return True
        return False


def get_docker_service() -> IDockerService:
    """
    Factory function following the Dependency Inversion Principle.
    Attempts to initialize the RealDockerService; if the docker library is missing 
    or connecting to the docker daemon fails, it gracefully falls back to MockDockerService.

    Returns:
        IDockerService: A concrete implementation of Docker management service.
    """
    try:
        # Check if python docker module is importable
        import docker
        # Attempt to connect to local docker daemon
        return RealDockerService()
    except (ImportError, DockerServiceError) as e:
        print(f"[INFO] Running in fallback mode. Reason: {e}")
        return MockDockerService()


# Quick verification script that runs only if this file is executed directly
if __name__ == "__main__":
    print("[TEST] Running DockerService verification...")
    service = get_docker_service()
    
    # List and display containers
    containers = service.list_containers()
    print(f"\n[TEST] Found {len(containers)} containers:")
    for c in containers:
        print(f"  - [{c.id}] Name: {c.name:25} | Status: {c.status:10} | CPU: {c.cpu_percentage:5}% | Mem: {c.memory_usage_mb:6}MB / {c.memory_limit_mb}MB")
        
    # Test Interactive Control (Mock or Real)
    if containers:
        first_id = containers[0].id
        print(f"\n[TEST] Simulating action: Restarting first container (ID: {first_id})")
        success = service.restart_container(first_id)
        print(f"[TEST] Action output: {'SUCCESS' if success else 'FAILED'}")
