"""
Charging Station Search Tool.
Provides functionality to search for nearby charging stations.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for importing common module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import ChargingMockData

logger = logging.getLogger(__name__)


class StationSearchTool:
    """
    Tool for searching nearby charging stations.

    Uses mock data for development, can be replaced with real API.
    """

    def __init__(self):
        """Initialize station search tool."""
        self.mock_data = ChargingMockData()

    async def search_nearby(
        self,
        latitude: float = 31.2304,
        longitude: float = 121.4737,
        radius_km: float = 5.0,
        limit: int = 5,
        fast_charging_only: bool = False,
        min_available_ports: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearby charging stations.

        Args:
            latitude: User's latitude
            longitude: User's longitude
            radius_km: Search radius in kilometers
            limit: Maximum number of results
            fast_charging_only: Filter for fast charging stations only
            min_available_ports: Minimum required available ports

        Returns:
            List of charging station information
        """
        logger.info(
            f"Searching stations near ({latitude}, {longitude}), "
            f"radius={radius_km}km, limit={limit}"
        )

        # Get mock data
        stations = self.mock_data.generate_nearby_stations(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            limit=limit * 2,  # Get more to allow filtering
        )

        # Apply filters
        if fast_charging_only:
            stations = [s for s in stations if s.get("fast_charging")]

        if min_available_ports > 0:
            stations = [
                s for s in stations
                if s.get("available_ports", 0) >= min_available_ports
            ]

        # Limit results
        stations = stations[:limit]

        logger.info(f"Found {len(stations)} stations")
        return stations

    def format_station_info(self, station: Dict[str, Any]) -> str:
        """
        Format station information for display.

        Args:
            station: Station information dictionary

        Returns:
            Formatted string
        """
        lines = [
            f"**{station['name']}** ({station['type']})",
            f"  距离: {station['distance_km']}公里",
            f"  可用充电桩: {station['available_ports']}/{station['total_ports']}",
            f"  电价: {station['price_per_kwh']}元/度",
            f"  服务费: {station['service_fee']}元/度",
            f"  最大功率: {station['max_power_kw']}kW",
            f"  评分: {station['rating']}/5.0",
            f"  状态: {station['status']}",
        ]
        return "\n".join(lines)

    def format_stations_list(self, stations: List[Dict[str, Any]]) -> str:
        """
        Format list of stations for display.

        Args:
            stations: List of station dictionaries

        Returns:
            Formatted string
        """
        if not stations:
            return "附近没有找到充电站。"

        lines = [f"找到 {len(stations)} 个附近充电站：\n"]
        for i, station in enumerate(stations, 1):
            lines.append(f"{i}. {self.format_station_info(station)}\n")

        return "\n".join(lines)
