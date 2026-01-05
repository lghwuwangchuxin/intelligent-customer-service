"""
Mock Data Generators for Multi-Agent System.
Provides simulated data for charging and energy management domains.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List


class ChargingMockData:
    """Mock data generator for charging domain."""

    # Sample charging station data
    STATION_NAMES = [
        "国网充电站", "特来电充电站", "星星充电站", "云快充站点",
        "万达广场充电站", "银泰百货充电站", "奥特莱斯充电站",
        "华润万家充电站", "沃尔玛停车场站", "宜家家居充电站",
    ]

    STATION_TYPES = ["快充站", "慢充站", "超级充电站"]

    ERROR_CODES = {
        "E001": {"name": "充电枪锁定故障", "solution": "请尝试按住充电枪解锁按钮3秒，如仍无法解锁请联系客服"},
        "E002": {"name": "通信中断", "solution": "请检查网络信号，尝试重新扫码启动充电"},
        "E003": {"name": "过温保护", "solution": "充电桩温度过高，已自动保护停机，请等待10分钟后重试"},
        "E004": {"name": "电压异常", "solution": "请检查车辆充电口是否正常，如问题持续请联系客服"},
        "E005": {"name": "充电中断", "solution": "可能由于车辆BMS保护导致，请检查车辆状态后重试"},
        "E006": {"name": "支付失败", "solution": "请检查账户余额或更换支付方式"},
        "E007": {"name": "充电桩离线", "solution": "该充电桩暂时无法使用，请选择其他充电桩"},
        "E008": {"name": "功率不匹配", "solution": "您的车辆不支持当前充电桩功率，建议选择其他充电桩"},
    }

    @classmethod
    def generate_nearby_stations(
        cls,
        latitude: float = 31.2304,
        longitude: float = 121.4737,
        radius_km: float = 5.0,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate mock nearby charging stations."""
        stations = []
        for i in range(limit):
            # Generate random distance within radius
            distance = round(random.uniform(0.1, radius_km), 2)
            available = random.randint(0, 8)
            total = random.randint(available, 12)

            stations.append({
                "station_id": f"CS{str(i+1).zfill(4)}",
                "name": random.choice(cls.STATION_NAMES),
                "type": random.choice(cls.STATION_TYPES),
                "distance_km": distance,
                "latitude": latitude + random.uniform(-0.01, 0.01),
                "longitude": longitude + random.uniform(-0.01, 0.01),
                "available_ports": available,
                "total_ports": total,
                "price_per_kwh": round(random.uniform(0.8, 2.0), 2),
                "service_fee": round(random.uniform(0.1, 0.5), 2),
                "fast_charging": random.choice([True, False]),
                "max_power_kw": random.choice([60, 120, 180, 250]),
                "operating_hours": "24小时",
                "rating": round(random.uniform(3.5, 5.0), 1),
                "status": "运营中" if available > 0 else "满载",
            })

        # Sort by distance
        stations.sort(key=lambda x: x["distance_km"])
        return stations

    @classmethod
    def generate_charging_status(cls, session_id: str = None) -> Dict[str, Any]:
        """Generate mock charging session status."""
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        soc = random.randint(20, 95)
        power = random.choice([30, 60, 90, 120, 150])
        energy = round(random.uniform(10, 50), 2)
        cost = round(energy * random.uniform(1.0, 1.8), 2)

        # Estimate remaining time based on SOC
        remaining_percent = 100 - soc
        estimated_minutes = int(remaining_percent * 0.8)  # Rough estimate

        statuses = ["charging", "completed", "waiting", "paused"]
        status = random.choice(statuses) if soc < 100 else "completed"

        return {
            "session_id": session_id,
            "status": status,
            "status_cn": {
                "charging": "充电中",
                "completed": "充电完成",
                "waiting": "等待中",
                "paused": "已暂停",
            }.get(status, "未知"),
            "soc_percent": soc,
            "target_soc": 100,
            "power_kw": power if status == "charging" else 0,
            "voltage_v": random.randint(380, 420),
            "current_a": round(power * 1000 / 400, 1) if status == "charging" else 0,
            "energy_kwh": energy,
            "duration_minutes": random.randint(10, 120),
            "estimated_minutes": estimated_minutes if status == "charging" else 0,
            "cost_current": cost,
            "start_time": (datetime.now() - timedelta(minutes=random.randint(10, 60))).isoformat(),
            "station_name": random.choice(cls.STATION_NAMES),
            "port_number": random.randint(1, 12),
        }

    @classmethod
    def generate_billing_history(
        cls,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate mock billing history."""
        bills = []
        for i in range(limit):
            days_ago = random.randint(1, 90)
            date = datetime.now() - timedelta(days=days_ago)
            energy = round(random.uniform(15, 60), 2)
            price = round(random.uniform(1.0, 1.8), 2)
            service_fee = round(random.uniform(5, 20), 2)
            electricity_cost = round(energy * price, 2)
            total = round(electricity_cost + service_fee, 2)

            bills.append({
                "bill_id": f"BILL{str(uuid.uuid4())[:8].upper()}",
                "user_id": user_id,
                "station_name": random.choice(cls.STATION_NAMES),
                "date": date.strftime("%Y-%m-%d"),
                "start_time": date.strftime("%H:%M"),
                "duration_minutes": random.randint(30, 180),
                "energy_kwh": energy,
                "price_per_kwh": price,
                "electricity_cost": electricity_cost,
                "service_fee": service_fee,
                "parking_fee": 0.0,
                "discount": round(random.uniform(0, total * 0.1), 2),
                "total_amount": total,
                "payment_method": random.choice(["微信", "支付宝", "银行卡"]),
                "payment_status": "已支付",
            })

        # Sort by date descending
        bills.sort(key=lambda x: x["date"], reverse=True)
        return bills

    @classmethod
    def get_fault_solution(cls, error_code: str) -> Dict[str, Any]:
        """Get fault diagnosis and solution."""
        error_info = cls.ERROR_CODES.get(error_code.upper())
        if error_info:
            return {
                "error_code": error_code.upper(),
                "error_name": error_info["name"],
                "solution": error_info["solution"],
                "severity": random.choice(["低", "中", "高"]),
                "estimated_fix_time": f"{random.randint(5, 30)}分钟",
                "need_support": random.choice([True, False]),
                "support_phone": "400-xxx-xxxx",
            }
        else:
            return {
                "error_code": error_code,
                "error_name": "未知错误",
                "solution": "请联系客服获取帮助",
                "severity": "未知",
                "need_support": True,
                "support_phone": "400-xxx-xxxx",
            }


class EnergyMockData:
    """Mock data generator for energy management domain."""

    @classmethod
    def generate_energy_report(
        cls,
        station_id: str,
        period: str = "month",
    ) -> Dict[str, Any]:
        """Generate mock energy efficiency report."""
        if period == "day":
            total_energy = round(random.uniform(500, 2000), 2)
            days = 1
        elif period == "week":
            total_energy = round(random.uniform(3000, 10000), 2)
            days = 7
        else:  # month
            total_energy = round(random.uniform(10000, 50000), 2)
            days = 30

        peak_load = round(random.uniform(300, 500), 2)
        avg_load = round(total_energy / (days * 24), 2)
        efficiency = round(random.uniform(88, 98), 1)
        utilization = round(random.uniform(50, 85), 1)

        return {
            "station_id": station_id,
            "report_period": period,
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "total_energy_kwh": total_energy,
            "efficiency_percent": efficiency,
            "peak_load_kw": peak_load,
            "average_load_kw": avg_load,
            "min_load_kw": round(avg_load * 0.3, 2),
            "utilization_percent": utilization,
            "total_sessions": random.randint(100, 1000),
            "avg_session_duration_min": random.randint(30, 90),
            "revenue": round(total_energy * random.uniform(1.2, 1.8), 2),
            "cost": round(total_energy * random.uniform(0.6, 0.9), 2),
            "carbon_saved_kg": round(total_energy * 0.5, 2),
            "comparison_with_previous": {
                "energy_change_percent": round(random.uniform(-15, 20), 1),
                "efficiency_change_percent": round(random.uniform(-5, 10), 1),
                "utilization_change_percent": round(random.uniform(-10, 15), 1),
            },
        }

    @classmethod
    def generate_device_health(cls, device_id: str) -> Dict[str, Any]:
        """Generate mock device health status."""
        health_score = random.randint(60, 100)
        if health_score >= 90:
            status = "健康"
            status_color = "green"
        elif health_score >= 70:
            status = "良好"
            status_color = "yellow"
        else:
            status = "需维护"
            status_color = "red"

        components = [
            {"name": "充电模块", "health": random.randint(70, 100), "last_check": "2024-01-15"},
            {"name": "通信模块", "health": random.randint(80, 100), "last_check": "2024-01-14"},
            {"name": "计量模块", "health": random.randint(75, 100), "last_check": "2024-01-13"},
            {"name": "安全模块", "health": random.randint(85, 100), "last_check": "2024-01-12"},
            {"name": "显示屏", "health": random.randint(60, 100), "last_check": "2024-01-11"},
        ]

        return {
            "device_id": device_id,
            "device_type": random.choice(["直流快充", "交流慢充", "超级充电"]),
            "health_score": health_score,
            "status": status,
            "status_color": status_color,
            "uptime_percent": round(random.uniform(95, 99.9), 2),
            "total_charging_sessions": random.randint(1000, 50000),
            "total_energy_delivered_kwh": random.randint(10000, 500000),
            "last_maintenance_date": (datetime.now() - timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d"),
            "next_maintenance_date": (datetime.now() + timedelta(days=random.randint(30, 90))).strftime("%Y-%m-%d"),
            "components": components,
            "alerts": [
                {"type": "warning", "message": "建议近期进行常规维护", "time": datetime.now().isoformat()}
            ] if health_score < 80 else [],
            "predicted_failure_risk": round(random.uniform(0, 0.3), 2) if health_score < 70 else round(random.uniform(0, 0.1), 2),
        }

    @classmethod
    def generate_optimization_suggestions(cls, station_id: str) -> Dict[str, Any]:
        """Generate cost optimization suggestions."""
        current_cost = round(random.uniform(50000, 200000), 2)
        potential_savings = round(current_cost * random.uniform(0.1, 0.25), 2)

        suggestions = [
            {
                "id": 1,
                "title": "峰谷电价优化",
                "description": "建议引导用户在谷时段充电，可降低电力成本",
                "potential_savings": round(potential_savings * 0.4, 2),
                "implementation_difficulty": "低",
                "priority": "高",
            },
            {
                "id": 2,
                "title": "功率调度优化",
                "description": "根据电网负荷动态调整充电功率，避免电力增容费",
                "potential_savings": round(potential_savings * 0.3, 2),
                "implementation_difficulty": "中",
                "priority": "中",
            },
            {
                "id": 3,
                "title": "储能系统配置",
                "description": "配置储能系统进行削峰填谷，降低电力需求费用",
                "potential_savings": round(potential_savings * 0.2, 2),
                "implementation_difficulty": "高",
                "priority": "低",
            },
            {
                "id": 4,
                "title": "光伏发电接入",
                "description": "接入光伏发电系统，降低用电成本",
                "potential_savings": round(potential_savings * 0.1, 2),
                "implementation_difficulty": "高",
                "priority": "低",
            },
        ]

        return {
            "station_id": station_id,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "current_monthly_cost": current_cost,
            "potential_monthly_savings": potential_savings,
            "savings_percentage": round(potential_savings / current_cost * 100, 1),
            "suggestions": suggestions,
            "peak_hours": ["10:00-12:00", "18:00-21:00"],
            "valley_hours": ["23:00-07:00"],
            "recommended_pricing_strategy": {
                "peak_price": 2.0,
                "normal_price": 1.5,
                "valley_price": 0.8,
            },
        }

    @classmethod
    def generate_load_forecast(
        cls,
        station_id: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Generate load forecast data."""
        base_load = random.randint(50, 150)
        predictions = []

        for i in range(hours):
            hour = (datetime.now() + timedelta(hours=i)).hour
            # Simulate daily load pattern
            if 7 <= hour <= 9:  # Morning peak
                factor = 1.3
            elif 12 <= hour <= 14:  # Lunch
                factor = 0.9
            elif 17 <= hour <= 21:  # Evening peak
                factor = 1.5
            elif 0 <= hour <= 6:  # Night
                factor = 0.5
            else:
                factor = 1.0

            load = round(base_load * factor * random.uniform(0.9, 1.1), 2)
            predictions.append({
                "hour": i,
                "timestamp": (datetime.now() + timedelta(hours=i)).isoformat(),
                "predicted_load_kw": load,
                "confidence": round(random.uniform(0.8, 0.95), 2),
                "is_peak": factor > 1.2,
            })

        max_load = max(p["predicted_load_kw"] for p in predictions)
        avg_load = round(sum(p["predicted_load_kw"] for p in predictions) / len(predictions), 2)

        return {
            "station_id": station_id,
            "forecast_generated_at": datetime.now().isoformat(),
            "forecast_hours": hours,
            "predictions": predictions,
            "summary": {
                "max_predicted_load_kw": max_load,
                "avg_predicted_load_kw": avg_load,
                "min_predicted_load_kw": min(p["predicted_load_kw"] for p in predictions),
                "peak_hours": [p["hour"] for p in predictions if p["is_peak"]],
                "total_predicted_energy_kwh": round(sum(p["predicted_load_kw"] for p in predictions), 2),
            },
            "recommendations": [
                "建议在预测低谷时段安排设备维护",
                "高峰时段建议启用功率调度策略",
            ],
        }
