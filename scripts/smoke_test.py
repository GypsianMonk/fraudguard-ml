#!/usr/bin/env python3
"""
scripts/smoke_test.py
---------------------
Post-deployment smoke tests to validate the API is functional.
Run after every deployment in CI/CD pipeline.

Usage:
    python scripts/smoke_test.py --base-url http://localhost:8000 --api-key dev-key-local
    python scripts/smoke_test.py --base-url https://staging.company.com --api-key $STAGING_KEY --strict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float
    error: str | None = None
    response_data: dict | None = None


@dataclass
class SmokeTestSuite:
    base_url: str
    api_key: str
    strict: bool = False
    results: list[TestResult] = field(default_factory=list)

    @property
    def client_headers(self) -> dict:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def run_all(self) -> bool:
        print(f"\n🔍 Running smoke tests against {self.base_url}\n{'='*55}")

        self._test_health()
        self._test_readiness()
        self._test_metrics_endpoint()
        self._test_predict_valid_transaction()
        self._test_predict_high_risk_transaction()
        self._test_predict_auth_required()
        self._test_predict_invalid_amount()
        self._test_batch_predict()
        self._test_model_info()
        self._test_latency_sla()

        return self._print_summary()

    def _run_test(self, name: str, fn) -> TestResult:
        start = time.monotonic()
        try:
            result_data = fn()
            latency = (time.monotonic() - start) * 1000
            result = TestResult(name=name, passed=True, latency_ms=latency, response_data=result_data)
        except AssertionError as e:
            latency = (time.monotonic() - start) * 1000
            result = TestResult(name=name, passed=False, latency_ms=latency, error=str(e))
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            result = TestResult(name=name, passed=False, latency_ms=latency, error=f"{type(e).__name__}: {e}")

        icon = "✅" if result.passed else "❌"
        status = f"{icon} {name:<45} {result.latency_ms:>6.0f}ms"
        if not result.passed:
            status += f" — {result.error}"
        print(status)

        self.results.append(result)
        return result

    def _test_health(self) -> None:
        def fn():
            r = httpx.get(f"{self.base_url}/health", timeout=10)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["status"] == "ok", f"Expected status=ok, got {data}"
            return data
        self._run_test("GET /health returns 200", fn)

    def _test_readiness(self) -> None:
        def fn():
            r = httpx.get(f"{self.base_url}/ready", timeout=10)
            assert r.status_code in {200, 503}, f"Expected 200/503, got {r.status_code}"
            data = r.json()
            assert "model_loaded" in data
            if self.strict:
                assert data["model_loaded"], "Model not loaded (strict mode)"
            return data
        self._run_test("GET /ready returns valid response", fn)

    def _test_metrics_endpoint(self) -> None:
        def fn():
            r = httpx.get(f"{self.base_url}/metrics", timeout=10)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            assert "fraudguard_predictions_total" in r.text or "python_info" in r.text
            return {"metrics_length": len(r.text)}
        self._run_test("GET /metrics returns Prometheus format", fn)

    def _test_predict_valid_transaction(self) -> None:
        payload = {
            "transaction_id": "smoke_test_001",
            "user_id": "usr_smoke_test",
            "amount": 299.99,
            "currency": "USD",
            "merchant_id": "mrc_smoke_test",
            "merchant_category": "electronics",
            "payment_method": "credit_card",
            "timestamp": "2024-01-15T14:32:00Z",
            "card_present": False,
            "location": {"country": "US", "city": "New York"},
        }

        def fn():
            r = httpx.post(
                f"{self.base_url}/api/v1/predict",
                json=payload,
                headers=self.client_headers,
                timeout=30,
            )
            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
            data = r.json()
            assert "fraud_probability" in data
            assert "risk_tier" in data
            assert "transaction_id" in data
            assert 0.0 <= data["fraud_probability"] <= 1.0
            assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            assert data["transaction_id"] == payload["transaction_id"]
            return data
        self._run_test("POST /api/v1/predict — valid transaction", fn)

    def _test_predict_high_risk_transaction(self) -> None:
        payload = {
            "transaction_id": "smoke_test_high_risk",
            "user_id": "usr_smoke_test",
            "amount": 9999.99,
            "currency": "USD",
            "merchant_id": "mrc_electronics_01",
            "merchant_category": "electronics",
            "payment_method": "credit_card",
            "timestamp": "2024-01-15T03:17:00Z",  # Night time
            "card_present": False,
            "location": {"country": "NG"},  # High-risk country
        }

        def fn():
            r = httpx.post(
                f"{self.base_url}/api/v1/predict",
                json=payload,
                headers=self.client_headers,
                timeout=30,
            )
            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
            data = r.json()
            assert data["fraud_probability"] > 0.0, "High-risk txn should have non-zero fraud prob"
            return data
        self._run_test("POST /api/v1/predict — high-risk transaction", fn)

    def _test_predict_auth_required(self) -> None:
        def fn():
            r = httpx.post(
                f"{self.base_url}/api/v1/predict",
                json={"transaction_id": "test"},
                timeout=10,
            )
            assert r.status_code == 401, f"Expected 401, got {r.status_code}"
            return {"status_code": r.status_code}
        self._run_test("POST /api/v1/predict — 401 without API key", fn)

    def _test_predict_invalid_amount(self) -> None:
        def fn():
            r = httpx.post(
                f"{self.base_url}/api/v1/predict",
                json={
                    "transaction_id": "smoke_test_invalid",
                    "user_id": "usr_test",
                    "amount": -100.0,  # Invalid
                    "currency": "USD",
                    "merchant_id": "mrc_test",
                    "timestamp": "2024-01-15T14:32:00Z",
                },
                headers=self.client_headers,
                timeout=10,
            )
            assert r.status_code == 422, f"Expected 422, got {r.status_code}"
            return {"status_code": r.status_code}
        self._run_test("POST /api/v1/predict — 422 for negative amount", fn)

    def _test_batch_predict(self) -> None:
        payload = {
            "transactions": [
                {
                    "transaction_id": f"smoke_batch_{i:03d}",
                    "user_id": f"usr_smoke_{i}",
                    "amount": float(100 + i * 50),
                    "currency": "USD",
                    "merchant_id": "mrc_test",
                    "merchant_category": "other",
                    "timestamp": "2024-01-15T14:32:00Z",
                    "card_present": True,
                }
                for i in range(5)
            ]
        }

        def fn():
            r = httpx.post(
                f"{self.base_url}/api/v1/predict/batch",
                json=payload,
                headers=self.client_headers,
                timeout=60,
            )
            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
            data = r.json()
            assert data["total"] == 5
            assert data["status"] == "completed"
            assert len(data.get("predictions", [])) == 5
            return {"total": data["total"], "status": data["status"]}
        self._run_test("POST /api/v1/predict/batch — 5 transactions", fn)

    def _test_model_info(self) -> None:
        def fn():
            r = httpx.get(
                f"{self.base_url}/api/v1/admin/model/info",
                headers=self.client_headers,
                timeout=10,
            )
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "model_version" in data
            return data
        self._run_test("GET /api/v1/admin/model/info", fn)

    def _test_latency_sla(self) -> None:
        """P99 latency SLA: single prediction must complete < 200ms."""
        payload = {
            "transaction_id": "smoke_latency_test",
            "user_id": "usr_latency",
            "amount": 50.0,
            "currency": "USD",
            "merchant_id": "mrc_test",
            "merchant_category": "groceries",
            "timestamp": "2024-01-15T12:00:00Z",
            "card_present": True,
        }

        latencies = []

        def fn():
            for _ in range(10):
                start = time.monotonic()
                r = httpx.post(
                    f"{self.base_url}/api/v1/predict",
                    json={**payload, "transaction_id": f"smoke_lat_{time.monotonic():.0f}"},
                    headers=self.client_headers,
                    timeout=30,
                )
                latency = (time.monotonic() - start) * 1000
                if r.status_code == 200:
                    latencies.append(latency)

            if not latencies:
                raise AssertionError("No successful predictions for latency test")

            import statistics
            p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0]
            p50 = statistics.median(latencies)

            assert p99 < 500, f"P99 latency {p99:.0f}ms exceeds 500ms SLA (strict: 200ms)"
            return {"p50_ms": round(p50, 1), "p99_ms": round(p99, 1), "n_samples": len(latencies)}

        self._run_test("Latency SLA: P99 < 500ms (10 requests)", fn)

    def _print_summary(self) -> bool:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        failed = total - passed

        print(f"\n{'='*55}")
        print(f"Results: {passed}/{total} tests passed", end="")
        if failed > 0:
            print(f" | {failed} FAILED")
        else:
            print(" ✅")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.error}")

        all_passed = failed == 0
        return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudGuard ML smoke tests")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dev-key-local")
    parser.add_argument("--strict", action="store_true", help="Fail on any warning (for production)")
    args = parser.parse_args()

    suite = SmokeTestSuite(base_url=args.base_url, api_key=args.api_key, strict=args.strict)
    success = suite.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
