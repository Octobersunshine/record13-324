import sys
import json
import urllib.request


def test_stacked_options():
    base_url = "http://127.0.0.1:5000"
    headers = {"Content-Type": "application/json"}

    times = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"]
    series = {
        "产品A": [120, 150, 180, 200, 170, 210, 250],
        "产品B": [80, 100, 90, 120, 140, 130, 160],
        "产品C": [50, 60, 70, 65, 80, 90, 100],
    }

    test_cases = [
        {"name": "堆叠-zero 基线", "stack_baseline": "zero", "normalize": False},
        {"name": "堆叠-sym 基线", "stack_baseline": "sym", "normalize": False},
        {"name": "堆叠-wiggle 基线", "stack_baseline": "wiggle", "normalize": False},
        {"name": "堆叠-weighted_wiggle 基线", "stack_baseline": "weighted_wiggle", "normalize": False},
        {"name": "堆叠-zero+百分比", "stack_baseline": "zero", "normalize": True},
        {"name": "堆叠-sym+百分比", "stack_baseline": "sym", "normalize": True},
        {"name": "普通面积图（非堆叠）", "stacked": False, "stack_baseline": "zero", "normalize": False},
        {"name": "堆叠-无效基线回退到zero", "stack_baseline": "invalid_baseline", "normalize": False},
    ]

    results = []
    for i, tc in enumerate(test_cases):
        body = {
            "times": times,
            "series": series,
            "stacked": tc.get("stacked", True),
            "stack_baseline": tc.get("stack_baseline", "zero"),
            "normalize": tc.get("normalize", False),
            "title": tc["name"],
            "format": "json",
        }
        try:
            req = urllib.request.Request(
                f"{base_url}/api/chart",
                data=json.dumps(body).encode(),
                headers=headers,
            )
            resp = urllib.request.urlopen(req)
            result = json.loads(resp.read().decode())
            chart_url = base_url + result["url"]

            img_req = urllib.request.Request(chart_url)
            img_resp = urllib.request.urlopen(img_req)
            img_data = img_resp.read()

            ok = len(img_data) > 1000 and img_resp.headers["Content-Type"] == "image/png"
            results.append({
                "case": tc["name"],
                "status": "PASS" if ok else "FAIL",
                "chart_id": result.get("chart_id", ""),
                "size": len(img_data),
            })
            print(f"  [{i+1}] {tc['name']}: {'PASS' if ok else 'FAIL'} ({len(img_data)} bytes)")
        except Exception as e:
            results.append({
                "case": tc["name"],
                "status": "ERROR",
                "error": str(e),
            })
            print(f"  [{i+1}] {tc['name']}: ERROR - {e}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n测试总结: {passed}/{total} 通过")

    if passed < total:
        print("\n失败详情:")
        for r in results:
            if r["status"] != "PASS":
                print(f"  - {r['case']}: {r['status']} - {r.get('error', r.get('size', ''))}")
        sys.exit(1)


if __name__ == "__main__":
    print("开始堆叠面积图功能测试...\n")
    test_stacked_options()
    print("\n所有测试通过！堆叠面积图选项已正确支持。")
