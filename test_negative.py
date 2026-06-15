import sys
import json
import urllib.request


def test_negative_data():
    base_url = "http://127.0.0.1:5000"
    headers = {"Content-Type": "application/json"}

    test_cases = [
        {
            "name": "普通面积图-含负值",
            "stacked": False,
            "data": {
                "times": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"],
                "series": {
                    "利润": [100, -50, 80, -120, 60, -30, 150],
                    "成本": [-80, 60, -40, 90, -70, 50, -60],
                },
                "title": "普通面积图-含负数测试",
            },
        },
        {
            "name": "堆叠面积图-含负值",
            "stacked": True,
            "data": {
                "times": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"],
                "series": {
                    "收入": [200, 180, 220, 190, 250, 210, 280],
                    "支出": [-150, -160, -140, -170, -130, -180, -120],
                    "投资收益": [30, -20, 50, -40, 60, -10, 80],
                },
                "title": "堆叠面积图-含负数测试",
            },
        },
        {
            "name": "堆叠面积图-全正值（对比）",
            "stacked": True,
            "data": {
                "times": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
                "series": {
                    "产品A": [100, 120, 150, 130, 180],
                    "产品B": [80, 90, 110, 100, 130],
                },
                "title": "堆叠面积图-全正值（对比测试）",
            },
        },
        {
            "name": "普通面积图-全负值",
            "stacked": False,
            "data": {
                "times": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
                "series": {
                    "亏损": [-50, -80, -60, -100, -70],
                    "负债": [-30, -40, -35, -50, -45],
                },
                "title": "普通面积图-全负值测试",
            },
        },
    ]

    results = []
    for i, tc in enumerate(test_cases):
        body = {**tc["data"], "stacked": tc["stacked"], "format": "json"}
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
    print("开始负数面积图测试...\n")
    test_negative_data()
    print("\n所有测试通过！负数数据已正确支持。")
