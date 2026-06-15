import csv
import io
import json
import os
import uuid

from flask import Flask, render_template, request, jsonify, send_file

from chart_generator import generate_area_chart, generate_from_csv_data, generate_from_json_data

app = Flask(__name__)

CHART_DIR = os.path.join(os.path.dirname(__file__), "generated_charts")
os.makedirs(CHART_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chart", methods=["POST"])
def create_chart():
    try:
        data = request.get_json(force=True)

        fmt = data.get("format", "json")
        stacked = data.get("stacked", False)
        stack_baseline = data.get("stack_baseline", "zero")
        normalize = data.get("normalize", False)
        title = data.get("title", "面积图")
        xlabel = data.get("xlabel", "时间")
        ylabel = data.get("ylabel", "数值")
        alpha = float(data.get("alpha", 0.6))

        if fmt == "csv":
            csv_text = data["csv_text"]
            time_col = data["time_col"]
            value_cols = data["value_cols"]
            buf = generate_from_csv_data(
                csv_text, time_col, value_cols, stacked=stacked,
                stack_baseline=stack_baseline, normalize=normalize,
                title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha,
            )
        else:
            json_str = json.dumps(data)
            buf = generate_from_json_data(
                json_str, stacked=stacked,
                stack_baseline=stack_baseline, normalize=normalize,
                title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha,
            )

        chart_id = uuid.uuid4().hex
        filepath = os.path.join(CHART_DIR, f"{chart_id}.png")
        with open(filepath, "wb") as f:
            f.write(buf.getvalue())

        return jsonify({"chart_id": chart_id, "url": f"/api/chart/{chart_id}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chart/upload", methods=["POST"])
def upload_chart():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        stacked = request.form.get("stacked", "false").lower() == "true"
        stack_baseline = request.form.get("stack_baseline", "zero")
        normalize = request.form.get("normalize", "false").lower() == "true"
        title = request.form.get("title", "面积图")
        xlabel = request.form.get("xlabel", "时间")
        ylabel = request.form.get("ylabel", "数值")
        alpha = float(request.form.get("alpha", 0.6))
        time_col = request.form.get("time_col", "")
        value_cols_str = request.form.get("value_cols", "")

        content = file.read().decode("utf-8-sig")

        if file.filename.endswith(".json"):
            buf = generate_from_json_data(
                content, stacked=stacked,
                stack_baseline=stack_baseline, normalize=normalize,
                title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha,
            )
        elif file.filename.endswith(".csv"):
            if not time_col or not value_cols_str:
                return jsonify({"error": "CSV requires time_col and value_cols"}), 400
            value_cols = [c.strip() for c in value_cols_str.split(",")]
            buf = generate_from_csv_data(
                content, time_col, value_cols, stacked=stacked,
                stack_baseline=stack_baseline, normalize=normalize,
                title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha,
            )
        else:
            return jsonify({"error": "Unsupported file format. Use .csv or .json"}), 400

        chart_id = uuid.uuid4().hex
        filepath = os.path.join(CHART_DIR, f"{chart_id}.png")
        with open(filepath, "wb") as f:
            f.write(buf.getvalue())

        return jsonify({"chart_id": chart_id, "url": f"/api/chart/{chart_id}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chart/<chart_id>")
def get_chart(chart_id):
    filepath = os.path.join(CHART_DIR, f"{chart_id}.png")
    if not os.path.exists(filepath):
        return jsonify({"error": "Chart not found"}), 404
    return send_file(filepath, mimetype="image/png")


@app.route("/api/demo", methods=["GET"])
def demo_chart():
    import numpy as np

    base = "2026-01-01"
    from datetime import datetime, timedelta

    start = datetime.strptime(base, "%Y-%m-%d")
    times = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]

    np.random.seed(42)
    sales = np.cumsum(np.random.randn(30) * 10 + 50).clip(min=0).tolist()
    returns = np.cumsum(np.random.randn(30) * 5 + 20).clip(min=0).tolist()
    visits = np.cumsum(np.random.randn(30) * 8 + 30).clip(min=0).tolist()

    stacked = request.args.get("stacked", "false").lower() == "true"
    stack_baseline = request.args.get("baseline", "zero")
    normalize = request.args.get("normalize", "false").lower() == "true"

    data = {
        "times": times,
        "series": {
            "销售额": sales,
            "退货额": returns,
            "访问量": visits,
        },
        "stacked": stacked,
        "stack_baseline": stack_baseline,
        "normalize": normalize,
        "title": "示例面积图 - 30天数据趋势",
        "ylabel": "数值",
    }

    buf = generate_from_json_data(
        json.dumps(data),
        stacked=stacked,
        stack_baseline=stack_baseline,
        normalize=normalize,
        title=data["title"],
        ylabel=data["ylabel"],
    )

    chart_id = uuid.uuid4().hex
    filepath = os.path.join(CHART_DIR, f"{chart_id}.png")
    with open(filepath, "wb") as f:
        f.write(buf.getvalue())

    return jsonify({"chart_id": chart_id, "url": f"/api/chart/{chart_id}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
