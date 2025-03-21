from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def index():
    # Supongamos que tu index.html está en templates/index.html
    return render_template("index.html")

@app.route("/topology")
def get_topology():
    # EJEMPLO: Datos estáticos
    # Queremos representar 3 switches: s1, s2, s3
    # Y 2 enlaces: (s1 - s2) y (s2 - s3)
    # Con métricas ficticias
    topology_data = {
        "switches": ["s1", "s2", "s3"],
        "links": [
            {"src": "s1", "dst": "s2", "load": 0.45, "delay": 10, "packet_loss": 0.02},
            {"src": "s2", "dst": "s3", "load": 0.80, "delay": 20, "packet_loss": 0.05}
        ],
        # Si quieres también un "best_path" ficticio:
        "best_path": ["s1", "s2", "s3"]
    }
    return jsonify(topology_data)

if __name__ == "__main__":
    app.run(debug=True)
