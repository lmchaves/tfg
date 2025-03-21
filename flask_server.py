from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import random
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Datos iniciales (simulación de switches y enlaces)
network_data = {
    "switches": ["s1", "s2", "s3", "s4"],
    "links": [
        {"src": "s1", "dst": "s2", "load": 0.1, "delay": 10, "packet_loss": 0.01},
        {"src": "s2", "dst": "s3", "load": 0.2, "delay": 20, "packet_loss": 0.02},
        {"src": "s3", "dst": "s4", "load": 0.05, "delay": 5, "packet_loss": 0.005},
    ],
    "best_path": []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/topology')
def get_topology():
    return jsonify(network_data)

@app.route('/update', methods=['POST'])
def update_data():
    """Simula la recepción de datos de Ryu (o lo puedes conectar con Ryu)"""
    global network_data
    for link in network_data["links"]:
        link["load"] = random.uniform(0.01, 1.0)
        link["delay"] = random.randint(5, 50)
        link["packet_loss"] = random.uniform(0, 0.05)
    
    network_data["best_path"] = ["s1", "s2", "s3"]
    socketio.emit('update_topology', network_data)  # Enviar actualización a la web
    return jsonify({"status": "updated"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
