from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Variable global para almacenar los datos actualizados de la red.
network_data = {
    "switches": [],
    "links": [],
    "best_path": [],
    "best_cost": 0,
    "counter": 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/topology')
def get_topology():
    return jsonify(network_data)

@app.route('/update', methods=['POST'])
def update_data():
    """Actualiza los datos de la red recibidos del monitor de Ryu."""
    global network_data
    network_data = request.get_json()  # Se espera el formato D3-friendly
    socketio.emit('update_topology', network_data)
    return jsonify({"status": "updated"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
