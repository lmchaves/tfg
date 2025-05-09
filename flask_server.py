from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import requests 
import json
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# URL del endpoint de control en el monitor Ryu
RYU_CONTROL_URL = "http://127.0.0.1:8080/control"

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

@socketio.on('control_command') 
def handle_control_command(data):
    """
    Recibe comandos de control del frontend y los env\u00EDa al monitor de Ryu.
    Esta funci\u00F3n es llamada por el frontend v\u00EDa Socket.IO.
    """
    print(f"Comando de control recibido del frontend: {data}")
    try:
        # Envía el comando al endpoint de control en el monitor Ryu  HTTP POST
        response = requests.post(RYU_CONTROL_URL, json=data)
        if response.status_code == 200:
            print("Comando enviado a Ryu exitosamente.")
            socketio.emit('control_response', {"status": "success", "command": data.get('command')})
        else:
            print(f"Error al enviar comando a Ryu: {response.status_code} - {response.text}")
            socketio.emit('control_response', {"status": "error", "command": data.get('command'), "message": response.text})
    except requests.exceptions.RequestException as e:
        print(f"Error de conexi\u00F3n al enviar comando a Ryu: {e}")
        socketio.emit('control_response', {"status": "error", "command": data.get('command'), "message": f"Connection error: {e}"})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)