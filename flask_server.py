import eventlet
eventlet.monkey_patch()


from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import requests 
import json
import eventlet
import collections
import xmlrpc.client
import threading
import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# URL del endpoint de control en el monitor Ryu
RYU_CONTROL_URL = "http://127.0.0.1:8080/control"
MININET_RPC_URL = "http://127.0.0.1:8000/"
mininet_rpc = xmlrpc.client.ServerProxy(MININET_RPC_URL, allow_none=True) # Cliente RPC

RYU_NOTIFY_PARAM_URL = "http://127.0.0.1:8080/notify_param_change"

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
    network_data = request.get_json()  
    socketio.emit('update_topology', network_data)
    return jsonify({"status": "updated"})

@socketio.on('control_command') 
def handle_control_command(data):
    """
    Recibe comandos de control del frontend y los envía al monitor de Ryu.
    Esta función es llamada por el frontend vía Socket.IO.
    """
    print(f"Comando de control recibido del frontend: {data}")
    command = data.get('command') 
    if command in ['pause', 'continue', 'skip', 'set_endpoints']:
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
            print(f"Error de conexión al enviar comando a Ryu: {e}")
            socketio.emit('control_response', {"status": "error", "command": data.get('command'), "message": f"Connection error: {e}"})
    elif command == 'set_link_param':
        src_dpid = data.get('src_dpid')
        dst_dpid = data.get('dst_dpid')
        param_name = data.get('param_name')
        value = data.get('value')
        print(f"RPC: Recibida petición para set_link_param: {src_dpid}-{dst_dpid}, {param_name}={value}")
        try:
            success = mininet_rpc.set_link_param(src_dpid, dst_dpid, param_name, value)
            if success:
                print(f"RPC: Parámetro de enlace {param_name} actualizado en Mininet.")


                notification_payload = {
                    'src_dpid': src_dpid,
                    'dst_dpid': dst_dpid,
                    'param_name': param_name,
                    'value': value
                }
                print(f"Flask: Iniciando hilo para enviar notificación a Ryu: {notification_payload}")

                def send_ryu_notification(url, data):
                         try:
                             requests.post(url, json=data, timeout=1) 
                         except requests.exceptions.RequestException as e:
                              print(f"Error enviando notificación a Ryu: {e}") 
                
                # Enviar a Ryu en un hilo separado
                notification_thread = threading.Thread(target=send_ryu_notification, args=(RYU_NOTIFY_PARAM_URL, notification_payload))
                notification_thread.start()


                socketio.emit('control_response', {"status": "success", "message": f"Parámetro {param_name} de enlace {src_dpid}-{dst_dpid} cambiado."})
            else:
                print(f"RPC: Fallo al actualizar parámetro {param_name} en Mininet.")
                socketio.emit('control_response', {"status": "error", "message": f"Fallo al cambiar parámetro {param_name} de enlace {src_dpid}-{dst_dpid}."})
        except xmlrpc.client.Fault as e:
            print(f"RPC ERROR: Fallo XML-RPC al llamar a set_link_param: {e}")
            socketio.emit('control_response', {"status": "error", "message": f"Error RPC al cambiar {param_name}: {e}"})
        except Exception as e:
            print(f"ERROR: Conexión RPC fallida o error inesperado: {e}")
            socketio.emit('control_response', {"status": "error", "message": f"Error de conexión RPC o inesperado: {e}. Asegúrate que Mininet RPC está corriendo."})

    elif command == 'run_iperf_traffic':
        host_name = data.get('host_name')
        target_ip = data.get('target_ip')
        bandwidth_mbps = data.get('bandwidth_mbps')
        duration_s = data.get('duration_s', 60) # Por defecto 60 segundos
        print(f"RPC: Recibida petición para run_iperf_traffic: {host_name} a {target_ip} con {bandwidth_mbps}Mbps")
        try:
            success = mininet_rpc.run_host_iperf(host_name, target_ip, bandwidth_mbps, duration_s)
            if success:
                print(f"RPC: Tráfico iperf iniciado en {host_name}.")
                socketio.emit('control_response', {"status": "success", "message": f"Tráfico iperf iniciado en {host_name}."})
            else:
                print(f"RPC: Fallo al iniciar tráfico iperf en {host_name}.")
                socketio.emit('control_response', {"status": "error", "message": f"Fallo al iniciar tráfico iperf en {host_name}."})
        except xmlrpc.client.Fault as e:
            print(f"RPC ERROR: Fallo XML-RPC al llamar a run_iperf_traffic: {e}")
            socketio.emit('control_response', {"status": "error", "message": f"Error RPC al iniciar iperf: {e}"})
        except Exception as e:
            print(f"ERROR: Conexión RPC fallida o error inesperado: {e}")
            socketio.emit('control_response', {"status": "error", "message": f"Error de conexión RPC o inesperado: {e}. Asegúrate que Mininet RPC está corriendo."})

    elif command == 'save_snapshot':
        global network_data # Acceder a la variable global
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_snapshot_{timestamp}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(network_data, f, indent=4)
            print(f"Instantánea de la red guardada en: {filename}")
            socketio.emit('control_response', {"status": "success", "command": "save_snapshot", "message": f"Instantánea guardada como {filename}"})
        except Exception as e:
            print(f"Error al guardar la instantánea: {e}")
            socketio.emit('control_response', {"status": "error", "command": "save_snapshot", "message": f"Error al guardar la instantánea: {e}"})
    else:
        print(f"Comando desconocido: {command}")
        


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)