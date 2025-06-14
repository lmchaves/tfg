# mininet_rpc_server.py

from xmlrpc.server import SimpleXMLRPCServer
import threading
import time
from mininet.log import info
from mininet.node import OVSKernelSwitch, Host # Necesario para isinstance

# Las variables globales que se llenarán desde la topología
# IMPORTANTE: Estas deben ser llenadas DESPUÉS de que la red Mininet se construya.
GLOBAL_NET = None
GLOBAL_SWITCHES_BY_DPID = {} # {int_dpid: switch_obj}
GLOBAL_LINKS_BY_DPID_PAIR = {} # {(int_dpid1, int_dpid2): link_obj}
GLOBAL_HOSTS_BY_NAME = {} # {host_name: host_obj}

class MininetRpcApi:
    """
    Clase que expone la API RPC para interactuar con la red Mininet.
    Debe ser instanciada con las referencias a los objetos Mininet
    después de que la red se haya construido.
    """
    def __init__(self, net_instance, switches_map, links_map, hosts_map):
        self.net = net_instance
        self.switches = switches_map # Mapeo de DPID a objeto switch
        self.links = links_map       # Mapeo de (dpid1, dpid2) a objeto de enlace
        self.hosts = hosts_map       # Mapeo de nombre_host a objeto de host

    # Función RPC para cambiar parámetros de un enlace (Delay, Loss, BW)
    def set_link_param(self, src_dpid, dst_dpid, param_name, value):
        info(f"RPC: Solicitud para cambiar {param_name} en enlace {src_dpid}-{dst_dpid} a {value}\n")
        
        # Normalizar la tupla de DPIDs para buscar en el mapeo (siempre ordenada)
        sorted_dpids = tuple(sorted((src_dpid, dst_dpid)))
        link = self.links.get(sorted_dpids)

        if not link:
            info(f"RPC ERROR: Enlace entre {src_dpid}-{dst_dpid} no encontrado en el mapa de RPC.\n")
            # Intentar buscar en la otra dirección si la topología tiene enlaces unidireccionales o el mapa no los normaliza bien
            # Aunque tu global_links_by_dpid_pair ya usa sorted((dpid1, dpid2))
            return False

        try:
            intf1 = link.intf1 # Interfaz del primer nodo en el enlace
            intf2 = link.intf2 # Interfaz del segundo nodo en el enlace

            # Identificar la interfaz que pertenece al src_dpid
            src_intf = None
            if int(intf1.node.dpid, 16) == src_dpid:
                src_intf = intf1
            elif int(intf2.node.dpid, 16) == src_dpid:
                 src_intf = intf2
            
            if not src_intf:
                 info(f"RPC ERROR: No se pudo identificar la interfaz de origen para DPID {src_dpid} en el enlace {src_dpid}-{dst_dpid}.")
                 return False

            src_node = src_intf.node # Obtener el nodo de Mininet directamente de la interfaz

            # Los qdiscs htb y netem a menudo tienen handles fijos para TCLink
            htb_handle = "5:"
            netem_handle = "10:" # Handle del qdisc netem
            tbf_handle = "6:"    # Handle para el tbf si se usa con HTB

            command = None 

            if param_name == "delay":
                # Asegurarse de que el qdisc netem exista o se añada
                # Un "change" fallará si no existe, un "add" fallará si existe.
                # TCLink ya lo añade, así que "change" es lo correcto.
                command = f"tc qdisc change dev {src_intf.name} parent {netem_handle} netem delay {value}ms"
                info(f"RPC: Ejecutando tc command en {src_node.name} ({src_intf.name}): {command}")
                output = src_node.cmd(command)
                info(f"RPC: Output tc delay: {output.strip()}")

            elif param_name == "loss":
                 command = f"tc qdisc change dev {src_intf.name} parent {netem_handle} netem loss {value}%"
                 info(f"RPC: Ejecutando tc command en {src_node.name} ({src_intf.name}): {command}")
                 output = src_node.cmd(command)
                 info(f"RPC: Output tc loss: {output.strip()}")

            elif param_name == "bw":
                 # El comando TCLink ya crea un HTB qdisc en el handle 5:
                 # y un TBF qdisc con parent 5:1 y handle 6: para el bw
                 # (esto es un detalle interno de TCLink, puede variar)
                 rate_bit = float(value) * 1e6 # Convertir Mbps a bits/seg
                 burst_bytes = min(rate_bit / 8, 1000) # Burst no debe ser mayor que el rate
                 limit_bytes = int(burst_bytes * 10)   # Un límite razonable, ej. 10 veces el burst

                 # Asegurarse de que el parent 5:1 esté bien, o usar el qdisc root de HTB
                 # Si TCLink ya puso un TBF, 'change' es el comando correcto.
                 # Si no, 'add' sería necesario. Para simplificar, asumimos que TCLink lo establece.
                 command = f"tc qdisc change dev {src_intf.name} parent {htb_handle} handle {tbf_handle} tbf rate {rate_bit:.0f}bit burst {burst_bytes:.0f} limit {limit_bytes:.0f}"
                 info(f"RPC: Ejecutando tc command en {src_node.name} ({src_intf.name}): {command}")
                 output = src_node.cmd(command)
                 info(f"RPC: Output tc bw: {output.strip()}")

            else:
                info(f"RPC ERROR: Parámetro desconocido: {param_name}. No se ejecutó comando tc.")
                return False

            # Evaluar si el comando tc fue exitoso basándose en su salida
            # tc suele imprimir "RTNETLINK answers: No such file or directory" o "device busy" si falla
            if "RTNETLINK answers" in output or "device busy" in output or "No such device" in output:
                info(f"RPC WARNING: Comando tc en {src_node.name} para {src_intf.name} falló (output: {output.strip()}).")
                return False
            return True # Asume éxito si no hay mensajes de error conocidos de tc

        except Exception as e:
            info(f"RPC ERROR: Fallo GENERAL al intentar establecer {param_name} en enlace {src_dpid}-{dst_dpid} vía tc command: {e}\n")
            return False

    # Función RPC para ejecutar un comando iperf en un host
    def run_host_iperf(self, host_name, target_ip, bandwidth_mbps, duration_s):
        info(f"RPC: Solicitud para ejecutar iperf en {host_name} a {target_ip} con {bandwidth_mbps}Mbps por {duration_s}s\n")
        host = self.hosts.get(host_name)
        if not host:
            info(f"RPC ERROR: Host {host_name} no encontrado.\n")
            return False
        
        try:
            # Ejecutar iperf como cliente UDP. El & al final lo ejecuta en segundo plano.
            # Puedes añadir -P para múltiples flujos, -i para reportes, etc.
            command = f"iperf -c {target_ip} -u -b {bandwidth_mbps}M -t {duration_s} &"
            host.cmd(command)
            info(f"RPC: Comando '{command}' ejecutado en {host_name}.\n")
            return True
        except Exception as e:
            info(f"RPC ERROR: Fallo al ejecutar iperf en {host_name}: {e}\n")
            return False

def start_mininet_rpc_server():
    """
    Inicia el servidor RPC para Mininet en un hilo separado.
    Debe llamarse DESPUÉS de que la topología haya sido construida
    y las variables globales (GLOBAL_NET, etc.) se hayan llenado.
    """
    # Crea una instancia de la API con las referencias globales a la red y los mapeos
    rpc_api = MininetRpcApi(GLOBAL_NET, GLOBAL_SWITCHES_BY_DPID, GLOBAL_LINKS_BY_DPID_PAIR, GLOBAL_HOSTS_BY_NAME)

    # Crea el servidor XML-RPC
    server_address = ("127.0.0.1", 8000) 
    server = SimpleXMLRPCServer(server_address, logRequests=False, allow_none=True) 
    server.register_instance(rpc_api) 

    info(f"*** Servidor Mininet RPC escuchando en {server_address[0]}:{server_address[1]}...\n")
    server.serve_forever() # Inicia el servidor RPC, se ejecuta hasta que se detenga