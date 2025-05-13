from mininet.node import OVSKernelSwitch, RemoteController, Host
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import threading 
from xmlrpc.server import SimpleXMLRPCServer
import time

# Variables globales para almacenar la instancia de Mininet y los mapeos de enlaces/hosts
# Necesitamos que sean globales o pasadas al objeto RPC para que las funciones puedan acceder a ellas
global_net = None
global_switches_by_dpid = {}
global_links_by_dpid_pair = {} # Almacena enlaces por tupla (dpid1, dpid2) 
global_hosts_by_name = {}


class MininetRpcApi:
    def __init__(self, net_instance, switches_map, links_map, hosts_map):
        self.net = net_instance
        self.switches = switches_map
        self.links = links_map # Mapeo de (dpid1, dpid2) a objeto de enlace
        self.hosts = hosts_map # Mapeo de nombre_host a objeto de host

    # Función RPC para cambiar parámetros de un enlace (Delay, Loss, BW)
    def set_link_param(self, src_dpid, dst_dpid, param_name, value):
        info(f"RPC: Solicitud para cambiar {param_name} en enlace {src_dpid}-{dst_dpid} a {value}\n")
        
        # Normalizar la tupla de DPIDs para buscar en el mapeo
        sorted_dpids = tuple(sorted((src_dpid, dst_dpid)))
        link = self.links.get(sorted_dpids)

        if not link:
            info(f"RPC ERROR: Enlace entre {src_dpid}-{dst_dpid} no encontrado.\n")
            return False

        try:
            if param_name == "delay":
                link.delay = f"{value}ms" # Asigna el valor de string directamente para TCLink
                info(f"RPC: Retardo del enlace {src_dpid}-{dst_dpid} establecido a {value}ms\n")
            elif param_name == "loss":
                link.loss = float(value)
                info(f"RPC: Pérdida del enlace {src_dpid}-{dst_dpid} establecida a {value}%\n")
            elif param_name == "bw":
                link.bw = float(value) # Ancho de banda en Mbps
                info(f"RPC: Ancho de banda del enlace {src_dpid}-{dst_dpid} establecido a {value}Mbps\n")
            else:
                info(f"RPC ERROR: Parámetro desconocido: {param_name}\n")
                return False
            
            return True
        except Exception as e:
            info(f"RPC ERROR: Fallo al establecer {param_name} en enlace {src_dpid}-{dst_dpid}: {e}\n")
            return False

    # Función RPC para ejecutar un comando iperf en un host
    # ¡! El servidor iperf esté corriendo en el destino antes de llamar al cliente.
    def run_host_iperf(self, host_name, target_ip, bandwidth_mbps, duration_s):
        info(f"RPC: Solicitud para ejecutar iperf en {host_name} a {target_ip} con {bandwidth_mbps}Mbps por {duration_s}s\n")
        host = self.hosts.get(host_name)
        if not host:
            info(f"RPC ERROR: Host {host_name} no encontrado.\n")
            return False
        
        try:
            # Ejecutar iperf como cliente UDP. El & al final lo ejecuta en segundo plano.
            command = f"iperf -c {target_ip} -u -b {bandwidth_mbps}M -t {duration_s} &"
            host.cmd(command)
            info(f"RPC: Comando '{command}' ejecutado en {host_name}.\n")
            return True
        except Exception as e:
            info(f"RPC ERROR: Fallo al ejecutar iperf en {host_name}: {e}\n")
            return False

def topologia_20():
    global global_net, global_switches_by_dpid, global_links_by_dpid_pair, global_hosts_by_name

    net = Mininet(topo=None, build=False, ipBase='212.18.0.0/24')
    info('*** Agregar controlador remoto\n')
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    info('*** Crear 20 switches y 20 hosts\n')
    switches = []
    hosts = []
    for i in range(1, 21):
        sw = net.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13')
        h  = net.addHost  (f'h{i}', cls=Host, ip=f'212.18.0.{i}', defaultRoute=None)
        switches.append(sw)
        hosts.append(h)

    info('*** Conectar cada host a su switch con TCLink (100Mbps, 2ms, 0% loss)\n')
    link_conf = dict(bw=100, delay='2ms', loss=0)
    for sw, h in zip(switches, hosts):
        net.addLink(h, sw, cls=TCLink, **link_conf)

    info('*** Interconectar switches en anillo básico\n')
    for i in range(len(switches)):
        sw1 = switches[i]
        sw2 = switches[(i+1) % len(switches)]
        link = net.addLink(sw1, sw2, cls=TCLink, **link_conf)
        sorted_dpids = tuple(sorted((sw1.dpid, sw2.dpid))) 
        global_links_by_dpid_pair[sorted_dpids] = link

    info('*** Añadir enlaces adicionales (chords) para múltiples caminos\n')
    # Enlaces no consecutivos para generar rutas alternativas
    extra_pairs = [
        (7, 16), (3, 10), (5, 15), (2, 18), (11, 20), (4, 14), (8, 13)
    ]
    for a, b in extra_pairs:
        info(f'    - Añadiendo enlace entre s{a} <-> s{b}\n')
        net.addLink(switches[a-1], switches[b-1], cls=TCLink, **link_conf)

    info('*** Iniciar la red\n')
    net.build()



    info("*** Llenando mapas globales con DPIDs, enlaces y hosts despu\u00E9s de net.build()...\n")

    # Llenar mapa de switches por DPID (ENTEROS)
    global_switches_by_dpid.clear() # Limpiar por si acaso
    for sw in net.switches:
        if sw.dpid is not None:
             # Convertir DPID a INT para usar como clave
             global_switches_by_dpid[int(sw.dpid, 16)] = sw
             info(f"  Mapeado switch {sw.name} con DPID: {sw.dpid} (ENTERO: {int(sw.dpid, 16)})\n")
        else:
             info(f"  ADVERTENCIA: Switch {sw.name} no tiene DPID asignado despu\u00E9s de build.\n")

    # Llenar mapa de hosts por nombre
    global_hosts_by_name.clear() # Limpiar por si acaso
    for h in net.hosts:
        global_hosts_by_name[h.name] = h # Mapear nombre a objeto host
        info(f"  Mapeado host {h.name} con IP: {h.IP()}\n")


    # Llenar mapa de enlaces ENTRE SWITCHES por par de DPIDs ENTEROS
    global_links_by_dpid_pair.clear() # Limpiar por si acaso
    for link in net.links:
        # Solo considerar enlaces entre dos switches
        if (isinstance(link.intf1.node, OVSKernelSwitch) and
            isinstance(link.intf2.node, OVSKernelSwitch)):
             sw1 = link.intf1.node
             sw2 = link.intf2.node
             # Asegurarse de que ambos switches tienen DPID asignado (ya deber\u00EDa ser el caso aqu\u00ED)
             if sw1.dpid is not None and sw2.dpid is not None:
                  dpid1 = int(sw1.dpid, 16) # Convertir DPID a INT
                  dpid2 = int(sw2.dpid, 16) # Convertir DPID a INT
                  sorted_dpids = tuple(sorted((dpid1, dpid2))) # Tupla de ENTEROS ordenada
                  global_links_by_dpid_pair[sorted_dpids] = link # Usar tupla de ENTEROS como clave
                  info(f"  Mapeado enlace entre {sw1.name} (DPID: {dpid1}) y {sw2.name} (DPID: {dpid2})\n")
             else:
                  info(f"  ADVERTENCIA: Enlace entre {sw1.name} y {sw2.name} ignorado en el mapa RPC (uno o ambos sin DPID).\n")
        # Puedes a\u00F1adir l\u00F3gica aqu\u00ED si tambi\u00E9n quieres controlar enlaces host-switch por nombre/IP
        # elif isinstance(link.intf1.node, Host) and isinstance(link.intf2.node, OVSKernelSwitch):
        #     host = link.intf1.node
        #     sw = link.intf2.node
        #     # Mapeo host-switch si es necesario para RPC


    info("*** Mapas globales llenos despu\u00E9s de build().\n")




    c0.start()
    for sw in switches:
        sw.start([c0])

    info('*** Topología lista: hosts-switch, anillo + enlaces adicionales.\n')

    # Iniciar el servidor iperf en h1 (ejemplo para probar carga)
    h1 = global_hosts_by_name.get('h1')
    if h1:
        info("Iniciando servidor iperf en h1...\n")
        h1.cmd('iperf -s -u -i 1 > /dev/null &') # Ejecutar en segundo plano, redirigir salida
        info("Servidor iperf iniciado en h1 (212.18.0.1)\n")

        info("\n*** Contenido de global_switches_by_dpid:\n")
    if global_switches_by_dpid:
        for dpid, sw in sorted(global_switches_by_dpid.items()):
             info(f"    {dpid}: {sw.name}\n")
    else:
        info("    (Mapa de switches vacío)\n")



    # Imprimir el mapa de enlaces por par de DPIDs
    info("\n*** Contenido de global_links_by_dpid_pair:\n")
    if global_links_by_dpid_pair:
        for dpid_pair, link_obj in sorted(global_links_by_dpid_pair.items()):
            # Intentar obtener los nombres de los switches para una salida m\u00E1s legible
            sw1_name = global_switches_by_dpid.get(dpid_pair[0])
            sw2_name = global_switches_by_dpid.get(dpid_pair[1])

            sw1_info = sw1_name.name if sw1_name else dpid_pair[0]
            sw2_info = sw2_name.name if sw2_name else dpid_pair[1]
            
            info(f"    {dpid_pair}: Enlace entre {sw1_info} y {sw2_info}\n")
    else:
        info("    (Mapa de enlaces vac\u00EDo)\n")

    # Imprimir el mapa de hosts por nombre
    info("\n*** Contenido de global_hosts_by_name:\n")
    if global_hosts_by_name:
         for name, host_obj in sorted(global_hosts_by_name.items()):
              info(f"    {name}: {host_obj.IP()}\n")
    else:
         info("    (Mapa de hosts vac\u00EDo)\n")

    info("*** Fin de la impresión de mapas ***\n")

    CLI(net)
    net.stop()

def start_mininet_rpc_server():
    # Crea una instancia de la API con las referencias globales a la red y los mapeos
    rpc_api = MininetRpcApi(global_net, global_switches_by_dpid, global_links_by_dpid_pair, global_hosts_by_name)

    # Crea el servidor XML-RPC
    # Puedes cambiar el puerto si 8000 ya está en uso
    server_address = ("127.0.0.1", 8000) 
    server = SimpleXMLRPCServer(server_address, logRequests=False, allow_none=True) # logRequests=False para menos verbosidad
    server.register_instance(rpc_api) # Registra la instancia de la clase para que sus métodos sean accesibles

    info(f"*** Servidor Mininet RPC escuchando en {server_address[0]}:{server_address[1]}...\n")
    server.serve_forever() # Inicia el servidor RPC, se ejecuta hasta que se detenga



if __name__ == '__main__':
    setLogLevel('info')

    # Inicia el servidor RPC en un hilo separado
    rpc_thread = threading.Thread(target=start_mininet_rpc_server)
    rpc_thread.daemon = True # El hilo RPC se detendrá cuando el programa principal termine
    rpc_thread.start()

    topologia_20()
