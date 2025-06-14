# topologia_8_switches.py (tu segundo script de topología)

from mininet.node import OVSKernelSwitch, RemoteController, Host
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import logging
import time
import random
import threading

# IMPORTAR el módulo RPC y sus variables globales
import mininet_rpc_server as rpc_api_module

logging.basicConfig(level=logging.INFO)

def topologia_8_switches(): # Cambiado el nombre de la función para ser más específico
    net = Mininet(topo=None, build=False, ipBase='212.18.0.0/24')

    info('*** Agregar controladores\n')
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    info('*** Agregar switches\n')
    switches = []
    for i in range(1, 9): # De s1 a s8
        sw = net.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13')
        switches.append(sw)
    s1, s2, s3, s4, s5, s6, s7, s8 = switches # Para facilitar la referenciación

    info('*** Agregar hosts\n')
    hosts = []
    # Diccionario para mapear hosts a switches
    host_to_switch_map = {
        'h1': s1, 'h2': s2, 'h3': s3, 'h4': s5, 
        'h5': s6, 'h6': s7, 'h7': s1, 'h8': s2, 'h9': s8
    }
    for i in range(1, 10):
        h = net.addHost(f'h{i}', cls=Host, ip=f'212.18.0.{i}', defaultRoute=None)
        hosts.append(h)
    h1, h2, h3, h4, h5, h6, h7, h8, h9 = hosts # Para facilitar la referenciación

    info('* Agregar conexiones\n')
    # Parámetros por defecto para los enlaces
    link_params = {'bw': 100, 'delay': '2ms', 'loss': 0}

    # Enlaces Host-Switch
    net.addLink(h1, s1, cls=TCLink, **link_params)
    net.addLink(h2, s2, cls=TCLink, **link_params)
    net.addLink(h3, s3, cls=TCLink, **link_params)
    net.addLink(h4, s5, cls=TCLink, **link_params)
    net.addLink(h5, s6, cls=TCLink, **link_params)
    net.addLink(h6, s7, cls=TCLink, **link_params)
    net.addLink(h7, s1, cls=TCLink, **link_params)
    net.addLink(h8, s2, cls=TCLink, **link_params)
    net.addLink(h9, s8, cls=TCLink, **link_params)

    # Enlaces Switch-Switch
    # Almacenar en el mapa global_links_by_dpid_pair DESPUÉS de net.build()
    # net.addLink devuelve el objeto Link. Lo usaremos más tarde.
    net.addLink(s1, s2, cls=TCLink, **link_params)
    net.addLink(s2, s3, cls=TCLink, **link_params)
    net.addLink(s3, s5, cls=TCLink, **link_params)
    net.addLink(s5, s6, cls=TCLink, **link_params)
    net.addLink(s6, s7, cls=TCLink, **link_params)
    net.addLink(s7, s1, cls=TCLink, **link_params) 
    net.addLink(s5, s8, cls=TCLink, **link_params)
    net.addLink(s8, s2, cls=TCLink, **link_params) 

    net.addLink(s1, s4, cls=TCLink, **link_params)
    net.addLink(s2, s4, cls=TCLink, **link_params)
    net.addLink(s3, s4, cls=TCLink, **link_params)
    net.addLink(s5, s4, cls=TCLink, **link_params)
    net.addLink(s6, s4, cls=TCLink, **link_params)
    net.addLink(s7, s4, cls=TCLink, **link_params)
    net.addLink(s8, s3, cls=TCLink, **link_params) 

    info('* Iniciando la red\n')
    net.build() # <-- ¡IMPORTANTE! Aquí se asignan los DPIDs y las interfaces.

    # --- LLENAR LAS VARIABLES GLOBALES DESPUÉS DE net.build() ---
    rpc_api_module.GLOBAL_NET = net # Guardar la instancia de la red

    rpc_api_module.GLOBAL_SWITCHES_BY_DPID.clear()
    for sw in net.switches:
        if sw.dpid is not None:
            rpc_api_module.GLOBAL_SWITCHES_BY_DPID[int(sw.dpid, 16)] = sw

    rpc_api_module.GLOBAL_HOSTS_BY_NAME.clear()
    for h in net.hosts:
        rpc_api_module.GLOBAL_HOSTS_BY_NAME[h.name] = h

    rpc_api_module.GLOBAL_LINKS_BY_DPID_PAIR.clear()
    for link in net.links:
        if (isinstance(link.intf1.node, OVSKernelSwitch) and
            isinstance(link.intf2.node, OVSKernelSwitch)):
            sw1 = link.intf1.node
            sw2 = link.intf2.node
            if sw1.dpid is not None and sw2.dpid is not None:
                dpid1 = int(sw1.dpid, 16)
                dpid2 = int(sw2.dpid, 16)
                sorted_dpids = tuple(sorted((dpid1, dpid2)))
                rpc_api_module.GLOBAL_LINKS_BY_DPID_PAIR[sorted_dpids] = link

    info("*** Mapas globales llenos para RPC después de build().\n")
    # --- FIN DE LLENADO DE VARIABLES GLOBALES ---


    info('* Iniciando los controladores\n')
    c0.start()  # Inicia el controlador Ryu

    info('* Iniciando los switches\n')
    for sw in switches: # Conecta todos los switches al controlador
        sw.start([c0])


    info('* Switches y hosts configurados\n')
    dumpNodeConnections(net.hosts)
    dumpNodeConnections(net.switches)

    logging.info("Esperando 5 segundos antes de iniciar el tráfico...")
    time.sleep(5)

    logging.info("Generando tráfico intenso con iperf...")

    # Iniciar el servidor iperf en h1
    h1 = rpc_api_module.GLOBAL_HOSTS_BY_NAME.get('h1')
    if h1:
        h1.cmd('iperf -s -u -i 1 &') # El & es importante para que no bloquee
        logging.info("Servidor iperf iniciado en h1 (212.18.0.1)")

    # Generar tráfico desde otros hosts hacia h1
    for host_name in ['h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9']: # Asegúrate de incluir h9
        if host_name in rpc_api_module.GLOBAL_HOSTS_BY_NAME: # Verificar si el host existe
            host = rpc_api_module.GLOBAL_HOSTS_BY_NAME[host_name]
            bandwidth = random.randint(50, 100)
            logging.info(f"{host_name} enviando tráfico de {bandwidth} Mbps a h1")
            host.cmd(f'iperf -c 212.18.0.1 -u -b {bandwidth}M -t 60 -P 30 &')
        else:
            logging.warning(f"Host {host_name} no encontrado en GLOBAL_HOSTS_BY_NAME, omitiendo tráfico.")

    logging.info("Accediendo a la CLI de Mininet. Puede interactuar con la red...")
    CLI(net)

    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    # Inicia el servidor RPC en un hilo separado
    rpc_thread = threading.Thread(target=rpc_api_module.start_mininet_rpc_server)
    rpc_thread.daemon = True 
    rpc_thread.start()

    topologia_8_switches() # Llama a la función de topología