#!/usr/bin/python

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.node import OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.node import Host
from mininet.util import dumpNodeConnections


import logging

logging.basicConfig(level=logging.INFO)

def topologia():
    # Crear red Mininet con configuración básica
    net = Mininet(topo=None, build=False, ipBase='212.18.0.0/24')

    # ==========================
    # Configuración del Controlador
    # ==========================
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    # ==========================
    # Configuración de Switches
    # ==========================
    logging.info('*** Agregar switches\n')
    switches = {
        's1': net.addSwitch('s1', cls=OVSKernelSwitch),
        's2': net.addSwitch('s2', cls=OVSKernelSwitch),
        's3': net.addSwitch('s3', cls=OVSKernelSwitch),
        's4': net.addSwitch('s4', cls=OVSKernelSwitch),
        's5': net.addSwitch('s5', cls=OVSKernelSwitch),
        's6': net.addSwitch('s6', cls=OVSKernelSwitch),
        's7': net.addSwitch('s7', cls=OVSKernelSwitch)
    }



    # ==========================
    # Crear 8 hosts (h1-h8)
    # ==========================
    logging.info('*** Agregar hosts\n')
    hosts = {
        'h1': net.addHost('h1', cls=Host, ip='212.18.0.1', defaultRoute=None),
        'h2': net.addHost('h2', cls=Host, ip='212.18.0.2', defaultRoute=None),
        'h3': net.addHost('h3', cls=Host, ip='212.18.0.3', defaultRoute=None),
        'h4': net.addHost('h4', cls=Host, ip='212.18.0.4', defaultRoute=None),
        'h5': net.addHost('h5', cls=Host, ip='212.18.0.5', defaultRoute=None),
        'h6': net.addHost('h6', cls=Host, ip='212.18.0.6', defaultRoute=None),
        'h7': net.addHost('h7', cls=Host, ip='212.18.0.7', defaultRoute=None),
        'h8': net.addHost('h8', cls=Host, ip='212.18.0.8', defaultRoute=None)
    }

    # ==========================
    # Configuración de Enlaces
    # ==========================
    # Parámetros de QoS para los enlaces
    link_params = {'bw': 10, 'delay': '5ms', 'loss': 0}

    # ==========================
    # Conectar hosts a switches
    # ==========================

    net.addLink(hosts['h1'], switches['s1'], cls=TCLink, **link_params)# h1 -> s1
    net.addLink(hosts['h2'], switches['s1'], cls=TCLink, **link_params)# h2 -> s1
    net.addLink(hosts['h3'], switches['s2'], cls=TCLink, **link_params)  # h3 -> s2
    net.addLink(hosts['h4'], switches['s2'], cls=TCLink, **link_params)# h4 -> s2
    net.addLink(hosts['h5'], switches['s5'], cls=TCLink, **link_params) # h5 -> s5
    net.addLink(hosts['h6'], switches['s5'], cls=TCLink, **link_params)# h6 -> s5
    net.addLink(hosts['h7'], switches['s4'], cls=TCLink, **link_params)# h7 -> s4
    net.addLink(hosts['h8'], switches['s4'], cls=TCLink, **link_params) # h8 -> s4



    # Conectar switches entre sí
    net.addLink(switches['s1'], switches['s2'], cls=TCLink, **link_params)
    net.addLink(switches['s1'], switches['s4'], cls=TCLink, **link_params)
    net.addLink(switches['s2'], switches['s3'], cls=TCLink, **link_params)
    net.addLink(switches['s2'], switches['s4'], cls=TCLink, **link_params)
    net.addLink(switches['s3'], switches['s4'], cls=TCLink, **link_params)
    net.addLink(switches['s3'], switches['s5'], cls=TCLink, **link_params)
    net.addLink(switches['s3'], switches['s6'], cls=TCLink, **link_params)
    net.addLink(switches['s4'], switches['s5'], cls=TCLink, **link_params)
    net.addLink(switches['s4'], switches['s6'], cls=TCLink, **link_params)
    net.addLink(switches['s5'], switches['s6'], cls=TCLink, **link_params)

    # ======================
    # Iniciar la red
    # ======================
    logging.info("Iniciando la construcción de la red...")
    net.build()

    # Iniciar el controlador
    logging.info("Iniciando el controlador c0...")
    c0.start()

    # Iniciar los switches con el controlador
    logging.info("Iniciando los switches con el controlador...")
    for switch_name, switch in switches.items():
        logging.info(f"Iniciando switch {switch_name}...")
        switch.start([c0])


    logging.info( '* Switches y hosts configurados')
    dumpNodeConnections( net.hosts ) #FUNCION QUE PERMITE VER LAS CONEXIONES DE LOS HOST
    dumpNodeConnections( net.switches ) #FUNCION QUE PERMITE VER LAS CONEXIONES DE LOS SWITCHES

    # Generar tráfico entre h1 y h8
    logging.info("Generando tráfico entre h1 y h8...")

    net.pingAll() #FUNCION QUE REALIZA PING ENTRE TODOS LOS HOST 


    h1 = net.get('h1')
    h8 = net.get('h8')
    logging.info("Ejecutando iperf en h1 (servidor)...")
    h1.cmd('iperf -s &')  # Servidor en h1
    logging.info("Ejecutando iperf en h8 (cliente)...")
    h8.cmd('iperf -c 10.0.0.1 -t 60 &')  # Cliente en h8

    logging.info("Realizadno ping a h8 (cliente)...")
    h1.cmd('ping -i 0.2 10.0.0.8 &')

    # Mantener la CLI abierta
    logging.info("Accediendo a la CLI de Mininet. Puede interactuar con la red...")
    CLI(net)

    # Detener la red
    logging.info("Deteniendo la red...")
    net.stop()

if __name__ == '__main__':
    # Establecer el nivel de log para ocultar los mensajes 'packet in'
    topologia()
