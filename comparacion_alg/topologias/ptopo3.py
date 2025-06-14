from mininet.node import OVSKernelSwitch
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.node import Host
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
import logging
import time
import random

logging.basicConfig(level=logging.INFO)

def topologia():
    net = Mininet(topo=None, build=False, ipBase='212.18.0.0/24')

    info('*** Agregar controladores\n')
    # Agrega un controlador remoto que apunte a Ryu (localhost:6653)
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    info('*** Agregar switches\n')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch, protocols='OpenFlow13')
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch, protocols='OpenFlow13')
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch, protocols='OpenFlow13')
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch, protocols='OpenFlow13') 
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch, protocols='OpenFlow13')
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch, protocols='OpenFlow13')
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch, protocols='OpenFlow13')
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch, protocols='OpenFlow13')


    info('*** Agregar hosts\n')
    h1 = net.addHost('h1', cls=Host, ip='212.18.0.1', defaultRoute=None) # En s1
    h2 = net.addHost('h2', cls=Host, ip='212.18.0.2', defaultRoute=None) # En s2
    h3 = net.addHost('h3', cls=Host, ip='212.18.0.3', defaultRoute=None) # En s3
    h4 = net.addHost('h4', cls=Host, ip='212.18.0.4', defaultRoute=None) # En s5
    h5 = net.addHost('h5', cls=Host, ip='212.18.0.5', defaultRoute=None) # En s6
    h6 = net.addHost('h6', cls=Host, ip='212.18.0.6', defaultRoute=None) # En s7
    h7 = net.addHost('h7', cls=Host, ip='212.18.0.7', defaultRoute=None) # En s1 
    h8 = net.addHost('h8', cls=Host, ip='212.18.0.8', defaultRoute=None) # En s2 
    h9 = net.addHost('h9', cls=Host, ip='212.18.0.9', defaultRoute=None) # En s8 



    info('* Agregar conexiones\n')
    bd = {'bw': 100, 'delay': '2ms', 'loss': 0}

    # Enlaces Host-Switch
    net.addLink(h1, s1, cls=TCLink, **bd)
    net.addLink(h2, s2, cls=TCLink, **bd)
    net.addLink(h3, s3, cls=TCLink, **bd)
    net.addLink(h4, s5, cls=TCLink, **bd)
    net.addLink(h5, s6, cls=TCLink, **bd)
    net.addLink(h6, s7, cls=TCLink, **bd)
    net.addLink(h7, s1, cls=TCLink, **bd)
    net.addLink(h8, s2, cls=TCLink, **bd)
    net.addLink(h9, s8, cls=TCLink, **bd)


 
    net.addLink(s1, s2, cls=TCLink, **bd)
    net.addLink(s2, s3, cls=TCLink, **bd)
    net.addLink(s3, s5, cls=TCLink, **bd)
    net.addLink(s5, s6, cls=TCLink, **bd)
    net.addLink(s6, s7, cls=TCLink, **bd)
    net.addLink(s7, s1, cls=TCLink, **bd) 
    net.addLink(s5, s8, cls=TCLink, **bd)
    net.addLink(s8, s2, cls=TCLink, **bd) 

    net.addLink(s1, s4, cls=TCLink, **bd)
    net.addLink(s2, s4, cls=TCLink, **bd)
    net.addLink(s3, s4, cls=TCLink, **bd)
    net.addLink(s5, s4, cls=TCLink, **bd)
    net.addLink(s6, s4, cls=TCLink, **bd)
    net.addLink(s7, s4, cls=TCLink, **bd)
    net.addLink(s8, s3, cls=TCLink, **bd) 

    

    info('* Iniciando la red\n')
    net.build()

    info('* Iniciando los controladores\n')
    c0.start()  # Inicia el controlador Ryu

    info('* Iniciando los switches\n')
    # Conecta los switches al controlador Ryu
    net.get('s1').start([c0])
    net.get('s2').start([c0])
    net.get('s3').start([c0])
    net.get('s4').start([c0])
    net.get('s5').start([c0])
    net.get('s6').start([c0])
    net.get('s7').start([c0])
    net.get('s8').start([c0])


    info('* Switches y hosts configurados\n')
    dumpNodeConnections(net.hosts)
    dumpNodeConnections(net.switches)

    
    # Esperar antes de generar tráfico
    logging.info("Esperando 5 segundos antes de iniciar el tráfico...")
    time.sleep(5)

    logging.info("Generando tráfico intenso con iperf...")

    # Iniciar el servidor iperf en h1
    h1.cmd('iperf -s -u -i 1 &')

    # Generar tráfico desde todos los hosts hacia h1 con diferentes anchos de banda
    for host_name in ['h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']:
        host = net.get(host_name)
        bandwidth = random.randint(50, 100)  # Genera un valor aleatorio entre 50M y 100M
        logging.info(f"{host_name} enviando tráfico de {bandwidth} Mbps a h1")
        host.cmd(f'iperf -c 212.18.0.1 -u -b {bandwidth}M -t 60 -P 30 &')

    # Mantener la CLI abierta
    logging.info("Accediendo a la CLI de Mininet. Puede interactuar con la red...")
    CLI(net)

    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topologia()