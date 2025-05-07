from mininet.node import OVSKernelSwitch, RemoteController, Host
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def topologia_20():
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

    info('*** Interconectar switches en anillo\n')
    for i in range(len(switches)):
        sw1 = switches[i]
        sw2 = switches[(i+1) % len(switches)]
        net.addLink(sw1, sw2, cls=TCLink, **link_conf)

    info('*** Iniciar la red\n')
    net.build()
    c0.start()
    for sw in switches:
        sw.start([c0])

    info('*** Topología lista. Enlace hosts-switch y anillo de switches.\n')
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topologia_20()
