#!/usr/bin/env python
# Copyright (C) 2016 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import attrgetter
import time
from ryu.base import app_manager
from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.topology.api import get_switch, get_link
from ryu.topology import api as topo_api
import llbaco


class ExtendedMonitor(simple_switch_13.SimpleSwitch13):
    OFP_VERSIONS = [4]  # OpenFlow 1.3 (la versión 4 corresponde a 1.3)

    def __init__(self, *args, **kwargs):
        super(ExtendedMonitor, self).__init__(*args, **kwargs)
        # Diccionario de switches conectados
        self.datapaths = {}
        # Métricas de enlaces: {dpid: {port: {'load': ..., 'packet_loss': ..., 'delay': ...}}}
        self.link_metrics = {}
        # Estadísticas previas para calcular la carga
        self.last_stats = {}
        # Ancho de banda en bits por segundo (ejemplo: 10 Mbps)
        self.bw = 10 * 1e6  
        # Intervalo de monitoreo en segundos
        self.monitor_interval = 1  
        # Para medir el delay usando mensajes echo
        self.echo_timestamps = {}  
        
        self.topology = {'switches': [], 'links': []}  # Topología de la red

        self.max_load = 0.0
        self.max_delay = 0.0
        self.max_packet_loss = 0.0


        # Inicia el hilo de monitoreo
        self.monitor_thread = hub.spawn(self._monitor)

    def get_network_snapshot(self):
        # Actualizar la topología de la red
        self.get_topology_data()

        print("Topology Links:", self.topology['links'])

        # Generar el snapshot de métricas
        snapshot = {}
        for dpid in list(self.datapaths.keys()):  # Copia de las claves antes de iterar
            snapshot[dpid] = {}
            switches = topo_api.get_switch(self, dpid)
            if not switches:
                continue
            switch = switches[0]
            for port in switch.ports:
                port_no = port.port_no
                if port_no == 4294967294:  # Excluir el puerto virtual
                    continue
                snapshot[dpid][port_no] = {
                    'load': self.link_metrics.get(dpid, {}).get(port_no, {}).get('load', 0.0),
                    'packet_loss': self.link_metrics.get(dpid, {}).get(port_no, {}).get('packet_loss', 0.0),
                    'delay': self.link_metrics.get(dpid, {}).get(port_no, {}).get('delay', 0.0)
                }
        print("Snapshot generado:", snapshot) 
        return snapshot

    def get_topology_data(self):
        """
        Obtiene la topología de la red y la almacena en self.topology.
        """
        switches = get_switch(self, None)
        self.topology['switches'] = [switch.dp.id for switch in switches]

        links = get_link(self, None)
        self.topology['links'] = [(link.src.dpid, link.dst.dpid, {'port': link.src.port_no}) for link in links]

    def run_llbaco(self, snapshot):
        """
        Ejecuta LLBACO con los datos de la red.
        Recibe el snapshot como parámetro.
        """
        self.logger.info("Ejecutando LLBACO con la instantánea de métricas: %s", snapshot)

        nodes = self.topology['switches']
        topology_links = self.topology['links']
        delta = 0.5  # Ajusta según lo que prefieras

        # Construir la matriz de costos
        cost_matrix = llbaco.build_cost_matrix(snapshot, nodes, topology_links, delta)

        self.logger.info("Matriz de costos:")
        for row in cost_matrix:
            self.logger.info(row)

        # Ejecutar el algoritmo LLBACO
        best_path, best_cost = llbaco.run_aco_llbaco(
        cost_matrix, iterations=100, colony=50, alpha=1.0, beta=1.0, del_tau=1.0, rho=0.5, high_cost=1000
        )

        self.logger.info("Ruta óptima encontrada: %s con costo %.2f", best_path, best_cost)


    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info("Registrando datapath: %016x", datapath.id)
                self.datapaths[datapath.id] = datapath

                # Inicializar métricas para todos los puertos del switch
                self.link_metrics[datapath.id] = {}
                switches = topo_api.get_switch(self, datapath.id)
                if switches:
                    switch = switches[0]
                    for port in switch.ports:
                        self.link_metrics[datapath.id][port.port_no] = {
                            'load': 0.0,
                            'packet_loss': 0.0,
                            'delay': 0.0
                        }
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info("Eliminando datapath: %016x", datapath.id)
                del self.datapaths[datapath.id]
                if datapath.id in self.link_metrics:
                    del self.link_metrics[datapath.id]

    def _monitor(self):
        while True:
            # Para cada datapath, solicita estadísticas y un echo para delay.
            for dp in self.datapaths.values():
                self._request_stats(dp)
                self._send_echo_request(dp)

            # Generar el snapshot
            snapshot = self.get_network_snapshot()

            # Depuración: Imprimir self.link_metrics
            self.logger.info("Contenido de self.link_metrics:")
            for dpid, ports in self.link_metrics.items():
                self.logger.info("Switch %016x:", dpid)
                for port, metrics in ports.items():
                    self.logger.info("  Port %d: %s", port, metrics)

            # Ejecutar LLBACO con el snapshot generado
            if snapshot:  # Asegurarse de que el snapshot no esté vacío
                self.run_llbaco(snapshot)  # Pasar el snapshot como argumento

            # Esperar antes de la siguiente iteración
            hub.sleep(self.monitor_interval)

    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # Solicita estadísticas de flujo (opcional, según se necesite)
        req = parser.OFPFlowStatsRequest(datapath, table_id=ofproto.OFPTT_ALL, out_port=ofproto.OFPP_ANY)
        datapath.send_msg(req)
        # Solicita estadísticas de puerto
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _send_echo_request(self, datapath):
        parser = datapath.ofproto_parser
        self.echo_timestamps[datapath.id] = time.time()
        datapath.send_msg(parser.OFPEchoRequest(datapath, data=b''))

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def _echo_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        if dpid in self.echo_timestamps:
            rtt = time.time() - self.echo_timestamps[dpid]
            delay = rtt / 2
            # Actualiza el delay para cada puerto del switch (se podría ajustar para hacerlo por puerto)
            if dpid in self.link_metrics:
                for port in self.link_metrics[dpid]:
                    self.link_metrics[dpid][port]['delay'] = delay
            self.logger.info("Delay actualizado para dpid %s: %.3f ms", dpid, delay * 1000)
            del self.echo_timestamps[dpid]

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        if dpid not in self.link_metrics:
            self.link_metrics[dpid] = {}
        if dpid not in self.last_stats:
            self.last_stats[dpid] = {}

        self.logger.info("=== Estadísticas del switch %016x ===", dpid)
        self.logger.info("Port     Load       Delay (ms)  Packet Loss (%)")
        self.logger.info("------   --------   ----------  --------------")

        for stat in body:
            port = stat.port_no

            # Inicializar métricas si el puerto no está registrado
            if port not in self.link_metrics[dpid]:
                self.link_metrics[dpid][port] = {
                    'load': 0.0,
                    'packet_loss': 0.0,
                    'delay': 0.0
                }

            # Calcular Load
            if port in self.last_stats[dpid]:
                prev_stat = self.last_stats[dpid][port]
                tx_diff = stat.tx_bytes - prev_stat.tx_bytes
                load = (tx_diff * 8) / (self.bw * self.monitor_interval)
            else:
                load = 0.0

            # Calcular Packet Loss
            total_pkts = stat.rx_packets + stat.rx_errors
            packet_loss = (stat.rx_errors / float(total_pkts)) if total_pkts > 0 else 0.0

            # Obtener Delay previamente calculado
            delay = self.link_metrics[dpid][port]['delay'] if port in self.link_metrics[dpid] else 0.0

            # Guardar métricas en el diccionario
            self.link_metrics[dpid][port]['load'] = load
            self.link_metrics[dpid][port]['packet_loss'] = packet_loss
            self.link_metrics[dpid][port]['delay'] = delay

            # Actualizar los máximos observados
            if load > self.max_load:
                self.max_load = load
            if delay > self.max_delay:
                self.max_delay = delay
            if packet_loss > self.max_packet_loss:
                self.max_packet_loss = packet_loss

            # Imprimir métricas en formato claro
            self.logger.info("%6d   %8.6f   %10.3f   %14.6f", port, load, delay * 1000, packet_loss * 100)

            # Actualizar los últimos datos de este puerto
            self.last_stats[dpid][port] = stat



if __name__ == '__main__':
    from ryu.cmd import manager
    manager.main()
