from operator import attrgetter
import time

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.topology import event
from ryu.topology import api as topo_api
from ryu.lib.packet import packet, ethernet, lldp
import logging
import llbaco
import numpy as np
from ryu.topology.api import get_switch, get_link
from ryu.topology import event


logging.basicConfig(level=logging.INFO)


class EnhancedMonitor(simple_switch_13.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(EnhancedMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}  # Almacena los switches conectados
        self.link_metrics = {}  # Almacena métricas de enlaces {dpid: {port: {metrics}}}
        self.last_stats = {}  # Almacena estadísticas previas para el cálculo de carga
        self.bw = 10 * 1e6  # 10 Mbps (ajustar según la red)
        self.monitor_interval = 1 # Intervalo de monitoreo en segundos
        self.echo_timestamps = {}  # Para medir delay con OFPEchoRequest
        self.lldp_timestamps = {}  # Para medir delay con LLDP
        self.topology = {'switches': [], 'links': []}  # Topología de la red

        # Hilo para monitoreo continuo (puedes desactivarlo si solo quieres una instantánea)
        self.monitor_thread = hub.spawn(self._monitor)

    def get_network_snapshot(self):
        """
        Genera una instantánea del estado actual de la red.
        Recorre cada datapath y, para cada puerto, obtiene las métricas:
        - 'load' (carga del enlace)
        - 'packet_loss' (tasa de pérdida de paquetes)
        - 'delay' (retraso medido)
        Devuelve un diccionario con la estructura:
        { dpid: { port_no: { 'load': ..., 'packet_loss': ..., 'delay': ... }, ... }, ... }
        """
        snapshot = {}
        for dpid in list(self.datapaths.keys()):  # Copia de las claves antes de iterar
            snapshot[dpid] = {}
            switches = topo_api.get_switch(self, dpid)
            if not switches:
                continue
            switch = switches[0]
            for port in switch.ports:
                port_no = port.port_no
                snapshot[dpid][port_no] = {
                    'load': self.link_metrics.get(dpid, {}).get(port_no, {}).get('load', 0.0),
                    'packet_loss': self.link_metrics.get(dpid, {}).get(port_no, {}).get('packet_loss', 0.0),
                    'delay': self.link_metrics.get(dpid, {}).get(port_no, {}).get('delay', 0.0)
                }
        return snapshot

    def _get_port_connected_to(self, src, dst, topology_links):
        """
        Devuelve el puerto en src que conecta a dst, usando topology_links.
        """
        for link in topology_links:
            if link[0] == src and link[1] == dst:
                return link[2]['port']  # Devuelve el puerto en src que conecta a dst
        return None

    def run_llbaco(self, cost_matrix, topology_links, src, dst, weights):
        """
        Ejecuta el algoritmo LLBACO para encontrar la mejor ruta.
        
        Parámetros:
        - cost_matrix: matriz de costos.
        - topology_links: lista de enlaces en la red.
        - src: nodo de origen.
        - dst: nodo de destino.
        - weights: pesos para las métricas (load, delay, packet_loss).
        
        Devuelve la mejor ruta y su costo.
        """
        best_path, best_cost = llbaco.run_aco_llbaco(
            cost_matrix, iterations=100, colony=50, alpha=1.0, beta=1.0, del_tau=1.0, rho=0.5
        )
        return best_path, best_cost

    def install_best_path(self, best_path):
        """
        Instala la mejor ruta en los switches.
        
        Parámetros:
        - best_path: lista de nodos que forman la ruta óptima.
        """
        for i in range(len(best_path) - 1):
            src = best_path[i]
            dst = best_path[i + 1]
            self._install_flow(src, dst)

    def _install_flow(self, src, dst):
        """
        Instala un flujo en el switch para redirigir tráfico de src a dst.
        """
        datapath = self.datapaths.get(src)
        if datapath:
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(in_port=src)
            actions = [parser.OFPActionOutput(dst)]
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=1,
                match=match,
                instructions=inst
            )
            datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    @set_ev_cls(event.EventSwitchEnter)
    def _switch_enter_handler(self, ev):
        switch_list = topo_api.get_all_switch(self)
        self.topology['switches'] = [switch.dp.id for switch in switch_list]
        self.logger.info("Switches detectados: %s", self.topology['switches'])


    @set_ev_cls(event.EventLinkAdd)
    def _link_add_handler(self, ev):
        link_list = get_link(self, None)
        self.topology['links'] = [(link.src.dpid, link.dst.dpid, {'port': link.src.port_no}) for link in link_list]

        if not link_list:
            self.logger.warning("🚨 No se detectaron enlaces. Verifica si LLDP está funcionando.")
        
        for link in link_list:
            self.logger.info(f"✅ Enlace detectado: {link.src.dpid} -> {link.dst.dpid} por puerto {link.src.port_no}")


    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev=None):
        # Obtiene todos los switches conocidos
        switch_list = get_switch(self, None)
        switches = [switch.dp.id for switch in switch_list]
        
        # Obtiene todos los enlaces detectados
        links_list = get_link(self, None)
        # Puedes incluir información adicional, por ejemplo el puerto de salida
        links = [(link.src.dpid, link.dst.dpid, {'port': link.src.port_no}) for link in links_list]
        
        self.topology['switches'] = switches
        self.topology['links'] = links

        self.logger.info("Switches: %s", switches)
        self.logger.info("Links: %s", links)



    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
                self._send_echo_request(dp)
            # Después de actualizar las métricas, generar e imprimir la instantánea

            
            snapshot = self.get_network_snapshot()
            self.logger.info("Instantánea de métricas: %s", snapshot)

            self.get_topology_data()


            if snapshot:
                # Construir matriz de costos
                nodes = sorted(snapshot.keys())
                topology_links = self.topology['links']

                self.logger.info("Enlaces de la topología: %s", topology_links)

                weights = (1.0, 2.0, 5.0)  # Pesos para load, delay, packet_loss

                cost_matrix = llbaco.build_cost_matrix(snapshot, self.topology['switches'], self.topology['links'], delta=0.5)

                self.logger.info("Matriz de costos:")
                for row in cost_matrix:
                    self.logger.info(row)
                
                # Ejecutar LLBACO para encontrar la mejor ruta
                src = nodes[0]  # Origen (puede ser dinámico)
                dst = nodes[-1]  # Destino (puede ser dinámico)
                best_path, best_cost = self.run_llbaco(cost_matrix, topology_links, src, dst, weights)
                self.logger.info("Mejor ruta encontrada: %s, Costo: %.2f", best_path, best_cost)
                
                # Instalar la mejor ruta en los switches
                self.install_best_path(best_path)
                
            hub.sleep(self.monitor_interval)
            
    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath))
        datapath.send_msg(parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY))

    def _send_echo_request(self, datapath):
        parser = datapath.ofproto_parser
        self.echo_timestamps[datapath.id] = time.time()
        datapath.send_msg(parser.OFPEchoRequest(datapath, data=b''))

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def _echo_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        if dpid in self.echo_timestamps:
            rtt = time.time() - self.echo_timestamps[dpid]
            # Asignar delay a todos los puertos del switch (no recomendado, mejor usar LLDP)
            for port in self.link_metrics.get(dpid, {}):
                self.link_metrics[dpid][port]["delay"] = rtt / 2
            del self.echo_timestamps[dpid]

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        if dpid not in self.link_metrics:
            self.link_metrics[dpid] = {}
        if dpid not in self.last_stats:
            self.last_stats[dpid] = {}

        if dpid in self.last_stats:
            for stat in body:
                port = stat.port_no
                if port not in self.link_metrics[dpid]:
                    self.link_metrics[dpid][port] = {}  # Inicializar el diccionario del puerto
                if port in self.last_stats[dpid]:
                    prev_stat = self.last_stats[dpid][port]
                    tx_bytes_diff = stat.tx_bytes - prev_stat.tx_bytes
                    load = (tx_bytes_diff * 8) / (self.bw * self.monitor_interval)
                    self.link_metrics[dpid][port]["load"] = load
                    self.link_metrics[dpid][port]["packet_loss"] = 0.0  # Se actualizará después
                    self.link_metrics[dpid][port]["delay"] = 0.0

        self.last_stats[dpid] = {stat.port_no: stat for stat in body}

        for stat in body:
            port = stat.port_no
            rx_packets = stat.rx_packets
            rx_errors = stat.rx_errors
            pl = rx_errors / (rx_packets + rx_errors) if (rx_packets + rx_errors) > 0 else 0
            if dpid in self.link_metrics and port in self.link_metrics[dpid]:
                self.link_metrics[dpid][port]["packet_loss"] = pl

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == 0x88cc:  # LLDP
            lldp_pkt = pkt.get_protocol(lldp.lldp)
            src_dpid = msg.datapath.id
            dst_dpid = self._parse_lldp(lldp_pkt)

            if dst_dpid:
                key = (src_dpid, dst_dpid)
                if key in self.lldp_timestamps:
                    rtt = time.time() - self.lldp_timestamps[key]
                    # Actualiza el delay en ambos sentidos, considerando que el delay es RTT/2
                    self._update_delay(src_dpid, dst_dpid, rtt)
                    self._update_delay(dst_dpid, src_dpid, rtt)
                    del self.lldp_timestamps[key]
                else:
                    # Registra el timestamp para la respuesta del otro switch
                    self.lldp_timestamps[(dst_dpid, src_dpid)] = time.time()

    def _update_delay(self, src_dpid, dst_dpid, rtt):
        """
        Actualiza el delay en el puerto que conecta src_dpid con dst_dpid.
        """
        src_switch = topo_api.get_switch(self, src_dpid)[0]
        for port in src_switch.ports:
            if port.peer and port.peer.dpid == dst_dpid:
                port_no = port.port_no
                # Almacenar delay en ambos sentidos (enlace bidireccional)
                self.link_metrics.setdefault(src_dpid, {}).setdefault(port_no, {})["delay"] = rtt / 2
                self.logger.info("Delay entre %016x y %016x: %.3f ms", src_dpid, dst_dpid, (rtt / 2) * 1000)

    def _parse_lldp(self, lldp_pkt):
        for tlv in lldp_pkt.tlvs:
            if tlv.tlv_type == lldp.LLDP_TLV_SYSTEM_NAME:
                system_name = tlv.tlv_value.decode()
                if system_name.startswith("sw:"):
                    try:
                        return int(system_name.split(":")[1], 16)
                    except ValueError:
                        return None
        return None