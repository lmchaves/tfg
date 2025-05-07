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
import llbaco_aux
import numpy as np
import requests
import json 
import eventlet.wsgi 
import eventlet 
import eventlet.queue 

# Clase para la aplicaci\u00F3n WSGI que manejar\u00E1 las peticiones HTTP
class ControlHttpApp(object):
    """
    Aplicaci\u00F3n WSGI para recibir comandos de control.
    Guarda una referencia a la instancia del monitor para acceder a su cola.
    """
    def __init__(self, monitor_instance):
        # Guardamos una referencia a la instancia de ExtendedMonitor
        self.monitor = monitor_instance
        self.logger = monitor_instance.logger # Acceso al logger del monitor

    def __call__(self, environ, start_response):
        # Este m\u00E9todo se llama en cada petici\u00F3n HTTP
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')

        if method == 'POST' and path == '/control':
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size)
                command_data = json.loads(request_body)

                command = command_data.get('command')

                # Ponemos el comando en la cola de control del monitor
                # Usamos self.monitor para acceder a la instancia
                self.monitor.control_queue.put(command_data)
                self.logger.info("Comando de control recibido v\u00EDa HTTP: %s", command_data)

                if command == 'continue' and self.monitor.paused and self.monitor._resume_event is not None:

                     self.logger.debug("HTTP Handler: Intentando enviar signal. Tipo de _resume_event: %s, Valor: %s (ID: %s)",
                                       type(self.monitor._resume_event),
                                       self.monitor._resume_event,
                                       id(self.monitor._resume_event))

                     # \u00A1A\u00F1adir estas l\u00EDneas de prueba!
                     try:
                         self.logger.debug("HTTP Handler: --- Prueba diagn\u00F3stica ---")
                         temp_event = hub.Event()
                         self.logger.debug("HTTP Handler: Evento temporal creado: %s (ID: %s)", type(temp_event), id(temp_event))
                         temp_event.set() # \u00A1Intentar enviar en este nuevo evento!
                         self.logger.debug("HTTP Handler: temp_event.set() funcion\u00F3.")
                     except AttributeError as e:
                         self.logger.error("HTTP Handler: ERROR: temp_event.set() fall\u00F3 tambi\u00E9n: %s", e)
                         # Si este error aparece aqu\u00ED, el problema es con la librer\u00EDa eventlet

                     self.logger.debug("HTTP Handler: --- Fin Prueba diagn\u00F3stica ---")


                     self.logger.debug("HTTP Handler: Enviando signal a _resume_event (ID: %s)", id(self.monitor._resume_event))
                     self.monitor._resume_event.set() # <--- La l\u00EDnea original que falla




                status = '200 OK'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [json.dumps({"status": "success", "message": "Command received"}).encode('utf-8')]

            except Exception as e:
                self.logger.error("Error procesando comando HTTP: %s", e)
                status = '500 Internal Server Error'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [json.dumps({"status": "error", "message": str(e)}).encode('utf-8')]

        else:
            # Otros m\u00E9todos o rutas no soportadas
            status = '404 Not Found'
            headers = [('Content-Type', 'application/json')]
            start_response(status, headers)
            return [json.dumps({"status": "error", "message": "Endpoint not found"}).encode('utf-8')]



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
        self.monitor_interval = 5
        # Para medir el delay usando mensajes echo
        self.echo_timestamps = {}  
        
        self.topology = {'switches': [], 'links': []}  # Topología de la red

        self.max_load = 0.0
        self.max_delay = 0.0
        self.max_packet_loss = 0.0


        #### Pérdida de paquetes #######

        # Diccionario para guardar las conexiones entre puertos { (dpid, port): (peer_dpid, peer_port) }
        self.link_peers = {}
        # Estadísticas previas de tx y rx
        self.prev_tx = {}  # { (dpid, port): tx_packets }
        self.prev_rx = {}  # { (dpid, port): rx_packets }

        self.src_node_dpid = 15 # Origen
        self.dst_node_dpid = 1 # Destino

        self.snapshot_counter = 0

        self.control_queue = hub.Queue() # Cola para las isntantaenas
        self.paused = False  # Para pausar la instantanea
        self._resume_event = None

        # Inicia el hilo de monitoreo
        self.monitor_thread = hub.spawn(self._monitor)


        # Iniciar el servidor HTTP de control en un hilo verde
        # Escuchar en localhost:8080
        control_app_instance = ControlHttpApp(self)
        self.control_server_thread = hub.spawn(eventlet.wsgi.server, eventlet.listen(('127.0.0.1', 8080)), control_app_instance)
        self.logger.info("Servidor de control HTTP iniciado en 127.0.0.1:8080")




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
        cost_matrix, load_matrix = llbaco_aux.build_cost_load_matrix(snapshot, nodes, topology_links, delta)


        self.logger.info("Matriz de costos:")
        for row in cost_matrix:
            self.logger.info(row)
        
        self.logger.info("Matriz de cargas:")
        for row in load_matrix:
            self.logger.info(row)

        # Ejecutar el algoritmo LLBACO
        best_path, best_cost = llbaco_aux.run_aco_llbaco(
        nodes, cost_matrix,load_matrix, self.src_node_dpid, self.dst_node_dpid, iterations=200, colony_size=100, 
        alpha=1.0, beta=1.0, gamma=1.0, rho=0.5,Q=1.0, high_cost=1000)


        self.logger.info("Ruta óptima encontrada: %s con costo %.6f", best_path, best_cost)
        return best_path, best_cost

    def build_data_for_flask(self, snapshot, best_path=None, best_cost=None, counter=None):
        """
        Prepara los datos de la red para enviar a Flask.
        Incluye la ruta óptima, su costo y un contador.
        """
        data = {
            "switches": self.topology["switches"],
            "links": [],
            "best_path": best_path.tolist() if isinstance(best_path, np.ndarray) else (best_path if best_path is not None else []),
            "best_cost": best_cost if best_cost is not None else float('inf'), 
            "counter": counter if counter is not None else 0
        }
        for (src_dpid, dst_dpid, link_info) in self.topology["links"]:
            port = link_info['port']
            load = 0.0
            delay = 0.0
            packet_loss = 0.0
            if src_dpid in snapshot and port in snapshot[src_dpid]:
                load = snapshot[src_dpid][port]['load']
                delay = snapshot[src_dpid][port]['delay']
                packet_loss = snapshot[src_dpid][port]['packet_loss']

            data["links"].append({
                "src": src_dpid,
                "dst": dst_dpid,
                "load": load,
                "delay": delay,
                "packet_loss": packet_loss
            })
        return data

    def process_control_command(self, command_data):
        """Procesa un comando de control recibido."""
        command = command_data.get('command')
        if command == 'pause':
            self.logger.info("Comando: PAUSAR monitoreo.")
            self.paused = True
            # Log de estado del evento de resume antes de pausar
            self.logger.debug("PAUSE: _resume_event es %s (ID: %s)",
                              'None' if self._resume_event is None else 'Existente',
                              id(self._resume_event) if self._resume_event else 'N/A')

        elif command == 'continue':
            self.logger.info("Comando: CONTINUAR monitoreo.")
            self.paused = False
            # Log de estado del evento de resume antes de intentar enviar
            self.logger.debug("CONTINUE: _resume_event es %s antes de set() (ID: %s)",
                              'None' if self._resume_event is None else 'Existente',
                              id(self._resume_event) if self._resume_event else 'N/A')

            # \u00A1Solo enviar si el evento existe!
            if self._resume_event is not None:
                 self.logger.debug("CONTINUE: Llamado _resume_event.set() (ID: %s)", id(self._resume_event))
                 self._resume_event.set() # Despierta el hilo que est\u00E1 esperando
            else:
                 self.logger.debug("CONTINUE: _resume_event es None, no se llama set(). El monitor no estaba en wait().")


        elif command == 'skip':
            steps = command_data.get('steps', 1)
            self.logger.info("Comando: SALTAR %d instant\u00E1nea(s).", steps)
            self._skip_steps += steps
            self.logger.debug("SKIP: _resume_event es %s (ID: %s)",
                              'None' if self._resume_event is None else 'Existente',
                              id(self._resume_event) if self._resume_event else 'N/A')





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
        self._skip_steps = 0 # Inicializar variable de salto

        while True:
            # 1. Verificar comandos de control
            # Leer de la cola sin bloquear (si hay algo)
            while not self.control_queue.empty():
                try:
                    command_data = self.control_queue.get_nowait()
                    self.process_control_command(command_data)
                except eventlet.queue.Empty:
                    pass # La cola estaba vacía, lo cual es normal

            # 2. Pausar si es necesario
            if self.paused:
                self.logger.debug("Monitoreo PAUSADO. Esperando comando 'continue'...")

                # \u00A1Crear un nuevo evento cada vez que se pausa si a\u00FAn no hay uno!
                # El manejador HTTP lo necesitar\u00E1 para hacer .set()
                if self._resume_event is None:
                    self.logger.debug("MONITOR: Creando nuevo _resume_event para esta pausa.")
                    self._resume_event = hub.Event()
                    self.logger.debug("MONITOR: Nuevo _resume_event creado (ID: %s)", id(self._resume_event))


                # Log de estado del evento de resume justo antes de esperar
                self.logger.debug("MONITOR: _resume_event es %s antes de wait() (ID: %s)",
                                  'None' if self._resume_event is None else 'Existente',
                                  id(self._resume_event) if self._resume_event else 'N/A')


                self._resume_event.wait() # El hilo duerme aqu\u00ED

                # \u00A1Despu\u00E9s de wait() (cuando se reanuda), limpiar el evento!
                self.logger.debug("MONITOR: _resume_event es Existente despu\u00E9s de wait() (ID: %s)",
                                  id(self._resume_event) if self._resume_event else 'N/A') # <--- Log corregido

                self.logger.debug("Monitoreo reanudado.")

                # Limpiar el evento despu\u00E9s de que se us\u00F3 para reanudar
                self._resume_event = None # <--- Limpiar el evento


                continue


            # 3. Generar snapshot y ejecutar LLBACO (Solo si no estamos saltando)
            if self._skip_steps > 0:
                self.logger.debug("Saltando instant\u00E1nea (%d restantes).", self._skip_steps)
                self._skip_steps -= 1
            else:
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
                if snapshot: 
                    self.snapshot_counter += 1 
                    best_path, best_cost = self.run_llbaco(snapshot)  

                    dpids = self.topology['switches']      
                    best_dpid_path = best_path
                    data_for_flask = self.build_data_for_flask(snapshot, best_dpid_path, best_cost, self.snapshot_counter)

                # Enviar snapshot a Flask
                try:
                    requests.post("http://127.0.0.1:5000/update", json=data_for_flask)
                except Exception as e:
                    self.logger.error("Error enviando datos a Flask: %s", e)

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

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        self.logger.debug("Packet in (minimal log): dpid=%s in_port=%s", datapath.id, in_port)


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
