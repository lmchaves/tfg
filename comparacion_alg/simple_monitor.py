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
import pandas as pd

import time

from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


class ControlHttpApp(object):
    """
    Aplicación WSGI para recibir comandos de control.
    Guarda una referencia a la instancia del monitor para acceder a su cola.
    """
    def __init__(self, monitor_instance):
        self.monitor = monitor_instance
        self.logger = monitor_instance.logger 

    def __call__(self, environ, start_response):
        # Este método se llama en cada petición HTTP
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
                self.logger.info("Comando de control recibido por HTTP: %s", command_data)

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

        elif method == 'POST' and path == '/notify_param_change':
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size)
                notification_data = json.loads(request_body)

                # Esperamos datos como: {'src_dpid': 15, 'dst_dpid': 14, 'param_name': 'delay', 'value': 50}
                src_dpid = notification_data.get('src_dpid')
                dst_dpid = notification_data.get('dst_dpid')
                param_name = notification_data.get('param_name')
                value = notification_data.get('value')

                # Convertir DPIDs a int si vienen como otra cosa (aunque Flask los envía como int, pero mejor ser robusto)
                try:
                    src_dpid = int(src_dpid)
                    dst_dpid = int(dst_dpid)
                except (ValueError, TypeError):
                    self.logger.error("Notificación con DPIDs inválidos: src=%s, dst=%s", src_dpid, dst_dpid)
                    status = '400 Bad Request'
                    headers = [('Content-Type', 'application/json')]
                    start_response(status, headers)
                    return [json.dumps({"status": "error", "message": "Invalid source or destination DPID"}).encode('utf-8')]


                self.logger.info("Notificación de cambio de parámetro recibida: Enlace %d-%d, %s = %s",
                                 src_dpid, dst_dpid, param_name, value)

                # --- A1Actualizar self.monitor.link_metrics directamente ---
                dst_port = None
                src_port = None
                # Los enlaces en self.monitor.topology['links'] están como (src_dpid, dst_dpid, {'port': src_port})
                # Iterar sobre una copia si el hilo principal podría  modificarlo
                for link in list(self.monitor.topology.get('links', [])):
                    if link[0] == src_dpid and link[1] == dst_dpid:
                        src_port = link[2].get('port')
                        break

                if src_port is not None:
                    # Asegurarse de que la entrada para este DPID y puerto existe en link_metrics
                    if src_dpid not in self.monitor.link_metrics:
                        self.monitor.link_metrics[src_dpid] = {}
                    if src_port not in self.monitor.link_metrics[src_dpid]:
                         # Inicializar las métricas para este puerto si aún no existen
                         self.monitor.link_metrics[src_dpid][src_port] = {'load': 0.0, 'packet_loss': 0.0, 'delay': 0.0}


                    # --- Actualizar el valor específico de la métrica ---


                    if param_name == 'delay':
                        # value viene en ms, convertir a segundos
                        try:
                            delay_seconds = float(value) / 1000.0
                            self.monitor.link_metrics[src_dpid][src_port]['delay'] = delay_seconds
                            self.monitor.manual_metrics_set.add((dst_dpid, dst_port, 'delay'))
                            self.logger.info("M\u00E9trica de delay actualizada para enlace %d-%d (Puerto %d) a %.3fms",
                                             src_dpid, dst_dpid, src_port, delay_seconds * 1000)
                        except (ValueError, TypeError):
                             self.logger.error("Valor inv\u00E1lido para delay: %s", value)

                    elif param_name == 'loss':
                         # value viene en %, convertir a fracción [0, 1]
                         try:
                             packet_loss_fraction = float(value) / 100.0
                             # Asegurar que está en el rango [0, 1]
                             packet_loss_fraction = max(0.0, min(1.0, packet_loss_fraction))
                             self.monitor.link_metrics[src_dpid][src_port]['packet_loss'] = packet_loss_fraction
                             self.monitor.manual_metrics_set.add((dst_dpid, dst_port, 'packet_loss'))
                             self.logger.info("M\u00E9trica de loss actualizada para enlace %d-%d (Puerto %d) a %.2f%%",
                                             src_dpid, dst_dpid, src_port, packet_loss_fraction * 100)
                         except (ValueError, TypeError):
                             self.logger.error("Valor inv\u00E1lido para loss: %s", value)

                    elif param_name == 'bw':

                         self.logger.warning("Notificaci\u00F3n BW recibida para enlace %d-%d. Considerar guardar BW configurado por separado.", src_dpid, dst_dpid)

                
                    # Buscar en el otro extremo del enlace es simétrico
                    for link in self.monitor.topology.get('links', []):
                        if link[0] == dst_dpid and link[1] == src_dpid:
                            dst_port = link[2].get('port')
                            break

                    if dst_port is not None and dst_dpid in self.monitor.link_metrics and dst_port in self.monitor.link_metrics[dst_dpid]:
                         if param_name == 'delay':
                              self.monitor.link_metrics[dst_dpid][dst_port]['delay'] = delay_seconds
                              self.monitor.manual_metrics_set.add((dst_dpid, dst_port, 'delay'))
                              self.logger.info("M\u00E9trica de delay actualizada (otro extremo) para enlace %s-%s (Puerto %d) a %.3fms",
                                               dst_dpid, src_dpid, dst_port, delay_seconds * 1000)
                         elif param_name == 'loss':
                              self.monitor.link_metrics[dst_dpid][dst_port]['packet_loss'] = packet_loss_fraction
                              self.monitor.manual_metrics_set.add((dst_dpid, dst_port, 'packet_loss'))
                              self.logger.info("M\u00E9trica de loss actualizada (otro extremo) para enlace %s-%s (Puerto %d) a %.2f%%",
                                               dst_dpid, src_dpid, dst_port, packet_loss_fraction * 100)
                         # Repetir para bw si es necesario

                    else:
                        self.logger.warning("No se encontr\u00F3 el puerto destino (%s) o su entrada en link_metrics para actualizar m\u00E9tricas sim\u00E9tricas.", dst_port)


                else:
                    self.logger.warning("No se encontr\u00F3 la entrada para DPID %s, Puerto %s en self.monitor.link_metrics para actualizar.", src_dpid, src_port)

                
                
                dpid1, dpid2 = sorted((src_dpid, dst_dpid)) # Obtener los DPIDs ordenados
                if (dpid1 == 14 and dpid2 == 15):
                    self.logger.info("--- METRICAS ACTUALIZADAS (por notificaci\u00F3n) para enlace 14-15/15-14 ---")

                    # Obtener las m\u00E9tricas actualizadas para los puertos relevantes
                    # Sabemos que 15-eth2 <=> 14-eth3
                    metrics_15_eth2 = self.monitor.link_metrics.get(15, {}).get(2, {})
                    metrics_14_eth3 = self.monitor.link_metrics.get(14, {}).get(3, {})

                    if metrics_15_eth2:
                        self.logger.info("  Enlace 15 -> 14 (Puerto 15-eth2):")
                        self.logger.info("    Delay: %.3f ms", metrics_15_eth2.get('delay', 0.0) * 1000)
                        self.logger.info("    Loss: %.2f %%", metrics_15_eth2.get('packet_loss', 0.0) * 100)
                        self.logger.info("    Load: %.6f", metrics_15_eth2.get('load', 0.0))
                    else:
                        self.logger.info("  M\u00E9tricas para 15-eth2 no disponibles en link_metrics.")


                    if metrics_14_eth3:
                         self.logger.info("  Enlace 14 -> 15 (Puerto 14-eth3):")
                         self.logger.info("    Delay: %.3f ms", metrics_14_eth3.get('delay', 0.0) * 1000)
                         self.logger.info("    Loss: %.2f %%", metrics_14_eth3.get('packet_loss', 0.0) * 100)
                         self.logger.info("    Load: %.6f", metrics_14_eth3.get('load', 0.0))
                    else:
                         self.logger.info("  M\u00E9tricas para 14-eth3 no disponibles en link_metrics.")

                    self.logger.info("--------------------------------------------------------")
                # --- Fin del bloque de logueo espec\u00EDfico ---

                
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

        self.experiment_snapshot = 9 
        self.experiment_runs     = 30

        # --- \u00A1Nuevo! Diccionario para rastrear m\u00E9tricas establecidas manualmente ---
        # Clave: (dpid, port_no, param_name) , Valor: True
        self.manual_metrics_set = set()


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
        alpha=1.0, beta=1.0, gamma=1.0, rho=0.5,Q=1.0, high_cost=1000, q0=0.9, phi=0.5)


        self.logger.info("Ruta óptima encontrada: %s con costo %.6f", best_path, best_cost)
        return best_path, best_cost, cost_matrix, load_matrix

    def run_dijkstra(self, snapshot, metric_key='delay'):
        """
        Ejecuta el algoritmo de Dijkstra con los datos de la red.
        Recibe el snapshot como parámetro y la métrica a utilizar para el costo.
        """
        self.logger.info("Ejecutando Dijkstra, usando métrica: %s.", metric_key)
        # self.logger.debug("Snapshot para Dijkstra: %s", snapshot)

        nodes = list(self.topology['switches']) 
        if not nodes:
            self.logger.warning("Dijkstra: No hay nodos en la topología.")
            return [], float('inf')

        topology_links = self.topology['links']
        num_nodes = len(nodes)
        node_to_idx = {node_dpid: i for i, node_dpid in enumerate(nodes)}
        idx_to_node = {i: node_dpid for i, node_dpid in enumerate(nodes)}

        row_indices = []
        col_indices = []
        data_values = []

        for src_dpid, dst_dpid, link_info in topology_links:
            if src_dpid in node_to_idx and dst_dpid in node_to_idx:
                port_no = link_info['port'] # Puerto de salida en src_dpid
                cost = snapshot.get(src_dpid, {}).get(port_no, {}).get(metric_key, float('inf'))
                
                if cost < 0: # Dijkstra no funciona bien con costos negativos en general, aunque scipy podría manejarlo. Para métricas de red, es inusual.
                    self.logger.warning(f"Dijkstra: Costo negativo ({cost}) para enlace {src_dpid}(p{port_no}) -> {dst_dpid} usando métrica '{metric_key}'. Usando inf.")
                    cost = float('inf')
                
                if cost != float('inf'):
                    row_indices.append(node_to_idx[src_dpid])
                    col_indices.append(node_to_idx[dst_dpid])
                    data_values.append(cost)
            else:
                self.logger.warning(f"Dijkstra: Enlace ({src_dpid} -> {dst_dpid}) con DPID no en nodos. Omitiendo.")

        if not data_values: # No hay enlaces con costos finitos
             self.logger.warning(f"Dijkstra: No hay enlaces válidos o todos tienen costo infinito para métrica '{metric_key}'.")
             if self.src_node_dpid == self.dst_node_dpid and self.src_node_dpid in node_to_idx: # Origen y destino son el mismo
                 self.logger.info(f"Dijkstra: Origen y destino son el mismo nodo ({self.src_node_dpid}). Costo 0.")
                 return [self.src_node_dpid], 0.0
             return [], float('inf')

        graph_csr = csr_matrix((data_values, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
        self.logger.debug(f"Dijkstra: Matriz de costos (CSR) para '{metric_key}' construida con {graph_csr.nnz} elementos.")

        if self.src_node_dpid not in node_to_idx or self.dst_node_dpid not in node_to_idx:
            self.logger.warning(f"Dijkstra: DPID origen ({self.src_node_dpid}) o destino ({self.dst_node_dpid}) no en nodos.")
            return [], float('inf')

        source_idx = node_to_idx[self.src_node_dpid]
        target_idx = node_to_idx[self.dst_node_dpid]

        dist_matrix, predecessors = dijkstra(csgraph=graph_csr, directed=True, indices=source_idx, return_predecessors=True, unweighted=False)
        
        # dist_matrix es un array donde dist_matrix[i] es la distancia desde source_idx a i
        shortest_cost = dist_matrix[target_idx]
        
        if shortest_cost == float('inf') or shortest_cost == np.inf : # numpy.inf can also occur
            self.logger.info(f"Dijkstra: No se encontró ruta de {self.src_node_dpid} a {self.dst_node_dpid} con métrica '{metric_key}'.")
            return [], float('inf')

        # Reconstruir la ruta
        path_indices = []
        curr_idx = target_idx
        max_path_len = num_nodes # Evitar bucles infinitos si hay un problema con predecessors
        count = 0
        while curr_idx != source_idx and curr_idx != -9999 and count < max_path_len: # -9999 es el valor si no hay predecesor
            path_indices.append(curr_idx)
            curr_idx = predecessors[curr_idx]
            count +=1
        
        if curr_idx == -9999 and target_idx != source_idx : # No se pudo llegar al origen
             self.logger.error(f"Dijkstra: Error reconstruyendo ruta para '{metric_key}'. Origen no alcanzado. Predecesores desde {self.src_node_dpid}: {predecessors}")
             return [], float('inf')
        if count >= max_path_len and target_idx != source_idx: # Posible ciclo o error
            self.logger.error(f"Dijkstra: Error reconstruyendo ruta para '{metric_key}', posible ciclo o ruta demasiado larga. Último índice: {curr_idx}")
            return [], float('inf')


        path_indices.append(source_idx)
        path_indices.reverse() # La ruta está de destino a origen, invertirla
        
        best_path_dpid = [idx_to_node[i] for i in path_indices]
        self.logger.info(f"Dijkstra: Ruta óptima ({metric_key}): {' -> '.join(map(str, best_path_dpid))} con costo {shortest_cost:.6f}")
        return best_path_dpid, shortest_cost


    def run_llbaco_noacs(self, snapshot):
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
        return best_path, best_cost, cost_matrix, load_matrix

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

        elif command == 'set_endpoints': # \u00A1Nuevo comando!
            src_dpid = command_data.get('src')
            dst_dpid = command_data.get('dst')
            if src_dpid is not None and dst_dpid is not None:
                self.src_node_dpid = src_dpid
                self.dst_node_dpid = dst_dpid
                self.logger.info("Puntos finales actualizados: Origen=%s, Destino=%s", self.src_node_dpid, self.dst_node_dpid)
                # Opcional: Podr\u00EDas querer forzar una ejecuci\u00F3n del algoritmo inmediatamente
                # despu\u00E9s de cambiar los puntos finales. Esto requerir\u00EDa se\u00F1alizar el hilo _monitor
                # para que se ejecute sin esperar el monitor_interval.
                # Por ejemplo, podr\u00EDas poner un evento en el hilo _monitor
                # if self._execute_now_event is not None:
                #     self._execute_now_event.send()
            else:
                self.logger.warning("Comando 'set_endpoints' recibido sin src o dst v\u00E1lidos.")

        else:
            self.logger.warning("Comando de control desconocido recibido: %s", command)

    



    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info("Registrando datapath: %016x", datapath.id)
                self.datapaths[datapath.id] = datapath

                # Inicializar métricas para todos los puertos del switch si no existen o sus métricas
                if datapath.id not in self.link_metrics:
                    self.link_metrics[datapath.id] = {}

                switches = topo_api.get_switch(self, datapath.id)
                if switches:
                    switch = switches[0]
                    for port in switch.ports:
                        if port.port_no not in self.link_metrics[datapath.id]:
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
                    if self.snapshot_counter == self.experiment_snapshot:
                        self.logger.info("=== Experimento: ejecutando %d runs en snapshot %d ===",
                                        self.experiment_runs, self.snapshot_counter)
                        costs = []
                        paths = []
                        aco_path_delays = [] 
                        aco_times = []
                        aco_path_loads = []

                        # Captura las matrices de costo y carga solo una vez por snapshot
                        # ya que run_llbaco las construye internamente y son las mismas para todas las 'runs' de ese snapshot.
                        # Las llamamos para la primera run y las almacenamos.
                        # Aunque run_llbaco las devuelva en cada run, solo necesitamos una copia.
                        
                        # Ejecutar la primera run para obtener las matrices
                        start_time_aco = time.time()
                        path_i, cost_i, snapshot_cost_matrix, snapshot_load_matrix = self.run_llbaco(snapshot)
                        end_time_aco = time.time()
                        time_aco = end_time_aco - start_time_aco
                        costs.append(cost_i)
                        paths.append(" → ".join(map(str, path_i)))
                        aco_times.append(time_aco)
                        self.logger.debug("Run %2d/%2d: cost=%.6f",1, self.experiment_runs, cost_i)

                        current_aco_path_delay = 0.0
                        for j in range(len(path_i) - 1):
                            u, v = path_i[j], path_i[j+1]
                                
                            if u in snapshot and v in snapshot[u] and 'delay' in snapshot[u][v]:
                                current_aco_path_delay += snapshot[u][v]['delay']
                            else:
                                self.logger.warning(f"Delay no encontrado para el enlace ({u}, {v}) en el snapshot para la ruta ACO.")
                        aco_path_delays.append(current_aco_path_delay)

                        current_aco_path_load = 0.0
                        for j in range(len(path_i) - 1):
                            u, v = path_i[j], path_i[j+1]
                            if u in snapshot and v in snapshot[u] and 'load' in snapshot[u][v]: # Usar 'load'
                                current_aco_path_load += snapshot[u][v]['load']
                        aco_path_loads.append(current_aco_path_load)

                        self.logger.debug("Run %2d/%2d: cost=%.6f, path_delay=%.6f", 1, self.experiment_runs, cost_i, current_aco_path_delay)


                        for i in range(2, self.experiment_runs + 1):

                            start_time_aco = time.time()
                            path_i, cost_i, _,_  = self.run_llbaco(snapshot)
                            end_time_aco = time.time()
                            time_aco = end_time_aco - start_time_aco

                            costs.append(cost_i)
                            paths.append(" → ".join(map(str, path_i)))
                            aco_times.append(time_aco)

                            # Calcular el retraso de la ruta para las runs adicionales de ACO
                            current_aco_path_delay = 0.0
                            for j in range(len(path_i) - 1):
                                u, v = path_i[j], path_i[j+1]
                                if u in snapshot and v in snapshot[u] and 'delay' in snapshot[u][v]:
                                    current_aco_path_delay += snapshot[u][v]['delay']
                            aco_path_delays.append(current_aco_path_delay)

                            current_aco_path_load = 0.0
                            for j in range(len(path_i) - 1):
                                u, v = path_i[j], path_i[j+1]
                                if u in snapshot and v in snapshot[u] and 'load' in snapshot[u][v]: # Usar 'load'
                                    current_aco_path_load += snapshot[u][v]['load']
                            aco_path_loads.append(current_aco_path_load)

                            self.logger.debug("Run %2d/%2d: cost=%.6f", i+1, self.experiment_runs, cost_i)

                        mean_cost = float(np.mean(costs))
                        std_cost  = float(np.std(costs, ddof=1))

                        # --- Calcular LSD (Desviación Estándar de la Carga de Enlaces) ---
                        active_loads = snapshot_load_matrix[snapshot_load_matrix > 0]
                        lsd_link_load = float(np.std(active_loads, ddof=1)) if len(active_loads) > 1 else 0.0
                        if not active_loads.size:
                            lsd_link_load = 0.0
                        self.logger.info("LSD (Link Load Std Dev) para snapshot %d: %.6f", self.snapshot_counter, lsd_link_load)


                        # --- Calcular Delay promedio de la red ---
                        all_delays = []
                        for src_dpid, ports_data in snapshot.items():
                            for port, metrics in ports_data.items():
                                delay = metrics.get('delay')
                                # Considera solo los delays de enlaces que realmente tienen tráfico o son 'activos'
                                if delay is not None and metrics.get('load', 0.0) > 0:
                                    all_delays.append(delay)
                        mean_network_delay = float(np.mean(all_delays)) if all_delays else 0.0
                        self.logger.info("Retraso promedio de la red para snapshot %d: %.6f", self.snapshot_counter, mean_network_delay)


                        # Guardar a archivo CSV
                        df = pd.DataFrame({
                            'snapshot': [self.snapshot_counter] * self.experiment_runs,
                            'run': list(range(1, self.experiment_runs + 1)),
                            'cost': costs,
                            'path': paths
                        })
                        df['mean'] = mean_cost
                        df['std'] = std_cost
                        df['time'] = aco_times
                        df['algorithm'] = 'ACO'

                        df['lsd_link_load'] = lsd_link_load
                        df['mean_network_delay'] = mean_network_delay
                        df['route_delay'] = aco_path_delays
                        df['route_load'] = aco_path_loads
                        

                        filename = f"conf_{self.snapshot_counter}.csv"
                        df.to_csv(filename, index=False)
                        self.logger.info("Resultados del experimento guardados en: %s", filename)



                        # --- Experimento Dijkstra  ---
                        self.logger.info("=== Experimento Dijkstra (delay): ejecutando %d runs en snapshot %d ===",
                                        self.experiment_runs, self.snapshot_counter)
                        dijkstra_costs = []
                        dijkstra_paths = []
                        dijkstra_times = []

                        dijkstra_delay_path_loads = [] 

                        for i in range(self.experiment_runs):
                            start_time_dijkstra = time.time()
                            path_d_i, cost_d_i = self.run_dijkstra(snapshot, metric_key='delay')
                            end_time_dijkstra = time.time()
                            time_dijkstra = end_time_dijkstra - start_time_dijkstra

                            dijkstra_costs.append(cost_d_i if cost_d_i is not None else float('inf'))
                            dijkstra_paths.append(" -> ".join(map(str, path_d_i)) if path_d_i else "N/A")
                            dijkstra_times.append(time_dijkstra)

                            # Calculamos la carga acumulada de la ruta encontrada por Dijkstra (delay)
                            current_dijkstra_delay_path_load = 0.0
                            if path_d_i: # Asegúrate de que haya una ruta válida
                                for j in range(len(path_d_i) - 1):
                                    u, v = path_d_i[j], path_d_i[j+1]

                                    # Necesitamos encontrar el puerto de salida del enlace (u,v) para obtener la carga
                                    found_link_info = None
                                    for link_src, link_dst, link_info_data in self.topology["links"]:
                                        if link_src == u and link_dst == v:
                                            found_link_info = link_info_data
                                            break

                                    if found_link_info and u in snapshot and found_link_info['port'] in snapshot[u]:
                                        load_val = snapshot[u][found_link_info['port']].get('load', 0.0)
                                        current_dijkstra_delay_path_load += load_val
                                    else:
                                        self.logger.warning(f"Carga no encontrada para el enlace ({u}, {v}) en el snapshot para la ruta Dijkstra (Delay).")

                            # <--- ESTA LÍNEA DEBE IR DENTRO DEL BUCLE (justo después del cálculo de carga para la run actual)
                            dijkstra_delay_path_loads.append(current_dijkstra_delay_path_load)

                            self.logger.debug("Dijkstra (delay) Run %2d/%2d: cost=%.6f, path_load=%.6f",
                                                i + 1, self.experiment_runs,
                                                cost_d_i if cost_d_i is not None else float('inf'),
                                                current_dijkstra_delay_path_load) # <-- Añade path_load al debug
                        
                        valid_costs_dijkstra = [c for c in dijkstra_costs if c != float('inf')]
                        if valid_costs_dijkstra:
                            dijkstra_mean_cost = float(np.mean(valid_costs_dijkstra))
                            dijkstra_std_cost  = float(np.std(valid_costs_dijkstra, ddof=1)) if len(valid_costs_dijkstra) > 1 else 0.0
                        else:
                            dijkstra_mean_cost = float('inf')
                            dijkstra_std_cost = 0.0
                        
                        df_dijkstra = pd.DataFrame({
                            'snapshot': [self.snapshot_counter] * self.experiment_runs,
                            'run': list(range(1, self.experiment_runs + 1)),
                            'cost': dijkstra_costs, # Aquí 'cost' es el retraso de la ruta
                            'path': dijkstra_paths,
                            'mean': dijkstra_mean_cost,
                            'std': dijkstra_std_cost,
                            'time': dijkstra_times,
                            'algorithm': 'Dijkstra_Delay',
                            'lsd_link_load': lsd_link_load,
                            'mean_network_delay': mean_network_delay,
                            'route_delay': dijkstra_costs, # 'route_delay' es el retraso de la ruta, que es el 'cost' para este algoritmo
                            'route_load': dijkstra_delay_path_loads # <-- USAR LA LISTA CORRECTA Y RENOMBRADA
                        })

                        filename_dijkstra = f"dijkstra_delay_conf_{self.snapshot_counter}.csv"
                        df_dijkstra.to_csv(filename_dijkstra, index=False)
                        self.logger.info("Resultados del experimento Dijkstra (delay) guardados en: %s", filename_dijkstra)
                

                    # --- Experimento ACO2 (Copia de LLBACO con nuevo nombre) ---
                        self.logger.info("=== Experimento ACO2: ejecutando %d runs en snapshot %d ===",
                                        self.experiment_runs, self.snapshot_counter)
                        aco2_costs = []
                        aco2_paths = []
                        aco2_path_delays = []
                        aco2_times = []
                        aco2_path_loads = []

                        # Ejecutar la primera run para obtener las matrices (aunque para ACO2, no las usamos globalmente)
                        start_time_aco2 = time.time()
                        # Nota: run_llbaco devuelve las matrices, pero para ACO2 no las necesitamos para métricas globales
                        path_aco2_i, cost_aco2_i, _, _ = self.run_llbaco_noacs(snapshot)
                        end_time_aco2 = time.time()
                        time_aco2 = end_time_aco2 - start_time_aco2

                        aco2_costs.append(cost_aco2_i)
                        aco2_paths.append(" → ".join(map(str, path_aco2_i)))
                        aco2_times.append(time_aco2)
                        self.logger.debug("ACO2 Run %2d/%2d: cost=%.6f", 1, self.experiment_runs, cost_aco2_i)

                        current_aco2_path_delay = 0.0
                        if path_aco2_i:
                            for j in range(len(path_aco2_i) - 1):
                                u, v = path_aco2_i[j], path_aco2_i[j+1]
                                # Asumiendo que el snapshot tiene la estructura esperada
                                found_link_info_delay = None
                                for link_src, link_dst, link_info_data in self.topology["links"]:
                                    if link_src == u and link_dst == v:
                                        found_link_info_delay = link_info_data
                                        break
                                if found_link_info_delay and u in snapshot and found_link_info_delay['port'] in snapshot[u]:
                                    current_aco2_path_delay += snapshot[u][found_link_info_delay['port']].get('delay', 0.0)
                                else:
                                    self.logger.warning(f"Delay no encontrado para el enlace ({u}, {v}) en el snapshot para la ruta ACO2.")
                        aco2_path_delays.append(current_aco2_path_delay)

                        current_aco2_path_load = 0.0
                        if path_aco2_i:
                            for j in range(len(path_aco2_i) - 1):
                                u, v = path_aco2_i[j], path_aco2_i[j+1]
                                found_link_info_load = None
                                for link_src, link_dst, link_info_data in self.topology["links"]:
                                    if link_src == u and link_dst == v:
                                        found_link_info_load = link_info_data
                                        break
                                if found_link_info_load and u in snapshot and found_link_info_load['port'] in snapshot[u]:
                                    current_aco2_path_load += snapshot[u][found_link_info_load['port']].get('load', 0.0)
                                else:
                                    self.logger.warning(f"Carga no encontrada para el enlace ({u}, {v}) en el snapshot para la ruta ACO2.")
                        aco2_path_loads.append(current_aco2_path_load)

                        self.logger.debug("ACO2 Run %2d/%2d: cost=%.6f, path_delay=%.6f, path_load=%.6f",
                                          1, self.experiment_runs, cost_aco2_i, current_aco2_path_delay, current_aco2_path_load)


                        for i in range(2, self.experiment_runs + 1):
                            start_time_aco2 = time.time()
                            path_aco2_i, cost_aco2_i, _, _ = self.run_llbaco_noacs(snapshot)
                            end_time_aco2 = time.time()
                            time_aco2 = end_time_aco2 - start_time_aco2

                            aco2_costs.append(cost_aco2_i)
                            aco2_paths.append(" → ".join(map(str, path_aco2_i)))
                            aco2_times.append(time_aco2)

                            # Calcular el retraso de la ruta para las runs adicionales de ACO2
                            current_aco2_path_delay = 0.0
                            if path_aco2_i:
                                for j in range(len(path_aco2_i) - 1):
                                    u, v = path_aco2_i[j], path_aco2_i[j+1]
                                    found_link_info_delay = None
                                    for link_src, link_dst, link_info_data in self.topology["links"]:
                                        if link_src == u and link_dst == v:
                                            found_link_info_delay = link_info_data
                                            break
                                    if found_link_info_delay and u in snapshot and found_link_info_delay['port'] in snapshot[u]:
                                        current_aco2_path_delay += snapshot[u][found_link_info_delay['port']].get('delay', 0.0)
                            aco2_path_delays.append(current_aco2_path_delay)

                            current_aco2_path_load = 0.0
                            if path_aco2_i:
                                for j in range(len(path_aco2_i) - 1):
                                    u, v = path_aco2_i[j], path_aco2_i[j+1]
                                    found_link_info_load = None
                                    for link_src, link_dst, link_info_data in self.topology["links"]:
                                        if link_src == u and link_dst == v:
                                            found_link_info_load = link_info_data
                                            break
                                    if found_link_info_load and u in snapshot and found_link_info_load['port'] in snapshot[u]:
                                        current_aco2_path_load += snapshot[u][found_link_info_load['port']].get('load', 0.0)
                            aco2_path_loads.append(current_aco2_path_load)

                            self.logger.debug("ACO2 Run %2d/%2d: cost=%.6f, path_delay=%.6f, path_load=%.6f",
                                              i, self.experiment_runs, cost_aco2_i, current_aco2_path_delay, current_aco2_path_load)

                        mean_aco2_cost = float(np.mean(aco2_costs))
                        std_aco2_cost  = float(np.std(aco2_costs, ddof=1))

                        # Las métricas globales (lsd_link_load, mean_network_delay) se calculan una vez por snapshot
                        # y ya están disponibles desde el bloque ACO original. NO se recalculan aquí.

                        # Guardar a archivo CSV para ACO2
                        df_aco2 = pd.DataFrame({
                            'snapshot': [self.snapshot_counter] * self.experiment_runs,
                            'run': list(range(1, self.experiment_runs + 1)),
                            'cost': aco2_costs,
                            'path': aco2_paths,
                            'mean': mean_aco2_cost,
                            'std': std_aco2_cost,
                            'time': aco2_times,
                            'algorithm': 'ACO2', # <-- Nombre del algoritmo
                            'lsd_link_load': lsd_link_load, # Usar el global ya calculado
                            'mean_network_delay': mean_network_delay, # Usar el global ya calculado
                            'route_delay': aco2_path_delays,
                            'route_load': aco2_path_loads
                        })

                        filename_aco2 = f"aco2_conf_{self.snapshot_counter}.csv" # <-- Nombre de archivo
                        try:
                            df_aco2.to_csv(filename_aco2, index=False)
                            self.logger.info("Resultados del experimento ACO2 guardados en: %s", filename_aco2)
                        except Exception as e:
                            self.logger.error("Error al guardar CSV del experimento ACO2: %s", e)


                    else:                     
                        best_path, best_cost,_,_ = self.run_llbaco(snapshot)  

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
            # Actualiza el delay para cada puerto del switch
            if dpid in self.link_metrics:
                for port in list(self.link_metrics[dpid].keys()): # Iterar sobre una copia de las claves
                    # --- \u00A1Nuevo! Verificar si el delay de este puerto fue establecido manualmente ---
                    if (dpid, port, 'delay') in self.manual_metrics_set:
                        self.logger.debug("MONITOR: Saltando actualizaci\u00F3n autom\u00E1tica de delay para %d-%d (Puerto %d) - establecido manualmente.", dpid, port, port)
                        continue # Saltar la actualizaci\u00F3n autom\u00E1tica para este puerto/m\u00E9trica

                    # --- Fin de la verificaci\u00F3n manual ---

                    self.link_metrics[dpid][port]['delay'] = delay # \u00A1Actualizaci\u00F3n autom\u00E1tica solo si no es manual!
                    self.logger.debug("MONITOR: Delay actualizado autom\u00E1ticamente para %d-%d (Puerto %d): %.3f ms", dpid, port, port, delay * 1000)

        self.logger.info("Delay promedio actualizado para dpid %s (basado en eco): %.3f ms", dpid, delay * 1000)
        del self.echo_timestamps[dpid]

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        if dpid not in self.link_metrics:
            self.link_metrics[dpid] = {}
        if dpid not in self.last_stats:
            self.last_stats[dpid] = {}

        self.logger.info("=== Estad\u00EDsticas del switch %016x ===", dpid)
        self.logger.info("Port     Load       Delay (ms)  Packet Loss (%)")
        self.logger.info("------   --------   ----------  --------------")

        for stat in body:
            port = stat.port_no

            # Solo inicializa si el puerto no está YA en link_metrics para este dpid
            if port not in self.link_metrics.get(dpid, {}):
                if dpid not in self.link_metrics:
                    self.link_metrics[dpid] = {}
                self.link_metrics[dpid][port] = {
                    'load': 0.0,
                    'packet_loss': 0.0,
                    'delay': 0.0
                }

            if port in self.last_stats[dpid]:
                prev_stat = self.last_stats[dpid][port]
                tx_diff = stat.tx_bytes - prev_stat.tx_bytes
                load = (tx_diff * 8) / (self.bw * self.monitor_interval)
            else:
                load = 0.0

            # packet_loss se calcula a partir de las estad\u00EDsticas recibidas (rx_errors, rx_packets)
            total_pkts = stat.rx_packets + stat.rx_errors
            calculated_packet_loss = (stat.rx_errors / float(total_pkts)) if total_pkts > 0 else 0.0

            # --- Actualizar self.link_metrics respetando m\u00E9tricas manuales ---

            # Actualizar Load (asumimos que Load siempre se mide)
            self.link_metrics[dpid][port]['load'] = load

            # Actualizar Packet Loss solo si no est\u00E1 marcado como manual
            if (dpid, port, 'packet_loss') in self.manual_metrics_set:
                self.logger.debug("MONITOR: Saltando actualizaci\u00F3n autom\u00E1tica de packet_loss para %d-%d (Puerto %d) - establecido manualmente.", dpid, port, port)
            else:
                 self.link_metrics[dpid][port]['packet_loss'] = calculated_packet_loss
                 self.logger.debug("MONITOR: Packet_loss actualizado autom\u00E1ticamente para %d-%d (Puerto %d): %.6f", dpid, port, port, calculated_packet_loss)


            # --- Imprimir m\u00E9tricas y actualizar m\u00E1ximos usando los valores FINALES de self.link_metrics ---
            final_load = self.link_metrics.get(dpid, {}).get(port, {}).get('load', 0.0)
            final_packet_loss = self.link_metrics.get(dpid, {}).get(port, {}).get('packet_loss', 0.0)
            final_delay = self.link_metrics.get(dpid, {}).get(port, {}).get('delay', 0.0) # Obtener el valor actual, ya sea manual o automático.

            self.logger.info("%6d   %8.6f   %10.3f   %14.6f", port, final_load, final_delay * 1000, final_packet_loss * 100)

            if final_load > self.max_load:
                 self.max_load = final_load
            if final_delay > self.max_delay:
                 self.max_delay = final_delay
            if final_packet_loss > self.max_packet_loss:
                 self.max_packet_loss = final_packet_loss

            # Actualizar los \u00FAltimos datos de este puerto
            self.last_stats[dpid][port] = stat

if __name__ == '__main__':
    from ryu.cmd import manager
    manager.main()