# assume docker version >= 1.13
import sys
import os
import argparse
import logging
from pathlib import Path
import json
import math
# from socket import SOCK_STREAM, socket, AF_INET, SOL_SOCKET, SO_REUSEADDR

from pathlib import Path
sys.path.append(str(Path.cwd()))

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--nodes', dest='nodes', nargs='+', type=str, required=True)
parser.add_argument('--cluster-config', dest='cluster_config', type=str, required=True)
parser.add_argument('--replica-cpus', dest='replica_cpus', type=int, default=4)

# data collection parameters
# TODO: add argument parsing here

# -----------------------------------------------------------------------
# parse args
# -----------------------------------------------------------------------
args = parser.parse_args()
# todo: currently assumes all vm instances have the same #cpus
# MaxCpus = args.cpus
# StackName = args.stack_name
nodes = [s.strip() for s in args.nodes]
cluster_config_path = Path.cwd() / '..' / 'config' / args.cluster_config.strip()
replica_cpus = args.replica_cpus
# scale_factor = args.scale_factor
# cpu_percent = args.cpu_percent

IP_ADDR = {}
# IP_ADDR["node0.nevenas-236518.hwswcodesign-pg0.utah.cloudlab.us"]     = "10.10.1.1"
# IP_ADDR["node1.nevenas-236518.hwswcodesign-pg0.utah.cloudlab.us"]     = "10.10.1.2"
# IP_ADDR["node-0.urja.ragger-pg0.wisc.cloudlab.us"]     = "10.0.1.1"
# IP_ADDR["node-1.urja.ragger-pg0.wisc.cloudlab.us"]     = "10.0.1.2"

IP_ADDR["node0.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.1"
IP_ADDR["node1.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.2"
IP_ADDR["node2.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.3"
IP_ADDR["node3.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.4"
IP_ADDR["node4.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.5"
IP_ADDR["node5.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.6"
IP_ADDR["node6.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us"] = "10.10.1.7"

service_config = {
    "nginx-thrift":         {'max_replica': 4},
    "compose-post-service": {'max_replica': 1},
    "compose-post-redis":   {'max_replica': 1},
    "text-service":         {'max_replica': 1},
    "text-filter-service":  {'max_replica': 1},
    "user-service":         {'max_replica': 1},
    "user-memcached":       {'max_replica': 1},
    "user-mongodb":         {'max_replica': 1},
    "media-service":        {'max_replica': 4, 'max_cpus': 4},
    "media-filter-service": {'max_replica': 8, 'max_cpus': 32},
    "unique-id-service":    {'max_replica': 1},
    "url-shorten-service":  {'max_replica': 1},
    "user-mention-service": {'max_replica': 1},
    "post-storage-service": {'max_replica': 1, 'max_cpus': 16},
    "post-storage-memcached":   {'max_replica': 1},
    "post-storage-mongodb":     {'max_replica': 1},
    "user-timeline-service":    {'max_replica': 1},
    "user-timeline-redis":      {'max_replica': 1},
    "user-timeline-mongodb":    {'max_replica': 1},
    "write-home-timeline-service":  {'max_replica': 1},
    "write-home-timeline-rabbitmq": {'max_replica': 1},
    "write-user-timeline-service":  {'max_replica': 1},
    "write-user-timeline-rabbitmq": {'max_replica': 1},
    "home-timeline-service":    {'max_replica': 4},
    "home-timeline-redis":      {'max_replica': 1},
    "social-graph-service":     {'max_replica': 1},
    "social-graph-redis":   {'max_replica': 1},
    "social-graph-mongodb": {'max_replica': 1}
    # "jaeger": {"replica": 1}
}

scalable_service = [
    "nginx-thrift",
    "compose-post-service",
    "text-service",
    "text-filter-service",
    "user-service",
    "media-service",
    "unique-id-service",
    "url-shorten-service",
    "user-mention-service",
    "post-storage-service",
    "user-timeline-service",
    "write-home-timeline-service",
    "write-home-timeline-rabbitmq",
    "write-user-timeline-service",
    "write-user-timeline-rabbitmq",
    "home-timeline-service",
    "social-graph-service"
]

for service in service_config:
    service_config[service]['replica'] = service_config[service]['max_replica']
    # service_config[service]['replica_cpus'] = replica_cpus
    if 'max_cpus' not in service_config[service]:
        service_config[service]['max_cpus'] = replica_cpus * service_config[service]['max_replica']
    service_config[service]['cpus'] = service_config[service]['max_cpus']

node_config = {}
for node in nodes:
    assert node in IP_ADDR
    node_config[node] = {}
    node_config[node]['ip_addr'] = IP_ADDR[node]
    if node == 'node0.nevenas-236518.hwswcodesign-pg0.utah.cloudlab.us':
        node_config[node]['cpus'] = 16
        node_config[node]['label'] = 'type=compute'
    elif node == 'node1.nevenas-236518.hwswcodesign-pg0.utah.cloudlab.us':
        node_config[node]['cpus'] = 16
        node_config[node]['label'] = 'type=data'
    elif node == "node-0.urja.ragger-pg0.wisc.cloudlab.us":
        node_config[node]['cpus'] = 40
        node_config[node]['label'] = 'type=compute'
    elif node == "node-1.urja.ragger-pg0.wisc.cloudlab.us":
        node_config[node]['cpus'] = 40
        node_config[node]['label'] = 'type=data'
    elif node == "node0.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us":
        node_config[node]['cpus'] = 32
        node_config[node]['label'] = 'type=compute'
    elif node == "node1.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us":
        node_config[node]['cpus'] = 32
        node_config[node]['label'] = 'type=compute'
    elif node == "node2.nevenas-259343.hwswcodesign-pg0.wisc.cloudlab.us":
        node_config[node]['cpus'] = 32
        node_config[node]['label'] = 'type=compute'
    else:
        node_config[node]['cpus'] = 32
        node_config[node]['label'] = 'type=data'

cluster_config = {}
cluster_config['nodes'] = node_config
cluster_config['service'] = service_config
cluster_config['scalable_service'] = scalable_service
cluster_config['replica_cpus'] = replica_cpus

with open(str(cluster_config_path), 'w+') as f:
	json.dump(cluster_config, f, indent=4, sort_keys=True)
