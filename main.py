# mininet libraries
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel, info

# installed libraries
import networkx as nx
import matplotlib.pyplot as plt
from scapy.all import AsyncSniffer

# default libraries
import time
import random
import os
import re
import csv
import math
import argparse

# Simulation parameters
folder_captures = "captures"

# Connection parameters
HOST_LINK_MAX_BW = 2
HOST_LINK_MIN_BW = 1
SWITCH_LINK_MAX_BW = 2
SWITCH_LINK_MIN_BW = 1

class Topology(Topo):
    def __init__(self, num_switches, num_hosts, interconnectivity, seed=0):
        super().__init__()
        self.num_switches = num_switches
        self.num_hosts = num_hosts
        self.interconnectivity = interconnectivity
        self.seed = seed

        random.seed(self.seed)
        host_count = 1
        # create all the switches and hosts
        for i in range(self.num_switches):
            switch_dpid = f"s{i+1}"
            self.addSwitch(switch_dpid, stp=True, failMode='standalone')  # stp to avoid loops in the networks

            if i > 0:
                self.addLink(switch_dpid, f"s{i}", bw=random.random() * (SWITCH_LINK_MAX_BW - SWITCH_LINK_MIN_BW) + SWITCH_LINK_MIN_BW)  # add link to previous switch in a line topology

            # for each switch create and connect all the hosts
            for j in range(random.randrange(self.num_hosts)):
                host_dpid = f"h{host_count}"
                self.addHost(host_dpid)
                self.addLink(switch_dpid, host_dpid, bw=random.random() * (HOST_LINK_MAX_BW - HOST_LINK_MIN_BW) + HOST_LINK_MIN_BW)
                host_count += 1

        # add connections between switches
        connected_pairs = set()
        for i in range(1, self.num_switches + 1):
            for j in range(1, self.num_switches + 1):
                # do not link with yourself
                if i == j:
                    continue
                # already linked with previous router
                if i == j + 1 or i == j - 1:
                    continue
                # check if the reverse connection already exists
                if (j, i) in connected_pairs:
                    continue

                if random.random() < self.interconnectivity:
                    connected_pairs.add((i, j))  # Add the connected pair to the set
                    self.addLink(f"s{i}", f"s{j}", bw=random.random() * (SWITCH_LINK_MAX_BW - SWITCH_LINK_MIN_BW) + SWITCH_LINK_MIN_BW)

    def saving_topology(self):
        G = nx.Graph()

        switch_color = 'blue'
        host_color = 'yellow'
        image_file = "structure_of_topology_image.png"

        for node in self.nodes():
            if node.startswith('s'):
                G.add_node(node, color=switch_color, node_type='switch')
            else:
                G.add_node(node, color=host_color, node_type='host')

        for link in self.links():
            G.add_edge(link[0], link[1])

        switch_nodes = [node for node, attr in G.nodes(data=True) if attr['node_type'] == 'switch']
        host_nodes = [node for node, attr in G.nodes(data=True) if attr['node_type'] == 'host']

        pos = nx.spring_layout(G, seed=42, k=1 / math.sqrt(len(G.nodes())), iterations=100)

        nx.draw_networkx_nodes(G, pos, nodelist=switch_nodes, node_color=switch_color, node_size=500, label='Switches')
        nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_color=host_color, node_size=300, label='Hosts')
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
        nx.draw_networkx_labels(G, pos)

        plt.savefig(image_file)
        print(f"\n*** Topology saved as '{image_file}'")

class NetworkManager:
    def __init__(self):
        self.net = None

    def create_net(self, topology):
        self.net = Mininet(
            topo=topology,
            switch=OVSKernelSwitch,
            build=False,
            autoSetMacs=True,
            autoStaticArp=True,
            link=TCLink,
            controller=None  # Use external controller
        )
        self.net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
        return self.net

    def check_stp_configuration(self):
        s1 = self.net.get("s1")  # check the state of stp, we wait until s1 says "forward" which indicates it is complete
        while (s1.cmdPrint('ovs-ofctl show s1 | grep -o FORWARD | head -n1') != "FORWARD\r\n"):
            time.sleep(3)

    def start_servers(self, base_flows, flows_per_host):
        random.seed(time.time())  # reset random seed
        for h in self.net.hosts:
            h.cmd('iperf -s -p 5050 &')  # start iperf -Server on TCP -Port 5050

        for h in random.sample(self.net.hosts, base_flows):
            hosts = self.net.hosts.copy()
            hosts.remove(h)  # do not pick yourself
            host = random.choice(hosts)
            h.cmd(f"iperf -t 0 -c {host.IP()} -p 5050 &")  # start iperf client
            print(f"Continuous flow started from {h.name} to {host.name}")

        for h in self.net.hosts:
            hosts = self.net.hosts.copy()
            hosts.remove(h)  # do not pick yourself
            for host in random.sample(hosts, flows_per_host):
                h.cmd(f"python3 utils/traffic_generation.py {host.IP()} &")
                print(f"Periodic flow started from {h.name} to {host.name}")

    def create_captures_folder(self):
        os.system(f"rm -rf {folder_captures}")  # delete the folder contents before starting
        os.mkdir(folder_captures)
        for s in self.net.switches:
            os.mkdir(os.path.join(folder_captures, s.name))

    def start_traffic_capture(self):
        interface_pattern = re.compile(r's\d+-eth\d+')  # find mn interfaces (s1-eth1) etc
        interfaces = [i for i in os.listdir('/sys/class/net/') if interface_pattern.match(i)]
        if len(interfaces) == 0:
            print(f"ERROR: could not find any mininet network adapters for some reason, quitting")
            exit(1)

        self.sniffers = []
        for i in interfaces:
            # by splitting the interface name we separate the switch from the port and therefore assign the correct folder
            path = os.path.join(folder_captures, *i.split('-'))
            csvfile = open(path + '.csv', 'w', newline='')
            writer = csv.writer(csvfile)

            print(f" Beginning capture on {i}")

            writer.writerow(['ds', 'y'])

            def handler(pkt, writer):
                writer.writerow([pkt.time, len(pkt)])

            packet_handler = lambda pkt, writer=writer: handler(pkt, writer)
            sniffer = AsyncSniffer(iface=i, store=False, prn=packet_handler)

            sniffer.start()
            self.sniffers.append((sniffer, csvfile))

    def stop_traffic_capture(self):
        for sniffer, csvfile in self.sniffers:
            sniffer.stop()
            csvfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Network Testing Script")
    parser.add_argument('--switches', type=int, default=7, help="Number of switches")
    parser.add_argument('--hosts', type=int, default=2, help="Number of hosts per switch")
    parser.add_argument('--cross-connection', type=float, default=0.30, help="Percentage of cross-connections between non-adjacent switches")
    parser.add_argument('--time', type=int, default=30, help="Duration of the test in seconds")
    parser.add_argument('--base-flows', type=int, default=3, help="Number of constant iperf flows")
    parser.add_argument('--flows', type=int, default=2, help="Number of periodic flows per host")
    args = parser.parse_args()

    # Arguments
    SWITCHES = args.switches
    HOSTS_PER_SWITCH = args.hosts
    CROSS_CONNECTION = args.cross_connection
    TEST_TIME = args.time
    NUM_IPERF_FLOWS = args.base_flows
    FLOWS_PER_HOST = args.flows

    network = NetworkManager()

    print('*** Clean network instances\n')
    os.system("mn -c")

    topology = Topology(SWITCHES, HOSTS_PER_SWITCH, CROSS_CONNECTION, 0)
    topology.saving_topology()

    setLogLevel('info')
    net = network.create_net(topology)

    net.build()
    net.start()
    time.sleep(1)

    print("\n*** Network built, waiting for STP to configure itself")
    network.check_stp_configuration()

    print("\n*** STP ok, waiting 5 seconds...")
    time.sleep(5)

    print("\n*** Testing ping connectivity...")
    net.pingAll()
    time.sleep(1)

    print("\n*** Begin traffic generation\n")
    network.start_servers(NUM_IPERF_FLOWS, FLOWS_PER_HOST)
    time.sleep(2)

    print("\n*** Begin traffic capturing\n\n")
    network.create_captures_folder()
    network.start_traffic_capture()

    print(f"\n*** Waiting for test time ({TEST_TIME} seconds)")
    time.sleep(TEST_TIME)

    print("\n*** Stopping traffic capture...")
    network.stop_traffic_capture()
    net.stop()

