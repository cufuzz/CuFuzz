import networkx as nx
import random
import pickle
import os


class Extended_Graph:
    """
    this class is to maintain a extended graph for fuzzing schedule, which contains origin cuda graph and api-based intra
    edge coverage bitmap and the cuda graph updating and the critical information printing

    the bitmap size is upper 65536 default(i.e. 64K), this is given by paras 'MAP_SIZE' ;
    """

    def __init__(self, **kwargs):

        if 'Origin_G' not in kwargs:
            raise ValueError("The 'Origin_G' parameter is required but was not provided.")
        self.origin_graph = kwargs['Origin_G']

        if 'MAP_SIZE' not in kwargs:
            raise ValueError("The 'Mould_len' parameter is required but was not provided.")
        self.MAP_SIZE = kwargs['MAP_SIZE']

        self.api_key = {}
        self.bitmap_api = set()
        self.bitmap_api_call_edge = set()
        self.bitmap_api_order_edge = set()

        self.bitmap_api_call_edge_trace = set()
        self.bitmap_api_order_edge_trace = set()


        self.fix_node()

        self.origin_node = len(self.origin_graph.nodes)
        self.origin_call_graph = sum(
            1 for u, v, data in self.origin_graph.edges(data=True) if data.get('attr1') == 'call')
        self.origin_order_graph = sum(
            1 for u, v, data in self.origin_graph.edges(data=True) if data.get('attr2') == 'order')

        if 'load_map' not in kwargs or not kwargs['load_map']:
            self.alloc_id_for_api()   ##  self.api_key, and self.bitmap_api have been filled.
            self.calculate_edge_id()  ##  self.bitmap_api_call_edge, and self.bitmap_api_order_edge have been filled.
        else:
            with open(kwargs['load_map'], 'rb') as f:
                data = pickle.load(f)
                self.api_key = data[0]
                self.bitmap_api = data[1]
                self.bitmap_api_call_edge = data[2]
                self.bitmap_api_order_edge = data[3]
                self.bitmap_api_call_edge_trace = data[4]
                self.bitmap_api_order_edge_trace = data[5]


    def fix_node(self):
        # some node is wrong, delete it. some api call itself, delete this edge
        if self.origin_graph.has_node('for'):
            self.origin_graph.remove_node('for')
        if self.origin_graph.has_node('void'):
            self.origin_graph.remove_node('void')

        for u, v, data in list(self.origin_graph.edges(data=True)):
            if data.get('attr1') == 'call':
                if u == v:
                    self.origin_graph.remove_edge(u, v)

    def alloc_id_for_api(self):
        for node in self.origin_graph.nodes:
            value = self._generate_unique_value()
            self.api_key[node] = value
            self.bitmap_api.add(value)

    def add_new_api_key(self, the_api):
        value = self._generate_unique_value()
        self.api_key[the_api] = value
        self.bitmap_api.add(value)

    def update_origin_graph(self, newapi_pred, newapi, newapi_succ):
        self.origin_graph.add_node(newapi)
        if newapi_pred:
            self.origin_graph.add_node(newapi_pred, newapi, attr2='order')
        if newapi_succ:
            self.origin_graph.add_node(newapi, newapi_succ, attr2='order')


    def _generate_unique_value(self):
        while True:
            value = random.randint(1, self.MAP_SIZE)
            if value not in self.bitmap_api:
                return value

    def calculate_edge_id(self):
        # according to api id, hashing out the edge id
        for u, v, data in self.origin_graph.edges(data=True):
            if data.get('attr1') == 'call':
                # if (self.api_key[u] ^ self.api_key[v]) in self.bitmap_api_call_edge:
                #     print(u,v)
                self.bitmap_api_call_edge.add((self.api_key[u] >> 1) ^ self.api_key[v])
            if data.get('attr2') == 'order':
                # print(u,self.api_key[u], v, self.api_key[v], '-----', (self.api_key[u] >> 1)^ self.api_key[v])
                # if ((self.api_key[u] >> 1) ^ self.api_key[v]) in self.bitmap_api_call_edge:
                #     print('$$$$$$$$$$$$$$$')
                self.bitmap_api_order_edge.add((self.api_key[u] >> 1)^ self.api_key[v])

    def has_new_hit(self, new_api_sequence:list):
        # judge new api appearing, and new call edge, new order edge
        for new_api in new_api_sequence:
            if new_api not in self.api_key:
                value = self._generate_unique_value()
                self.api_key[new_api] = value
                self.bitmap_api.add(value)


    def display_graph(self):
        print("------------------------------------------------")
        print("Let's see what this graph has grown!")
        print(f'the origin graph: nodes -> {self.origin_node}| call edge -> {self.origin_call_graph}| \
order edge -> {self.origin_order_graph}')
        print(f'the grown graph with fuzz: nodes -> {len(self.bitmap_api)}| call edge -> \
{len(self.bitmap_api_call_edge)}| order edge -> {len(self.bitmap_api_order_edge)}')
        print("------------------------------------------------")

    def save_bitmap(self, path1, the_time):
        save_name = os.path.join(path1, f'bitmap_running_{the_time}_hours.pkl')
        with open(save_name, 'wb') as f:
            pickle.dump([self.api_key, self.bitmap_api, self.bitmap_api_call_edge, self.bitmap_api_order_edge,
                         self.bitmap_api_call_edge_trace, self.bitmap_api_order_edge_trace], f)


    def get_call_edge_attribute_for_one_path(self, the_path:list) -> list:
        """
        this list record the call edge within the path(i.e. api sequence).
        Return's format is:
          1:  [ api1 calls api2, ..., apin calls apin+1 ]
          2:  [ (api1, api2), ..., (apin, apin+1) ]
        """
        call_list = []
        call_list_tuple = []
        for i in range(len(the_path) - 1):
            if (the_path[i] not in self.api_key) or (the_path[i+1] not in self.api_key):
                continue
            else:
                edge_id = (self.api_key[the_path[i]] >> 1)^ self.api_key[the_path[i+1]]
                if edge_id in self.bitmap_api_call_edge:
                    call_list.append(f'{the_path[i]} calls {the_path[i + 1]}, ')
                    call_list_tuple.append((the_path[i], the_path[i + 1]))

        return call_list, call_list_tuple