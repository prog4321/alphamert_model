# Original version created by prog4321, Aug 2024 ========================================
# Modified version created by prog4321, Jun 2025 ========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
from dotenv import load_dotenv
import psycopg2
import os
import sys
from decimal import Decimal


prediction_num_col = 0
route_id_col = 1
valid_route_col = 2
node_count_col = 3
total_cost_col = 4

# =======================================================================================
class BestRouteQLearner(object):

    def __init__(self,
                alpha=0.9,
                gamma=0.9,
                epochs=30_000,
                reward_coef=7,
                is_peak_hour=False,
                peak_hour_cost=2,
                off_peak_cost=5,
                show_dev_msg=False):

        self.alpha = alpha
        self.gamma = gamma
        self.epochs = epochs
        self.reward_coef = reward_coef

        if is_peak_hour == True:
            self.wait_cost = peak_hour_cost
        else:
            self.wait_cost = off_peak_cost

        self.show_dev_msg = show_dev_msg

        self.running_grid_search = False

    def get_available_actions(self, state):

        current_state_row = self.R[state,:]
        available_actions = np.where(current_state_row >= 0)[0]
        if len(available_actions) > 0:
            return True, available_actions
        else:
            return False, None

    def select_next_action(self, available_actions):

        next_action = np.random.choice(available_actions, 1)[0]
        return next_action

    def update_Q_table(self, state, action, end_node):

        next_state = action
        next_state_max_value = np.max(self.Q[next_state, :])

        # reward = self.R[state, action] + self.calc_reward(state, action, end_node)
        reward = self.R[state, action] + float(self.calc_reward(state, action, end_node))

        self.Q[state, action] = self.Q[state, action] + \
                            self.alpha * (reward + (self.gamma * next_state_max_value) - \
                            self.Q[state, action])

        if np.max(self.Q) > 0:
            return(np.sum(self.Q/np.max(self.Q)*100))
        else:
            return(0)
    
    def get_nett_cost(self, cost, state, action, end_node):

        if self.is_interchange_transfer(state, action) == True:
            if action != end_node:
                cost += self.wait_cost
        return cost

    def get_cost(self, state, action, end_node):
        
        index = np.where((self.routes == [state, action]).all(axis=1))[0]
        if len(index) > 0:
            cost = self.costs[index][0][0]
            return self.get_nett_cost(cost, state, action, end_node)

        index = np.where((self.routes == [action, state]).all(axis=1))[0]
        if len(index) > 0:
            cost = self.costs[index][0][0]
            return self.get_nett_cost(cost, state, action, end_node)

    def calc_reward(self, state, action, end_node):

        reward = self.reward_coef * (1 - (self.get_cost(state, action, end_node) / self.cost_max))
        return reward
    
    def is_interchange_transfer(self, state, action):

        if (state in self.interchanges) and (action in self.interchanges):
            if self.get_node_name(self.encoder.inverse_transform([state])[0]) == \
                self.get_node_name(self.encoder.inverse_transform([action])[0]):
                return True

        return False

    def get_node_name(self, node_id):

        node_name = self.orig_nodes[np.flatnonzero(self.orig_nodes[:,0]==node_id),1][0]
        return node_name
    
    def get_alt_interchange(self, node_id):

        interchange = [i for i in self.interchanges if node_id in i]
        if len(interchange) > 0:
            interchange = [i for i in interchange[0] if pd.notnull(i)]
            interchange.remove(node_id)
            if len(interchange) > 1:
                interchange = np.random.choice(interchange, 1)[0]
            else:
                interchange = interchange[0]
        else:
            interchange = node_id

        return interchange

    def generate_consolidated_logs(self, total_predictions):
        self.consolidated_route_log = []
        self.consolidated_cost_log = []

        self.perf_log = np.zeros((total_predictions, 5))
        self.perf_log_ctr = 0
    
    def update_consolidated_logs(self, route_id, is_valid_route, best_route, cost_log):

        if is_valid_route == True:
            self.consolidated_route_log.append(best_route)
            self.consolidated_cost_log.append(cost_log)

            self.perf_log[self.perf_log_ctr, prediction_num_col] = self.prediction_count
            self.perf_log[self.perf_log_ctr, route_id_col] = route_id
            self.perf_log[self.perf_log_ctr, valid_route_col] = True
            self.perf_log[self.perf_log_ctr, node_count_col] = len(cost_log)
            self.perf_log[self.perf_log_ctr, total_cost_col] = sum(cost_log)
        else:
            self.consolidated_route_log.append([])
            self.consolidated_cost_log.append([])

            self.perf_log[self.perf_log_ctr, prediction_num_col] = self.prediction_count
            self.perf_log[self.perf_log_ctr, route_id_col] = route_id
            self.perf_log[self.perf_log_ctr, valid_route_col] = False
        
        self.perf_log_ctr += 1

    def update_prediction_count(self, total_predictions):
        
        self.prediction_count += 1
        if self.running_grid_search == False:
            print(f'Running prediction {self.prediction_count} of {total_predictions}')

    def generate_ref_tables(self, end_node):

        self.generate_R_table(end_node)

        self.generate_Q_table(end_node)

    def generate_R_table(self, end_node):

        self.R = np.ones((self.table_size, self.table_size))
        self.R *= -1
        path_score = 0
        goal_score = 100

        for node in self.routes:
            if node[1] == end_node:
                self.R[node[0], node[1]] = goal_score
                self.R[node[1], node[0]] = path_score
            elif node[0] == end_node:
                self.R[node[0], node[1]] = path_score
                self.R[node[1], node[0]] = goal_score
            else:
                self.R[node[0], node[1]] = path_score
                self.R[node[1], node[0]] = path_score

    def generate_Q_table(self, end_node):

        self.Q = np.zeros((self.table_size, self.table_size))

        self.score_log = []

        for i in range(self.epochs):
            
            state = np.random.randint(0, self.table_size)
            has_available_actions, available_actions = self.get_available_actions(state)
            
            if has_available_actions == True:
                action = self.select_next_action(available_actions)
                score = self.update_Q_table(state, action, end_node)
                self.score_log.append(score)
        
        if self.show_dev_msg == True:
            # DISPLAY CONVERGENCE GRAPH
            fig, ax = plt.subplots(figsize=(4.4, 3.2))
            ax.tick_params(axis='both', which='major', labelsize=6)
            plt.plot(self.score_log)
            plt.xlabel('Training Epochs', fontsize=7)
            plt.ylabel('Sum of normalised Q values', fontsize=7)
            plt.title(f'Prediction {self.prediction_count}: Convergence graph of sum of normalised Q values',
                fontsize=8)
            plt.show()

    def get_best_route(self, start_node, end_node):

        self.generate_ref_tables(end_node)

        state = start_node
        best_route = [state]
        cost_log = []
        is_valid_route = True

        if start_node == end_node:
            cost_log.append(0)

        while state != end_node:

            state_row = self.Q[state,:]
            available_index = np.where(state_row > 0)[0]
            available_index = np.array([i for i in available_index if i not in best_route])

            if len(available_index) == 1:
                next_state = available_index[0]
            elif len(available_index) > 0:
                available_state_values = np.array([i for i in state_row[available_index]])
                max_index = \
                    np.flatnonzero(available_state_values == np.max(available_state_values))
                if len(max_index) == 1:
                    next_state = available_index[max_index][0]
                elif len(max_index) > 1:
                    max_index = np.random.choice(max_index, 1)[0]
                    next_state = available_index[max_index]
            else:
                is_valid_route = False
                break

            best_route.append(next_state)

            if state == start_node:
                if self.is_interchange_transfer(state, next_state) == True:
                    cost_log.append(0)
                else:
                    cost_log.append(self.wait_cost)

            cost = self.get_cost(state, next_state, end_node)
            cost_log.append(cost)

            state = next_state
        # End of while loop

        if is_valid_route == True:
            return True, best_route, cost_log
        else:
            return False, None, None

    def fit(self,
            nodes,
            routes,
            interchanges,
            node_id_alias=None,
            node_name_alias=None,
            cost_alias=None,
            cost_unit=None):
        
        print('Fitting the model')

        self.encoder = LabelEncoder()

        df_nodes = pd.DataFrame(nodes)
        df_nodes = df_nodes.set_axis(['node_id', 'node_name'], axis=1)
        self.df_nodes = df_nodes
        self.orig_nodes = df_nodes.values

        self.nodes = nodes[:, 0]
        self.nodes = self.encoder.fit_transform(self.nodes)
        self.nodes = self.nodes.reshape(-1, 1)

        route_start_nodes = routes[:,0]
        route_start_nodes = self.encoder.transform(route_start_nodes)
        route_start_nodes = route_start_nodes.reshape(-1, 1)
        route_end_nodes = routes[:,1]
        route_end_nodes = self.encoder.transform(route_end_nodes)
        route_end_nodes = route_end_nodes.reshape(-1, 1)
        self.routes = np.hstack((route_start_nodes, route_end_nodes))

        costs = routes[:,2]
        self.costs = costs.reshape(-1, 1)

        self.interchanges = interchanges
        interchange_col_count = self.interchanges.shape[1]
        # Encode the interchanges
        for interchange in self.interchanges:
            for index in range(interchange_col_count):
                if pd.notnull(interchange[index]):
                    interchange[index] = self.encoder.transform([interchange[index]])[0]

        self.node_id_alias = node_id_alias
        self.node_name_alias = node_name_alias
        self.cost_alias = cost_alias
        self.cost_unit = cost_unit

        self.table_size = self.nodes.shape[0]

        self.cost_max = np.max(self.costs) + self.wait_cost

    def predict(self, start_node, end_node):

        self.prediction_count = 0 # Reset the prediction count every time this is run

        if start_node not in self.orig_nodes[:,0]:
            print('The Start Node ' + start_node + \
                ' does not exist in the dataset.')
            return False, None
        
        if end_node not in self.orig_nodes[:,0]:
            print('The End Node ' + end_node + \
                ' does not exist in the dataset.')
            return False, None
        
        if start_node == end_node:
            print('The Start Node ' + start_node + \
                ' is the same as the End Node ' + end_node + '.')
            return False, None

        start_node = self.encoder.transform([start_node])[0]
        orig_end_node = end_node
        end_node = self.encoder.transform([end_node])[0]
# =================================================================================================
        forward_route_id = 0
        reverse_route_id = 1
        alt_start_intchg_route_id = 2
        alt_end_intchg_route_id = 3

        if start_node not in self.interchanges and end_node not in self.interchanges:

            total_predictions = 2

            self.generate_consolidated_logs(total_predictions)

            # DEFAULT FORWARD ROUTE
            self.update_prediction_count(total_predictions)
                
            is_valid_route, best_route, cost_log = \
                self.get_best_route(start_node, end_node)

            self.update_consolidated_logs(forward_route_id, is_valid_route,
                best_route, cost_log)
            
            # REVERSE ROUTE
            self.update_prediction_count(total_predictions)
                
            is_valid_route, best_route, cost_log = \
                self.get_best_route(end_node, start_node)

            if is_valid_route == True:
                best_route.reverse()
                cost_log.reverse()
                wait_cost = cost_log[-1]
                cost_log.pop(-1)
                cost_log.insert(0, wait_cost)

            self.update_consolidated_logs(reverse_route_id, is_valid_route,
                best_route, cost_log)

        else:

            if start_node in self.interchanges and end_node in self.interchanges:
                total_predictions = 3
            else:
                total_predictions = 2
            
            self.generate_consolidated_logs(total_predictions)

            # DEFAULT FORWARD ROUTE
            self.update_prediction_count(total_predictions)

            is_valid_route, best_route, cost_log = \
                self.get_best_route(start_node, end_node)
            
            self.update_consolidated_logs(forward_route_id, is_valid_route,
                best_route, cost_log)

            # ALTERNATE START INTERCHANGE ROUTE
            if start_node in self.interchanges:
                self.update_prediction_count(total_predictions)

                alt_interchange = self.get_alt_interchange(start_node)
                if alt_interchange == end_node:
                    is_valid_route, best_route, cost_log = \
                        self.get_best_route(start_node, end_node)
                else:
                    is_valid_route, best_route, cost_log = \
                        self.get_best_route(alt_interchange, end_node)
                    if is_valid_route == True:
                        best_route.insert(0, start_node)
                        cost_log.insert(0,0)
                        cost_log[1] = self.get_cost(start_node, alt_interchange, end_node)
                
                self.update_consolidated_logs(alt_start_intchg_route_id, is_valid_route,
                    best_route, cost_log)

            # ALTERNATIVE END INTERCHANGE ROUTE
            if end_node in self.interchanges:
                self.update_prediction_count(total_predictions)

                alt_interchange = self.get_alt_interchange(end_node)

                is_valid_route, best_route, cost_log = \
                    self.get_best_route(start_node, alt_interchange)
                if is_valid_route == True:
                    best_route.append(end_node)
                    cost_log.append(self.get_cost(alt_interchange, end_node, end_node))

                self.update_consolidated_logs(alt_end_intchg_route_id, is_valid_route,
                    best_route, cost_log)
# =================================================================================================
        if self.show_dev_msg == True:

            print('\nPERFORMANCE LOG:')
            df_perf_log = pd.DataFrame(self.perf_log)
            df_perf_log.columns = ['Prediction', 'Route ID', 'Valid Route', 'Steps', 'Duration']
            print(df_perf_log.to_string(index=False))

            ctr = 1
            decoded_route_log = \
                [self.encoder.inverse_transform(i) for i in self.consolidated_route_log]
            print('\nPREDICTED ROUTES:')
            for i in decoded_route_log:
                print(f'Prediction {ctr}: {i}')
                ctr += 1
# =================================================================================================
        route = self.perf_log[np.where(self.perf_log[:,valid_route_col]==True)[0]]
        if len(route) > 0:
            # Get rows with the minimum total cost
            index = np.where(route[:,total_cost_col]==np.min(route[:,total_cost_col]))[0]
            route = route[index,:]

            # Get rows with the minimum number of nodes
            index = np.where(route[:,node_count_col]==np.min(route[:,node_count_col]))[0]
            route = route[index,:]

            # Get the prediction number of the first of the remaining entries
            index = int(route[0][0]) -1
        else:
            return False, None

        best_route, cost_log = self.consolidated_route_log[index], self.consolidated_cost_log[index]

        best_route = [self.encoder.inverse_transform([i])[0] for i in best_route]

        self.total_cost = sum(cost_log)

        best_route_dict = {
                        'node_id': best_route,
                        'cost': cost_log
                        }
        df_best_route = pd.DataFrame(best_route_dict)
        df_best_route = df_best_route.merge(self.df_nodes, how='left', on='node_id')
        df_best_route = df_best_route[['node_id', 'node_name', 'cost']]

        if self.node_id_alias != None:
            node_id_alias = self.node_id_alias
        else:
            node_id_alias = 'ID'
        
        if self.node_name_alias != None:
            node_name_alias = self.node_name_alias
        else:
            node_name_alias = 'Name'
        
        if self.cost_alias != None:
            cost_alias = self.cost_alias
        else:
            cost_alias = 'Cost'

        if self.cost_unit != None:
            cost_header = cost_alias + ' (' + self.cost_unit + ')'
        else:
            cost_header = cost_alias

        df_best_route = df_best_route.set_axis([node_id_alias, node_name_alias,
                        cost_header], axis=1)
        df_best_route.index = np.arange(1, len(df_best_route)+1)


        index = np.arange(1, len(df_best_route)+1)
        tmp_route_dict = {
                        'route_index': index,
                        'node_id': best_route,
                        'cost': cost_log
                        }
        df_tmp_route = pd.DataFrame(tmp_route_dict)
        df_tmp_route = df_tmp_route[['route_index', 'node_id', 'cost']]



        # return True, df_best_route
        return True, df_tmp_route
# =================================================================================================


# CODE TO CONNECT TO POSTGRES DATABASE ===============================================
def db_connect():
    # Connect to database
    conn = None
    # Look for the following database connection parameters in the ".env" file
    # located inside the same directory
    conn = psycopg2.connect(
            host=os.environ['DB_HOST'],
            database=os.environ['DB_NAME'],
            user=os.environ['DB_USERNAME'],
            password=os.environ['DB_PASSWORD'],
            port=os.environ['DB_PORT'])
    return conn

def get_sql_results(conn, query):
   # Import data from a PostgreSQL database using a SELECT query
    cursor = conn.cursor()
    cursor.execute(query)
    # The execute returns a list of tuples:
    tuples_list = cursor.fetchall()
    cursor.close()
    # Now we need to transform the list into a pandas DataFrame:
    df = pd.DataFrame(tuples_list)
    return df

def execute_sql(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()

    cursor.close()
# ====================================================================================


show_dev_msg = False
is_peak_hour = False

run_grid_search = False

if is_peak_hour == True:
    period_text = 'peak hour'
else:
    period_text = 'off-peak'

node_id_col = 0
node_name_col = 1

start_node_id_col = 0
end_node_id_col = 1
cost_col = 2

def run_prediction():

    if show_dev_msg == True:
        start_time = datetime.datetime.now()

    print('Initialising the model')

    load_dotenv()
    conn = db_connect()

    query_nodes = "SELECT * FROM alphamert.nodes ORDER BY id;"
    query_routes = "SELECT * FROM alphamert.routes;"
    query_interchanges = "SELECT * FROM alphamert.interchanges;"

    df_nodes = get_sql_results(conn, query_nodes)
    df_routes = get_sql_results(conn, query_routes)
    df_interchanges = get_sql_results(conn, query_interchanges)

    nodes = df_nodes.iloc[:, [node_id_col, node_name_col]].values
    routes = df_routes.iloc[:, [start_node_id_col, end_node_id_col, cost_col]].values

    interchanges_start_col = 1
    interchanges_col_count = df_interchanges.shape[1]
    interchanges = df_interchanges.iloc[:, interchanges_start_col:interchanges_col_count].values


    # CREATE AN INSTANCE OF THE BestRouteQLearner CLASS AND PREDICT THE BEST PATH ============
    is_peak_hour_list = [True, False]
    for i in is_peak_hour_list:
        is_peak_hour = i

        brq = BestRouteQLearner(alpha=0.9,
                            gamma=0.9,
                            epochs=30_000,
                            reward_coef=7,
                            is_peak_hour=is_peak_hour,
                            show_dev_msg=show_dev_msg)
    
        brq.fit(nodes=nodes,
                routes=routes,
                interchanges=interchanges,
                node_id_alias='Station Code',
                node_name_alias='Station Name',
                cost_alias='Duration',
                cost_unit='min')
        
        query_start_nodes_list = f"SELECT id FROM alphamert.unresolved_nodes ORDER BY id;"
        df_start_nodes_list = get_sql_results(conn, query_start_nodes_list)
        df_start_nodes_list.columns = ['id']
        for start_row in df_start_nodes_list.itertuples():
            start_node = start_row.id

            query_end_nodes_list = f"SELECT id FROM alphamert.nodes WHERE NOT id = '{start_node}' ORDER BY id;"
            df_end_nodes_list = get_sql_results(conn, query_end_nodes_list)
            df_end_nodes_list.columns = ['id']
            for end_row in df_end_nodes_list.itertuples():
                end_node = end_row.id

                is_valid_route, df_best_route = \
                    brq.predict(start_node=start_node, end_node=end_node)

                if is_peak_hour:
                    is_peak_hour_flag = "TRUE"
                else:
                    is_peak_hour_flag = "FALSE"

                if is_valid_route:

                    query_route_mapping = f"SELECT id FROM alphamert.route_mappings WHERE start_node_id = '{start_node}' AND end_node_id = '{end_node}' AND is_peak_hour = {is_peak_hour_flag};"
                    df_route_mapping = get_sql_results(conn, query_route_mapping)

                    if len(df_route_mapping) == 0:
                        query_insert_route_mapping = f"INSERT INTO alphamert.route_mappings (start_node_id, end_node_id, is_peak_hour) VALUES ('{start_node}', '{end_node}', {is_peak_hour_flag});"
                        execute_sql(conn, query_insert_route_mapping)

                        df_route_mapping = get_sql_results(conn, query_route_mapping)

                    route_mapping_id = df_route_mapping[0].iloc[0]

                    query_delete_best_route = f"DELETE FROM alphamert.best_routes WHERE route_mapping_id = {route_mapping_id};"
                    execute_sql(conn, query_delete_best_route)

                    record_count = len(df_best_route)
                    ctr = 0

                    query_insert_best_route = 'INSERT INTO alphamert.best_routes (route_mapping_id, route_index, node_id, cost) VALUES '
                    for row in df_best_route.itertuples():
                        ctr += 1
                        query_insert_best_route += f"({route_mapping_id}, '{row.route_index}', '{row.node_id}', {row.cost})"
                        if ctr != record_count:
                            query_insert_best_route += ','
                        else:
                            query_insert_best_route += ';'
                    execute_sql(conn, query_insert_best_route)

                    print(f"Best route found for {start_node} to {end_node}")

            query_delete_resolved_node = f"DELETE FROM alphamert.unresolved_nodes WHERE id = '{start_node}';"
            execute_sql(conn, query_delete_resolved_node)

    print("Route-finding process completed.")
    conn.close()

run_prediction()