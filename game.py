
# IMPORTS ------------------------------------------------------------------------------------------------------------#

import random, copy, time
from   template     import GameState
from   func_timeout import func_timeout, FunctionTimedOut
from   template     import Agent as DummyAgent
    
# CONSTANTS ----------------------------------------------------------------------------------------------------------#

FREEDOM = False  #Whether or not to penalise agents for incorrect moves and timeouts. Useful for debugging.
WARMUP  = 15    #Warmup period (time given to each agent on their first turn).

# CLASS DEF ----------------------------------------------------------------------------------------------------------#

class Game:
    def __init__(self, GameRule,
                 agent_list, 
                 num_of_agent,
                 seed=1, 
                 time_limit=1, 
                 warning_limit=3, 
                 displayer = None, 
                 agents_namelist = ["Alice","Bob"],
                 interactive=False):
        
        self.seed = seed
        random.seed(self.seed)
        self.seed_list = [random.randint(0,int(1e10)) for _ in range(1000)]
        self.seed_idx = 0

        # Make sure we are forming a valid game, and that agent
        # id's range from 0 to N-1, where N is the number of agents.
        # assert(len(agent_list) <= 4)
        # assert(len(agent_list) > 1)
        i = 0
        for plyr in agent_list:
            assert(plyr.id == i)    
            i += 1

        self.game_rule = GameRule(num_of_agent)
        self.gamemaster = DummyAgent(num_of_agent) #GM/template agent used by some games (e.g. Azul, for signalling rounds).

        # need to handle the same without needed a validAction function
        # self.valid_action = self.game_rule.validAction
        if hasattr(type(self.game_rule), 'validAction') and callable(self.game_rule.validAction):
            self.valid_action = self.game_rule.validAction
        else:
            self.valid_action = None

        self.agents = agent_list
        self.agents_namelist = agents_namelist
        self.time_limit = time_limit
        self.warning_limit = warning_limit
        self.warnings = [0]*len(agent_list)
        self.warning_positions = []
        self.displayer = displayer
        if self.displayer is not None:
            self.displayer.InitDisplayer(self)
        self.interactive = interactive

    def _EndGame(self,num_of_agent,history, isTimeOut = True, id = None):
        history.update({"seed":self.seed,
                        "num_of_agent":num_of_agent,
                        "agents_namelist":self.agents_namelist,
                        "warning_positions":self.warning_positions,
                        "warning_limit":self.warning_limit})
        history["scores"]= {i:0 for i in range(num_of_agent)}
        if isTimeOut:
            history["scores"][id] = -1
        else:
            for i in range(num_of_agent):
                history["scores"].update({i:self.game_rule.calScore(self.game_rule.current_game_state,i)})

        if self.displayer is not None:
            self.displayer.EndGame(self.game_rule.current_game_state,history["scores"])
        return history

    def Run(self):
        history = {"actions":[]}
        action_counter = 0
        while not self.game_rule.gameEnds():
            agent_index = self.game_rule.getCurrentAgentIndex()
            agent = self.agents[agent_index] if agent_index < len(self.agents) else self.gamemaster
            game_state = self.game_rule.current_game_state
            game_state.agent_to_move = agent_index
            actions = self.game_rule.getLegalActions(game_state, agent_index)
            actions_copy = copy.deepcopy(actions)
            gs_copy = copy.deepcopy(game_state)
            
            # Delete all specified attributes in the agent state copies, if this isn't a perfect information game.
            if self.game_rule.private_information:
                delattr(gs_copy.deck, 'cards') # Upcoming cards cannot be observed.
                for i in range(len(gs_copy.agents)):
                    if gs_copy.agents[i].id != agent_index:
                        for attr in self.game_rule.private_information:
                            delattr(gs_copy.agents[i], attr)
            
            #Before updating the game, if this is the first move, allow the displayer an initial update.
            #This is used by some games to run simple pre-game animations.
            if action_counter==0 and self.displayer is not None:
                self.displayer._DisplayState(self.game_rule.current_game_state)
                        
            #If interactive mode, update displayer and obtain action via user input.
            if self.interactive and agent_index==1:
                self.displayer._DisplayState(self.game_rule.current_game_state)
                selected = self.displayer.user_input(actions_copy)
                
            else:
                #If freedom is given to agents, let them return any action in any time period, at the risk of breaking 
                #the simulation. This can be useful for debugging purposes.
                if FREEDOM:
                    selected = agent.SelectAction(actions_copy, gs_copy)
                else:
                    #"Gamemaster" agent has an agent index equal to the number of player agents in the game.
                    #If the gamemaster acts (e.g. to start or end a round in Azul), let it do so uninhibited.
                    #Else, allow player agent to select action within a time limit. 
                    #- If it times out, display TimeOutWarning. 
                    #- If it returns an illegal move, display IllegalWarning.
                    #  - Illegal move checked by self.validaction(), if implemented by the game being run.
                    #  - Else, look for move in actions list by equality according to Python.
                    #If this is the agent's first turn, allow warmup time.
                    try: 
                        selected = func_timeout(WARMUP if action_counter < len(self.agents) else self.time_limit, 
                                                agent.SelectAction,args=(actions_copy, gs_copy))
                    except:
                        selected = "timeout"
                        
                    if agent_index != self.game_rule.num_of_agent:
                        if selected != "timeout":
                            if self.valid_action:
                                if not self.valid_action(selected, actions):
                                    selected = "illegal"
                            elif not selected in actions:
                                selected = "illegal"
                            
                        if selected in ["timeout", "illegal"]:
                            self.warnings[agent_index] += 1
                            self.warning_positions.append((agent_index,action_counter))
                            if self.displayer is not None:
                                if selected=="timeout":
                                    self.displayer.TimeOutWarning(self,agent_index)
                                else:
                                    self.displayer.IllegalWarning(self,agent_index)                        
                            selected = random.choice(actions)

                
            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1
            history["actions"].append({action_counter:{"agent_id":self.game_rule.current_agent_index,"action":selected}})
            action_counter += 1
            
            self.game_rule.update(selected)
            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1

            if self.displayer is not None:
                self.displayer.ExcuteAction(agent_index,selected, self.game_rule.current_game_state)

            if (agent_index != self.game_rule.num_of_agent) and (self.warnings[agent_index] == self.warning_limit):
                history = self._EndGame(self.game_rule.num_of_agent,history,isTimeOut=True,id=agent_index)
                return history
                
        # Score agent bonuses
        return self._EndGame(self.game_rule.num_of_agent,history,isTimeOut=False)
            

class GameReplayer:
    def __init__(self,GameRule,replay, displayer = None):
        self.replay = replay
                    
        self.seed = self.replay["seed"]
        random.seed(self.seed)
        self.seed_list = [random.randint(0,int(1e10)) for _ in range(1000)]
        self.seed_idx = 0

        self.num_of_agent = self.replay["num_of_agent"]
        self.agents_namelist = replay["agents_namelist"]
        self.warning_limit = replay["warning_limit"]
        self.warnings = [0]*self.num_of_agent
        self.warning_positions = replay["warning_positions"]
        self.game_rule = GameRule(self.num_of_agent)
        self.scores=replay["scores"]

        self.displayer = displayer
        if self.displayer is not None:
            self.displayer.InitDisplayer(self)           
  
    def Run(self):
        for item in self.replay["actions"]:
            (index, info), = item.items()
            selected = info["action"]
            agent_index = info["agent_id"]
            self.game_rule.current_agent_index = agent_index          

            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1
            self.game_rule.update(selected)
            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1
            if self.displayer is not None:
                if (agent_index,index) in self.warning_positions:
                    self.warnings[agent_index] += 1
                    self.displayer.TimeOutWarning(self,agent_index)
                self.displayer.ExcuteAction(agent_index,selected, self.game_rule.current_game_state)
            
        if self.displayer is not None:
            self.displayer.EndGame(self.game_rule.current_game_state,self.scores)
   
