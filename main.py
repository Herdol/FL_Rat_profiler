'''
The main function to be called

Info and update notes will be added. 

'''
   
def train(model):
    model.buffer_empty()
    for _ in epochs:
        s = env.reset()
        t = 0
        avg_loss, total_r = 0,0
        while t<max_time_steps:
            actions = model.act(s)
            next_s, r, d, info = env.step(actions)
            total_r += r
            model.add_to_buffer(s, a, r, d, next_s)
            loss = model.train()
        logger.log({'loss': , 'reward': total_r})
            
            
if main == init: 
    algorithm = 'fedavg'
logger = Logger()
if algorithm == 'fedavg':
    model = FedAvg()
elif algorithm == 'single_agent':
    model = DQN()
     train(model)
     model.save('my_nobel_prize.pt')