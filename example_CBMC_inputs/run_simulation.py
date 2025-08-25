import os,sys
import npmc.move_class
import npmc.simulation_class as simc
#import npmc.eval_functions as ev


init_file = os.path.abspath('./system.in')
datafile = os.path.abspath('./system.data')
dumpfile = os.path.abspath('./trajectory.xyz')
temp=298.15

sim = simc.Simulation(init_file,datafile,dumpfile,temp,anchortype=5,max_disp=0.4,jump_dists=[0.93,1.95],numtrials_jump=20,type_lengths=(13,6),moves=[1,10,1,10,1],read_pdf=False)
print("Energy before exclude_type is "+str(sim.get_total_PE()))
sim.exclude_type(1,1)
print("Energy after exclude_type is "+str(sim.get_total_PE()))
sim.minimize(max_iter=500)
sim.initial_PE = sim.get_total_PE()

numsteps=2000000
for i in range(numsteps):
    if((i%10000)==0):
        sim.dump_atoms()
    sim.perform_mc_move(temp=temp)
