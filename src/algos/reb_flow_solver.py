import os
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str

def solveRebFlow(env,res_path,desiredAcc,CPLEXPATH):
    t = env.time
    accRLTuple = [(n,int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n,int(env.acc[n][t+1])) for n in env.acc]
    edgeAttr = [(i,j,env.G.edges[i,j]['time']) for i,j in env.G.edges]
    modPath = os.getcwd().replace('\\','/')+'/src/cplex_mod/'
    OPTPath = os.getcwd().replace('\\','/')+'/saved_files/cplex_logs/rebalancing/'+res_path + '/'
    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)
    datafile = OPTPath + f'data_{t}.dat'
    resfile = OPTPath + f'res_{t}.dat'
    with open(datafile,'w') as file:
        file.write('path="'+resfile+'";\r\n')
        file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
        file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
        file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
    modfile = modPath+'minRebDistRebOnly.mod'
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file =  OPTPath + f'out_{t}.dat'
    with open(out_file,'w') as output_f:
        subprocess.check_call([CPLEXPATH+"oplrun", modfile, datafile], stdout=output_f, env=my_env)
    output_f.close()

    # 3. collect results from file
    flow = defaultdict(float)
    with open(resfile,'r', encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(';').split('=')
            if item[0] == 'flow':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                        continue
                    i,j,f = v.split(',')
                    flow[int(i),int(j)] = float(f)
    action = [flow[i,j] for i,j in env.edges]
    return action