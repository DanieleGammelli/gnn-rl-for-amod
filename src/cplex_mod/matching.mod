/*********************************************
 * OPL 12.10.0.0 Model
 * Author: yangk
 * Creation Date: Aug 25, 2020 at 10:45:21 AM
 *********************************************/
tuple demandAttrTuple{
  	int i;
  	int j;
  	float v;
  	float p;
}

tuple Edge{
  int i;
  int j;
}



tuple accTuple{
  int i;
  float v;
}

string path = ...;
{demandAttrTuple} demandAttr = ...;
{accTuple} accInitTuple = ...;

{Edge} demandEdge = {<i,j>|<i,j,v,p> in demandAttr};
{int} region = {i|<i,v> in accInitTuple};
float accInit[region] = [i:v|<i,v> in accInitTuple];
float demand[demandEdge] = [<i,j>:v|<i,j,v,p> in demandAttr];
float price[demandEdge] = [<i,j>:p|<i,j,v,p> in demandAttr];
dvar float+ demandFlow[demandEdge];
maximize(sum(e in demandEdge) demandFlow[e]*price[e]);
subject to
{

    forall(i in region)    
    	0 <= accInit[i] - sum(e in demandEdge: e.i==i)demandFlow[e];
 	
     forall(e in demandEdge)
      		demandFlow[e] <= demand[e];  
}

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.demandEdge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.demandFlow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}




