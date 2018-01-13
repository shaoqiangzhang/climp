/*===================================================================
CLIMP: clustering via clique merging with OpenMP parallel design
Author: Shaoqiang Zhang, April-2016
Email: zhangshaoqiang@mail.tjnu.edu.cn
==================================================================*/
#include<map>
#include<string>
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<omp.h> //***********omp*******************
//#define NUM_THREADS 4 
//==================================================================
using namespace std;
typedef vector<vector<string> > MatrixStr;
typedef map<string,double> hashmap;
typedef map<int,int> intmap;
//typedef map<string,vector<int>> vecthashmap;
typedef map<string,hashmap> hash_of_hash;
//==================================================================
int main(int argc, const char** argv){
    int densitycutoff=100;//the density cutoff  of the constructed graph
    int threadnum=1;
    double similaritycutoff=0;
    double bin=0.05;
    double mergecutoffA=0.5; //mergecutoffA<=mergecutoffB
    double mergecutoffB=0.5;
    if(argc!=2 && argc!=4 && argc!=6 && argc!=8){
        cout<<"\n******\nUSAGE:\n******\n";
        cout<<argv[0]<<" <Input_File> [OPTIONS]  > Output_File\n";
        cout<<"\n<Input_File>\tfile containing three columns: LABEL1<tab>LABEL2<tab>Similarity_Value(0<=s<=1)\n";
        cout<<"\nOPTIONS:\n";
        cout<<"-d\t\tupper bound of graph density(default="<<densitycutoff<<")\n";
        cout<<"-s\t\tcutoff of similarity values(default="<<similaritycutoff<<")\n";
	cout<<"-t\t\tnumber of threads (default=1)\n";
	    cout<<"\n*******\nDesigned by Shaoqiang Zhang (zhangshaoqiang@mail.tjnu.edu.cn),  2014\n";
        cout<<endl;
        exit(1);
    }
    ifstream simfile(argv[1]);
    for(int i=2;i<argc-1;i=i+2){
        string strindex(argv[i]);
        if(strindex=="-s"){
            string ssim(argv[i+1]);
            istringstream issim(ssim);
            issim>>similaritycutoff;
            if(similaritycutoff<0){
                cout<<"ERROR: the wrong parameter of '-s', please input a number(>=0)."<<endl;
                exit(1);
            }
        }else if(strindex=="-d"){
            string sdensity(argv[i+1]);
            istringstream isdensity(sdensity);
            isdensity>>densitycutoff;
            if(densitycutoff<0){
                cout<<"ERROR: the wrong parameter of '-d', please input an integer(>=0)."<<endl;
                exit(1);
            }
	}else if(strindex=="-t"){
	  string sthread(argv[i+1]);
	  istringstream isthread(sthread);
	  isthread>>threadnum;
	  if(threadnum<1){
	      cout<<"ERROR: the wrong parameter of '-t', please input an integer(>=1)."<<endl;
	      exit(1);
	  }
        }else{
            cout<<"ERROR: the wrong settings of parameters. please type the program name to see the detail!"<<endl;
            exit(1);
        }
    }
    cout<<"The command line: ";
    for(int j=0;j<argc;j++){
        cout<<argv[j]<<" ";
    }
    cout<<"\n\n";
    //=============================end of input command line===========

    time_t tstart,tend;
    time(&tstart);//start to keep running time

    string leftnode, rightnode;
    double simscore;//a pair of nodes with the corresponding similarity score

    hash_of_hash nodesHashofHash;
    nodesHashofHash.clear();
    hashmap subhash;
    subhash.clear();

    //==============the following is input file reading and similarity graph construction==========
	
    for(string s; getline(simfile, s);){
        istringstream sin(s);
        sin>>leftnode>>rightnode>>simscore;
        if(simscore>=similaritycutoff){
            if(nodesHashofHash.count(leftnode)<1){
                subhash.clear();
                subhash.insert(hashmap::value_type(rightnode, simscore));
                nodesHashofHash.insert(hash_of_hash::value_type(leftnode,subhash));
            }else{
                nodesHashofHash[leftnode][rightnode]=simscore;
            }
            if(nodesHashofHash.count(rightnode)<1){
                subhash.clear();
                subhash.insert(hashmap::value_type(leftnode,simscore));
                nodesHashofHash.insert(hash_of_hash::value_type(rightnode,subhash));
            }else{
                nodesHashofHash[rightnode][leftnode]=simscore;
            }
		}
    }//==================end of constructing  similarity graph ===================
	
    //====================the following is to calcluate the density of the graph==========
    int graphcount=0;
    while(1){
        int edgeNum=0;
	//***************************************************************************************
		omp_set_num_threads(threadnum); 
		#pragma omp parallel for reduction(+:edgeNum)
		for(int i=0;i<threadnum;i++)
		{
			int td=-1;
			for (hash_of_hash::const_iterator hh=nodesHashofHash.begin(); hh!=nodesHashofHash.end(); ++hh){
				td++;
				if(td%threadnum== i)
	//***************************************************************************************
					edgeNum=edgeNum+(hh->second.size());
			}
		}	
        graphcount++;
        int graphdensity=edgeNum/(2*nodesHashofHash.size());
        cout<<"Graph#"<<graphcount<<" density: "<<graphdensity<<" #Nodes="<<nodesHashofHash.size()<<" #Edges="<<edgeNum/2<<" (simlarity cutoff: "<<similaritycutoff<<")\n"<<endl;
        if(graphdensity>densitycutoff){//graph density cutoff
            if(similaritycutoff>=1){
                cout<<"WARNING: We cannot get a graph with density <"<<densitycutoff<<"\n"<<endl;
                break;//exit(1);
            }else{
                hash_of_hash tempNodeHashofHash;
	//***************************************************************************************
			#pragma omp parallel for 
			for(int i=0;i<threadnum;i++)
			{
				int td=-1;
				for (hash_of_hash::const_iterator hh=nodesHashofHash.begin(); hh!=nodesHashofHash.end(); ++hh){
					td++;
					if(td%threadnum== i){
    //***************************************************************************************
					hashmap tempNeighborNodeMap=hh->second;
					for(hashmap::const_iterator h=hh->second.begin();h!=hh->second.end();++h){
						if(h->second ==similaritycutoff){
							tempNeighborNodeMap.erase(h->first);
						}
					}
					if(!tempNeighborNodeMap.empty()){
						tempNodeHashofHash.insert(hash_of_hash::value_type(hh->first,tempNeighborNodeMap));
					}
					}
				}
			}
                nodesHashofHash.clear();
                nodesHashofHash=tempNodeHashofHash;
                tempNodeHashofHash.clear();
                similaritycutoff=similaritycutoff+bin;
            }
        }else{
            break;
        }
    }    //======above: calculate the density of the graph and update the graph if it has high density=======

    string currentNode;
    hashmap neighborNodeHash;
    int localNeighborNum;
    string currentNeighborNode;
    hashmap currentNeighborHash;
    int minDegree;
	int minSumWeightNodeDegree;//the degree of node with min sum of edge weights
    //string minDegreeNode;
	string minSumWeightNode;
    int removeNum;
    double currentEdgesWeightSum;
    double minWeight;
    MatrixStr matStr;//store all cliques, each row is a clique
    //--------------------------------------------------------------
    //====below=======find the max clique associated with each node===============
	//***************************************************************************************
	omp_set_num_threads(threadnum); 
	#pragma omp parallel for private(currentNode,neighborNodeHash,localNeighborNum,currentNeighborNode,currentNeighborHash, minDegree,minSumWeightNode, minSumWeightNodeDegree,removeNum, minWeight, currentEdgesWeightSum)
	for(int i=0;i<threadnum;i++)
	{
		int td=-1;
	for (hash_of_hash::const_iterator hh=nodesHashofHash.begin(); hh!=nodesHashofHash.end(); ++hh){
		td++;
		if(td%threadnum== i){
	//**************************************************************************
        currentNode=hh->first;
        neighborNodeHash=hh->second;
        if(neighborNodeHash.size()>=2 ){
            vector<string> currentClique;//define the clique associated with current node using vector
            currentClique.push_back(currentNode);
            //cout<<currentNode;//
            while(1){
                minDegree=neighborNodeHash.size();
				minSumWeightNodeDegree=neighborNodeHash.size();
                minWeight=static_cast<double>(minDegree);
                for(hashmap::const_iterator h=neighborNodeHash.begin();h!=neighborNodeHash.end();++h){
                    //cout<<"("<<h->first<<" -> "<<h->second<<")";//
                    localNeighborNum=1;
                    currentNeighborNode=h->first;
                    currentEdgesWeightSum=h->second;
                    hash_of_hash::iterator chh=nodesHashofHash.find(currentNeighborNode);
                    //cout<<chh->first;//
                    currentNeighborHash=chh->second;
                    for(hashmap::const_iterator ch=currentNeighborHash.begin();ch!=currentNeighborHash.end();++ch){
                        if(neighborNodeHash.count(ch->first)>0){
                            localNeighborNum++;
                            currentEdgesWeightSum=+ch->second;
                        }
                    }
                    //cout<<"degree: "<<localNeighborNum<<" ) ";//
					if(localNeighborNum<minDegree){
						minDegree=localNeighborNum;
					}
                    if(currentEdgesWeightSum<minWeight){
                        minSumWeightNodeDegree=localNeighborNum;
                        minSumWeightNode=currentNeighborNode;
                        minWeight=currentEdgesWeightSum;
                    }else if(currentEdgesWeightSum==minWeight){
                        if(localNeighborNum<minDegree){
                            minSumWeightNodeDegree=localNeighborNum;
                            minSumWeightNode=currentNeighborNode;
                            minWeight=currentEdgesWeightSum;
                        }
                    }

                }
                if(minDegree == neighborNodeHash.size()){
                    break;
                }else{
                    removeNum=neighborNodeHash.erase(minSumWeightNode);
                }
            }
            for(hashmap::const_iterator ih=neighborNodeHash.begin();ih!=neighborNodeHash.end();++ih){
                currentClique.push_back(ih->first);
               // cout<<" "<<ih->first;//
            }
			#pragma omp critical
            matStr.push_back(currentClique);
            //cout<<endl;//
        }
		}
    }
	}//=====above======find the max clique associated with each node===============

    vector<double> cliquesWeightVect;
    hashmap cneighborhash;
    int matstrsize=matStr.size();
    //**********************************************************************
    for(int cc=0;cc<matstrsize;++cc){
      cliquesWeightVect.push_back(0.0);
    }
    #pragma omp parallel for private(cneighborhash)
    //********************************************************************
    for(int ci=0;ci<matstrsize;++ci){
        double cliqWeightSum=0.0;
        for(int cj=0;cj<matStr[ci].size()-1;++cj){
            //cout<<matStr[ci][cj]<<" ";//print cliques;
            hash_of_hash::iterator cnode=nodesHashofHash.find(matStr[ci][cj]);
            cneighborhash=cnode->second;
            for(int ck=cj+1;ck<matStr[ci].size();++ck){
                hashmap::iterator cn=cneighborhash.find(matStr[ci][ck]);
                cliqWeightSum=cliqWeightSum+(cn->second);
            }
        }
        //cout<<matStr[ci][matStr[ci].size()-1]<<" ";//print cliques
        cliquesWeightVect[ci]=cliquesWeightVect[ci]+cliqWeightSum;//****************
        //cliquesWeightVect.push_back(cliqWeightSum);
        //cout<<cliqWeightSum<<endl;//print the sum of weights of each clique
    }//===above=====calculate the sum of weights for each clique===================

    double maxWeight;
    int maxWeightIndex;
    vector<int> MergedIndex_of_matStr;
    vector<string> cliqueVect;
    vector<string> quasicliqueVect;
    vector<string> unmatchVect;
    MatrixStr tempCliquesMatStr;
	MatrixStr tempClustMat;
    MatrixStr quasiCliquesMatrix;//use a matrix to store all quasi-cliques
    quasiCliquesMatrix.clear();//
    vector<double> tempCliquesWeightVect;
	vector<string> tempClustVect;


    while(cliquesWeightVect.size()){
        maxWeight=0;
        maxWeightIndex;
        for(int w=0;w<cliquesWeightVect.size();++w){
            if(maxWeight<cliquesWeightVect[w]){
                maxWeight=cliquesWeightVect[w];
                maxWeightIndex=w;
            }
        }//==above====find the best clique by sorting weights=====

        MergedIndex_of_matStr.clear();
        cliqueVect=matStr[maxWeightIndex];
        quasicliqueVect=matStr[maxWeightIndex];
        matStr.erase(matStr.begin()+maxWeightIndex);
        cliquesWeightVect.erase(cliquesWeightVect.begin()+maxWeightIndex);
        for(int mi=0;mi<matStr.size(); ++mi){
            int matchCount=0;
            unmatchVect.clear();
            for(int mj=0;mj<matStr[mi].size();++mj){
                bool judgeExistInMaxWeight=0;
                for(int mm=0;mm<cliqueVect.size();++mm){
                    if(cliqueVect[mm]==matStr[mi][mj]){
                        matchCount++;
                        break;
                    }
                }
                for(int mq=0;mq<quasicliqueVect.size();++mq){
                    if(quasicliqueVect[mq]==matStr[mi][mj]){
                        judgeExistInMaxWeight=1;
                        break;
                    }
                }
                if(!judgeExistInMaxWeight){
                    unmatchVect.push_back(matStr[mi][mj]);
                }
            }
            if(static_cast<double>(matchCount)/static_cast<double>(matStr[maxWeightIndex].size())>=mergecutoffA
                || static_cast<double>(matchCount)/static_cast<double>(matStr[mi].size())>=mergecutoffB){//merge cliques' cutoff???????????????????
                MergedIndex_of_matStr.push_back(mi);
                for(int mp=0;mp<unmatchVect.size();++mp){
                    quasicliqueVect.push_back(unmatchVect[mp]);
                }
            }
        }//===========merge the best clique with other cliques into a quasi-clique=======
        tempCliquesWeightVect.clear();
        tempCliquesMatStr.clear();
        for(int ms=0;ms<matStr.size();++ms){
            bool bl=0;
            for(int mn=0;mn<MergedIndex_of_matStr.size();++mn){
                if(ms == MergedIndex_of_matStr[mn]){
                    bl=1;
                    break;
                }
            }
            if(!bl){
                tempCliquesMatStr.push_back(matStr[ms]);
                tempCliquesWeightVect.push_back(cliquesWeightVect[ms]);
            }
        }
        matStr=tempCliquesMatStr;
        cliquesWeightVect=tempCliquesWeightVect;
        //==========update the vector of cliques (matStr)=======================
        quasiCliquesMatrix.push_back(quasicliqueVect);
    }

    //==============================================================
    hashmap redundantNodesHash;
    redundantNodesHash.clear();
	tempClustMat.clear();
	tempClustVect.clear();

    for(int qi=0;qi<quasiCliquesMatrix.size()-1;++qi){
		//cout<<"Cluster No."<<qi+1<<":";
        for(int qj=0;qj<quasiCliquesMatrix[qi].size();++qj){
			hashmap::iterator myelement=redundantNodesHash.find(quasiCliquesMatrix[qi][qj]);
			if(myelement==redundantNodesHash.end()){
				//cout<<" "<<quasiCliquesMatrix[qi][qj];
				tempClustVect.push_back(quasiCliquesMatrix[qi][qj]);
			}
            for(int pi=qi+1;pi<quasiCliquesMatrix.size();++pi){
                for(int pj=0;pj<quasiCliquesMatrix[pi].size();++pj){
                    if( quasiCliquesMatrix[qi][qj]==quasiCliquesMatrix[pi][pj]){
						//cout<<" "<<quasiCliquesMatrix[qi][qj];
                        if(redundantNodesHash.count(quasiCliquesMatrix[qi][qj])<1){
                            redundantNodesHash.insert(hashmap::value_type(quasiCliquesMatrix[qi][qj], 0));
							//cout<<" "<<quasiCliquesMatrix[qi][qj];
                        }
                        break;
                    }
                }
            }
        }
		//cout<<endl;
		if(tempClustVect.size()>0){
			tempClustMat.push_back(tempClustVect);
		}
		tempClustVect.clear();
    }
	//================================================
	cout<<"\nThe following are the clusters without redundant nodes:(each line corresponds to a cluster)\n";
	intmap indexhash;
	indexhash.clear();
	int vectindex=0;
	for(int wi=0;wi<tempClustMat.size();++wi){
		int tempvectsize=0;
	//**********************************************************************
		#pragma omp parallel for
	//**********************************************************************
		for(int vi=0;vi<tempClustMat.size();++vi){
			intmap::iterator emt=indexhash.find(vi);
			if(emt==indexhash.end()){
				if(tempClustMat[vi].size()>tempvectsize){
					tempvectsize=tempClustMat[vi].size();
					vectindex=vi;
				}
			}
		}
		if(indexhash.count(vectindex)<1){
			indexhash.insert(intmap::value_type(vectindex,0));
			//cout<<vectindex<<endl;
			cout<<tempClustMat[vectindex][0];
	//**********************************************************************
			//#pragma omp parallel for
	//**********************************************************************
			for(int y=1;y<tempClustMat[vectindex].size();++y){
				cout<<" "<<tempClustMat[vectindex][y]<<" ";
			}
			cout<<endl;//==print each cluster======
		}
	}
	
    //cout<<"\n\n#Redundant nodes="<<redundantNodesHash.size()<<endl;
    //=====below: record the running time==============
    time(&tend);
    double dif = difftime (tend,tstart);
    cout<<"\n\nTotal running time: "<<dif<<" seconds\n"<<endl;
}

