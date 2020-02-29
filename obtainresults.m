close all
clear
clc

load('./alldata/pathallpatchc.mat');

% 
trainfeature = load('./Results/predicttrain.txt');
testfeature = load('./Results/predicttest.txt');
valfeature = load('./Results/predictval.txt');


trainfeature = trainfeature(:,2);
testfeature = testfeature(:,2);
valfeature = valfeature(:,2);

trainnum = unique(trainGroups);
for i=1:length(trainnum)
    ind = find(trainGroups==trainnum(i));
    temp1 = trainfeature(ind);
    trainpp(i,:) = mean(temp1);
    traintrue(i,:) = mean(trainlabel(ind));
    ss=trainpaths{ind(1)};
    ins = strfind(ss,'/'); ins2 =  strfind(ss,'.');   
    trainname{i} = ss(ins(end-1)+1:ins2(1)-1);
end
trainname=trainname';
testnum = unique(testGroups);
for i=1:length(testnum)
    ind = find(testGroups==testnum(i));
    temp1 = testfeature(ind);
    testpp(i,:) = mean(temp1);
    testtrue(i,:)=mean(testlabel(ind));

    ss=testpaths{ind(1)};
    ins = strfind(ss,'/'); ins2 =  strfind(ss,'.'); 
    testname{i} = ss(ins(end-1)+1:ins2(1)-1);
end
testname=testname';
evetrain = EvaluationModel(trainpp,traintrue,1,1)
evetest = EvaluationModel(testpp,testtrue,1,1)


valnum = unique(valGroups);
for i=1:length(valnum)
    ind = find(valGroups==valnum(i));
    temp1 = valfeature(ind);
    valpp(i,:) = mean(temp1);
    valtrue(i,:)=mean(vallabels(ind));
    

    ss=valpaths{ind(1)};
    ins = strfind(ss,'/'); ins2 = strfind(ss,'.'); 
    valname{i} = ss(ins(end-1)+1:ins2(end)-1);
    

end
valname=valname';
eveval = EvaluationModel(valpp,valtrue,1,1)

evetogether = EvaluationModel([testpp;valpp],[testtrue;valtrue],1,1)


 [~,cutoff1,~,~] = AllAuc (trainpp,traintrue)

 [~,cutoff2,~,~] = AllAuc (testpp,testtrue) 
 
 [~,cutoff3,~,~] = AllAuc (valpp,valtrue) 