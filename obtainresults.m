close all
clear
clc

load('./alldata/pathallpatchcsmall.mat');

% 
trainfeature = load('./Results/predicttrain.txt');
testfeature = load('./Results/predicttest.txt');



trainfeature = trainfeature(:,2);
testfeature = testfeature(:,2);

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


