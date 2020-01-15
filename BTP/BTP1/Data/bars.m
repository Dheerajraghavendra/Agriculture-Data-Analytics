fid = fopen('advices1.txt');
a = textscan(fid,'%s','delimiter','\n','whitespace', '');
[x,y]=size(a{1});
for i = 1:x
    date{i} = a{1}{i}(1:10);
    farm{i} = a{1}{i}(12:57);  
    crop{i} = strread(a{1}{i}(58:end),'%s %*[^\n]');
    obs{i} = a{1}{i}(61:end);
end
fileID = fopen('prob.txt');
pb = textscan(fileID,'%s %s %s %s %s','Delimiter',';');
fclose(fileID);
[x1,y1] = size(pb{1})
hg=zeros(x1,1);
hg = zeros(x1,6);
for j = 1:x
    s = obs{j};
    f=0;
    for i=1:x1
         k = strfind(lower(s),pb{1}{i});
         if ~isempty(k)
             hg(i,str2num(date{j}(3:4))-11) = hg(i,str2num(date{j}(3:4))-11)+1;
             %break;
             f=1;
         end
         k = strfind(lower(s),pb{2}{i});
         if ~isempty(k) && ~f
             hg(i,str2num(date{j}(3:4))-11) = hg(i,str2num(date{j}(3:4))-11)+1;
             f=1;
             %break;
         end
         k = strfind(lower(s),pb{3}{i});
         if ~isempty(k) && ~f
             hg(i,str2num(date{j}(3:4))-11) = hg(i,str2num(date{j}(3:4))-11)+1;
             f=1;
            % break;
         end
         k = strfind(lower(s),pb{4}{i});
         if ~isempty(k) && ~f
             hg(i,str2num(date{j}(3:4))-11) = hg(i,str2num(date{j}(3:4))-11)+1;
             f=1;
           %  break;
         end
         k = strfind(lower(s),pb{5}{i});
         if ~isempty(k) && ~f
             hg(i,str2num(date{j}(3:4))-11) = hg(i,str2num(date{j}(3:4))-11)+1;
             f=1;
             %break;
         end
    end
end

l=sum(sum(hg));
%edges = [1:38];
%bar1=histogram(hg(:,1),edges)
bar(hg(:,1)+hg(:,2)+hg(:,3)+hg(:,4)+hg(:,5)+hg(:,6),'r');
hold on;
bar(hg(:,1)+hg(:,2)+hg(:,3)+hg(:,4)+hg(:,5),'g');
hold on;
bar(hg(:,1)+hg(:,2)+hg(:,3)+hg(:,4),'b');
hold on;
bar(hg(:,1)+hg(:,2)+hg(:,3),'c');
hold on;
bar(hg(:,1)+hg(:,2),'m');
hold on;
bar(hg(:,1),'y');
set(gca,'xtick',1:38,'XTickLabel',pb{1},'xticklabelrotation',90);
legend('2017','2016','2015','2014','2013','2012');
xlabel('Crop condition');
ylabel('Number of observations');
title('Year wise problem count')
%{
ind = zeros(length(h));
h2 = sort(h,'descend');
for i = 1:size(h)
    for j = 1:size(h)
        if (ind(j)==0)&&(h2(i)==h(j))
            names2{i} = pb{1}{j};
            ind(j) = 1;
            break;
        end
    end
end
%explode = zeros(1,38)+1; 
h2=h2';
pie(h2);
newa = h2./max(h2).*100;
th = pie(newa);
thresh = 5;
tmp = round(newa) <= thresh;
tmp = reshape([tmp;zeros(1,length(tmp))],1,[]);
delete(th([find(tmp)+1]))
legend(names2)
%}
year = zeros(6,1);
for i = 1:x
    year(str2num(date{i}(3:4))-11) = year(str2num(date{i}(3:4))-11)+1;
end