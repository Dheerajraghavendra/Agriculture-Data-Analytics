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
h=zeros(x1,1);
for j = 1:x
    s = obs{j};
    f=0;
    for i=1:x1
         k = strfind(lower(s),pb{1}{i});
         if ~isempty(k)
             h(i) = h(i)+1;
             %break;
             f=1;
         end
         k = strfind(lower(s),pb{2}{i});
         if ~isempty(k) && ~f
             h(i) = h(i)+1;
             f=1;
             %break;
         end
         k = strfind(lower(s),pb{3}{i});
         if ~isempty(k) && ~f
             h(i) = h(i)+1;
             f=1;
            % break;
         end
         k = strfind(lower(s),pb{4}{i});
         if ~isempty(k) && ~f
             h(i) = h(i)+1;
             f=1;
           %  break;
         end
         k = strfind(lower(s),pb{5}{i});
         if ~isempty(k) && ~f
             h(i) = h(i)+1;
             f=1;
             %break;
         end
    end
end
l=sum(h);
FID = fopen('hist.txt','w');
%formatspec = '%s %d \r\n';
for i = 1:x1
%    fprintf(FID,formatspec,pb{1}{i},h(i));
    fprintf(FID,'%s ',pb{1}{i});
    fprintf(FID,'%d \r\n',h(i));
end
j=1;
for i=1:size(h)
    if h(i)>1
        l(j,1)=h(i);
        name{j} = pb{1}{i};
        j=j+1;
    end
end
%{
bar(h,0.2)
set(gca,'xtick',1:39,'XTickLabel',pb{1});
pie(h,pb{1});
%}
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